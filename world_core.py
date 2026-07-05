"""
world_core.py — the pure render core, shared by every interface.

world(seed, x, y, t) -> layers. This module owns everything between
worldgen.py's static fields and pixels on a screen, delegating each concern
to a focused module and re-exporting the public surface, so consumers keep
importing only `world_core`:

    worldgen.py    static field producers: windowed noise, D8 hydrology
    common.py      the time model + shared grid helpers
    hydrology.py   the vector river tree, stroking, window-local streams
    lighting.py    sun field, terrain normals, terrain & cloud shadows
    history.py     the M3 history CA, integrated once into a timeline
    ecosim.py      the stateful M4 EcoSim (forward-only, has memory)

The layer stack, each reading only the layers above it (per DESIGN.md):

    fields    elevation, temperature(t), moisture           noise + gradient
    climate   clouds(t)                                      advected noise
    water     D8 flow accumulation                           flow algorithm
    biome     lookup(elevation, temperature, moisture)       table
    ecology   flora(t)  -> fauna(t)                          field + Lotka-Volterra
    society   civilization(t), history(t)                    history CA (M3)

Seekability, the one invariant: render(t) is STATELESS. Fields that are
algebraic in t (clouds, flora, fauna) are pure functions. The M3 history —
which genuinely needs state, because "a plague starts on day 40 and spreads"
is by definition non-seekable — is INTEGRATED ONCE at build time into a
keyframed timeline; render(t) then only INTERPOLATES that timeline. So the
simulation is stateful but its consumption is not, and the exporter still
samples any t on demand.

Consumers: world_viewer.py (desktop matplotlib), web/index.html (same code
in the browser via Pyodide), godot_bridge.py (chunk streaming). No
matplotlib, no PIL, no I/O — numpy in, data/pixels out.
"""

import numpy as np

from worldgen import (elevation_field, moisture_field, compute_hydrology,
                      elevation_window, moisture_window, noise_window,
                      BIOME_COLORS)
from common import YEAR_DAYS, TIDE_PERIOD, FAUNA_PERIOD, smoothstep
from hydrology import (BROOK_MIN, _net_threshold, _extract_network,
                       _stroke_field, _local_streams)
from lighting import NORMAL_RELIEF_WORLD, SHADOW_RELIEF_WORLD, _lighting_fields
from history import (HIST_SIZE, build_history, civ_population,
                     _sample_history)
from ecosim import EcoSim


def _cloud_sheet(seed, cx, cy, span, size, base_period):
    """A [0,1]-normalized noise sheet for cloud cover, windowable like the
    terrain fields (so clouds stay coherent under pan/zoom)."""
    f = noise_window(seed, cx, cy, span, size, base_period=base_period)
    return ((f - f.min()) / (np.ptp(f) or 1.0)).astype(np.float32)


LAYERS = [
    ("composite", "World"),
    ("elevation", "Elevation"),
    ("temperature", "Temperature"),
    ("moisture", "Moisture"),
    ("clouds", "Clouds"),
    ("flow", "Flow"),
    ("biome", "Biomes"),
    ("flora", "Flora"),
    ("fauna", "Fauna"),
    ("civ", "Civilization"),
    ("history", "History"),
    ("light", "Daylight"),
]

BIOME_NAMES = list(BIOME_COLORS.keys())
BIOME_LUT = np.array([BIOME_COLORS[n] for n in BIOME_NAMES], dtype=np.float32)
BID = {n: i for i, n in enumerate(BIOME_NAMES)}

# distinct faction hues (RGB); world supports up to this many civilizations
CIV_COLORS = np.array([
    (214,  69,  65),   # crimson
    (232, 184,  58),   # gold
    ( 58, 176, 168),   # teal
    (150,  96, 210),   # violet
    (232, 128,  52),   # orange
    ( 96, 178,  84),   # green
], dtype=np.float32)


def biome_ids(e, t, m, sea, tide=0.0, lake_lv=None):
    """Vectorized biome classification -> int16 ids.

    `sea` is the GEOGRAPHIC sea level (the slider, no instantaneous tide), so
    the coastline/beach terrain only moves when you change the slider. `tide`
    is the tidal amplitude: the beach is the intertidal band it sweeps, so sand
    is drawn as wide as the tide reaches. The instantaneous waterline (which
    covers/uncovers that sand each cycle) is applied as a wet overlay in
    render(), not here — the sand geography itself stays put.

    `lake_lv` is the standing-water surface from the hydrology (0 = no lake):
    cells under it are water too, and the refined window elevation decides the
    shoreline, so lake edges gain fractal detail with zoom like coasts do."""
    lo = sea - tide                      # lowest the water ever falls (geo)
    beach_top = sea + tide + 0.015       # intertidal band + a little dry sand
    water = e < lo
    if lake_lv is not None:
        water = water | (e < lake_lv)
    conds = [
        water & (e < lo - 0.12),
        water & (e >= lo - 0.05),
        water,
        e > 0.92,
        e > 0.82,
        e < beach_top,
        t < 0.2,
        t < 0.35,
        t < 0.5,
        m < 0.25,
        m < 0.45,
        m < 0.65,
        m < 0.82,
    ]
    choices = [BID[n] for n in (
        "deep_ocean", "shallow", "ocean", "high_peak", "mountain", "beach",
        "snow", "tundra", "taiga", "desert", "savanna", "grassland", "forest")]
    return np.select(conds, choices, default=BID["jungle"]).astype(np.int16)


def temperature_t(elev, lat, lat_signed, sea_eff, season_off):
    t = lat + season_off * lat_signed - np.clip(elev - sea_eff, 0, None) * 0.9
    return np.clip(t, 0, 1)


def color_ramp(v, stops, colors):
    """v in [0,1] -> RGB via piecewise-linear ramp. colors: (k,3)."""
    colors = np.asarray(colors, dtype=np.float32)
    return np.stack([np.interp(v, stops, colors[:, c]) for c in range(3)], axis=-1)


TMP_RAMP = ([0.0, 0.5, 1.0], [(58, 103, 196), (232, 228, 216), (200, 80, 46)])
MOI_RAMP = ([0.0, 0.5, 1.0], [(201, 163, 90), (214, 214, 196), (62, 143, 122)])
FLOW_RAMP = ([0.0, 0.5, 1.0], [(12, 18, 30), (32, 110, 150), (180, 235, 250)])
FLORA_RAMP = ([0.0, 0.35, 0.7, 1.0],
              [(120, 96, 60), (150, 150, 70), (70, 150, 60), (24, 96, 40)])


# ===========================================================================
# Ecology fields — pure functions of the static layers + t (the FAR form).
# ===========================================================================
def flora_field(ws, tf, sea_eff):
    """Vegetation density in [0,1]: warmth x water, no ocean, taper at peaks."""
    e, m = ws.elev, ws.moist
    warmth = np.clip((tf - 0.26) / 0.55, 0, 1)          # needs above-freezing
    wet = smoothstep((m - 0.1) / 0.8)
    veg = warmth * (0.30 + 0.70 * wet)
    veg *= np.clip(1 - (e - 0.80) / 0.18, 0, 1)         # thin out toward peaks
    veg *= (e >= sea_eff)                                # nothing grows in sea
    return np.clip(veg, 0, 1).astype(np.float32)


def fauna_field(flora, t):
    """Herbivore & predator biomass as a Lotka-Volterra LIMIT CYCLE."""
    phase = 2 * np.pi * t / FAUNA_PERIOD
    prey_osc = 0.62 + 0.38 * np.sin(phase)
    pred_osc = 0.55 + 0.38 * np.sin(phase - np.pi / 2)   # quarter-cycle lag
    herbivore = flora * prey_osc
    predator = flora * flora * pred_osc
    return herbivore.astype(np.float32), predator.astype(np.float32)


def clouds_field(ws, t):
    """Cloud cover in [0,1]: two noise sheets advected by the wind, gated by
    moisture, piled up on windward slopes. Pure, seekable function of t."""
    size = ws.elev.shape[0]
    ox1 = int((t * size / 6.0)) % size
    ox2 = int((t * size / 11.0)) % size
    oy2 = int((t * size / 40.0)) % size
    c1 = np.roll(ws.cloud1, ox1, axis=1)
    c2 = np.roll(np.roll(ws.cloud2, ox2, axis=1), oy2, axis=0)
    sheet = 0.6 * c1 + 0.4 * c2
    density = sheet * (0.45 + 0.55 * ws.moist) + 0.35 * ws.orographic
    return smoothstep((density - 0.42) / 0.35).astype(np.float32)


# ===========================================================================
class WorldSlice:
    """Static per-resolution data + grids (full res, or strided for thumbs)."""

    def __init__(self, elev, moist, hyd, cloud1, cloud2, civ_count, seed,
                 cx=0.5, cy=0.5, span=1.0):
        self.size = elev.shape[0]
        self.seed = int(seed)
        self.planet_size = self.size
        # viewport on the unit torus: window centered at (cx, cy) of side span.
        # span == 1 -> the whole planet (the default, backward-compatible view).
        self.cx, self.cy, self.span = float(cx), float(cy), float(span)
        self.accum = hyd["accum"]                  # raw planetary accumulation
        self.lake_level = hyd["lake_level"]
        self.lake_lv = self.lake_level             # this window == the planet
        # trace the drainage into the vector TREE once; every view (including
        # this planet view) rasterizes that same world geometry.
        self.net = _extract_network(hyd["parent"], self.accum, self.size,
                                    _net_threshold(self.size), hyd["sea"])
        alpha, disc, carve = _stroke_field(self.net, self.cx, self.cy,
                                           self.span, self.size,
                                           self.planet_size, self.seed)
        self.river_alpha, self.river_disc = alpha, disc
        self.brook_alpha = np.zeros_like(alpha)    # local streams: zoomed only
        self.elev = np.clip(elev - carve, 0.0, 1.0)  # valleys under the rivers
        self.moist = moist
        self.cloud1, self.cloud2 = cloud1, cloud2
        self._derive_grids()
        self.has_history = False
        build_history(self, civ_count, seed)

    def _derive_grids(self):
        """Per-pixel grids from the window's WORLD coordinates (so latitude,
        longitude and hillshade are correct at any pan/zoom, not pixel-relative)."""
        size = self.size
        step = (np.arange(size, dtype=np.float32) / size - 0.5) * self.span
        v = (self.cy + step) % 1.0                 # world y per row
        u = (self.cx + step) % 1.0                 # world x per column
        self.lat = (1 - np.abs(v - 0.5) * 2)[:, None]
        self.lat_signed = ((0.5 - v) * 2)[:, None]
        self.xn = u
        self.pixel_world = max(self.span / size, 1e-6)
        gy, gx = np.gradient(self.elev, self.pixel_world, self.pixel_world)
        nx = -gx * NORMAL_RELIEF_WORLD
        ny = -gy * NORMAL_RELIEF_WORLD
        nz = np.ones_like(self.elev, np.float32)
        inv = 1.0 / np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
        self.normal = np.stack((nx * inv, ny * inv, nz * inv), axis=2).astype(np.float32)
        self.height = (self.elev * SHADOW_RELIEF_WORLD).astype(np.float32)
        slope = np.sqrt(nx * nx + ny * ny)
        self.shade = np.clip(1.05 - slope * 0.28, 0.74, 1.08).astype(np.float32)
        # the flow ramp is normalized by a PLANET-WIDE constant, not the window
        # max: otherwise the whole flow map re-brightens whenever a big river
        # enters or leaves the view (one of the "pixels changed" flickers).
        if not hasattr(self, "log_norm"):
            self.log_norm = float(np.log1p(max(float(self.accum.max()), 1.0)))
        self.log_accum = np.clip(np.log1p(np.maximum(self.accum, 0))
                                 / self.log_norm, 0, 1).astype(np.float32)
        self.orographic = np.clip(gx * 6.0, 0, 1).astype(np.float32)

    def _sample_planet(self, field, cx, cy, span, size):
        """Sample a planet-resolution field at the window's per-pixel world
        coordinates (nearest, wrapping the torus) -> a size x size window."""
        step = (np.arange(size, dtype=np.float32) / size - 0.5) * span
        ci = (((cx + step) % 1.0) * field.shape[1]).astype(np.int64) % field.shape[1]
        ri = (((cy + step) % 1.0) * field.shape[0]).astype(np.int64) % field.shape[0]
        return field[np.ix_(ri, ci)]

    def _sample_planet_linear_grid(self, width, height, cx, cy, span, size):
        step = (np.arange(size, dtype=np.float32) / size - 0.5) * span
        fx = ((cx + step) % 1.0) * width
        fy = ((cy + step) % 1.0) * height
        x0 = np.floor(fx).astype(np.int64) % width
        y0 = np.floor(fy).astype(np.int64) % height
        x1 = (x0 + 1) % width
        y1 = (y0 + 1) % height
        tx = (fx - np.floor(fx)).astype(np.float32)[None, :]
        ty = (fy - np.floor(fy)).astype(np.float32)[:, None]
        return x0, x1, y0, y1, tx, ty

    def _sample_planet_linear(self, field, cx, cy, span, size, grid=None):
        """Sample a planet-resolution float field with wrapped bilinear
        filtering. This is the fast crop path used for streaming windows that
        are still above the planet's native cell size."""
        if grid is None:
            grid = self._sample_planet_linear_grid(field.shape[1], field.shape[0],
                                                   cx, cy, span, size)
        x0, x1, y0, y1, tx, ty = grid
        f00 = field[np.ix_(y0, x0)].astype(np.float32)
        f01 = field[np.ix_(y0, x1)].astype(np.float32)
        f10 = field[np.ix_(y1, x0)].astype(np.float32)
        f11 = field[np.ix_(y1, x1)].astype(np.float32)
        top = f00 * (1.0 - tx) + f01 * tx
        bot = f10 * (1.0 - tx) + f11 * tx
        return (top * (1.0 - ty) + bot * ty).astype(np.float32)

    def _share_world_state(self, other):
        other.has_history = self.has_history
        for k in ("hist_days", "hist_pop", "hist_own", "hist_stress",
                  "hist_unrest", "civ_cores", "eco"):
            if hasattr(self, k):
                setattr(other, k, getattr(self, k))

    def stream_view(self, cx, cy, zoom, size=None):
        """Return a fast sampled crop for realtime streaming.

        For zoom levels that are still above the planet's native cell size, this
        reuses the already-built planet fields and samples them directly at the
        requested output resolution. Deep zoom still falls back to the refined
        window generator so sub-cell terrain detail and local streams remain
        available when you actually need them."""
        zoom = max(1.0, float(zoom))
        span = 1.0 / zoom
        size = self.size if size is None else max(8, int(size))
        if span >= 0.999 and size == self.size:
            return self
        if zoom > self.planet_size or size > self.size:
            return self.view(cx, cy, zoom)

        # PIXEL-SNAP the streaming window to the output tile lattice. Each output
        # tile spans (span / size) in world units; quantizing the center to that
        # lattice means consecutive frames either reuse the same crop or translate
        # by whole tiles, so panning slides instead of shimmering. Without this,
        # every sub-tile pan re-samples the planet at a shifted phase and the whole
        # chunk pops by a fraction of a tile on each fetch.
        step = span / size
        cx = round((cx % 1.0) / step) * step
        cy = round((cy % 1.0) / step) * step

        key = (round(cx % 1.0, 9), round(cy % 1.0, 9), round(zoom, 6), size)
        cached = getattr(self, "_stream_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        s = WorldSlice.__new__(WorldSlice)
        s.size, s.seed = size, self.seed
        s.planet_size = self.planet_size
        s.cx, s.cy, s.span = float(cx % 1.0), float(cy % 1.0), span
        grid = self._sample_planet_linear_grid(self.elev.shape[1], self.elev.shape[0],
                                               s.cx, s.cy, span, size)
        s.elev = np.clip(self._sample_planet_linear(self.elev, s.cx, s.cy, span, size, grid),
                         0.0, 1.0)
        s.moist = np.clip(self._sample_planet_linear(self.moist, s.cx, s.cy, span, size, grid),
                          0.0, 1.0)
        s.accum = np.maximum(self._sample_planet_linear(self.accum, s.cx, s.cy, span, size, grid),
                             0.0)
        s.river_alpha = np.clip(self._sample_planet_linear(
            self.river_alpha, s.cx, s.cy, span, size, grid), 0.0, 1.0)
        s.river_disc = np.maximum(self._sample_planet_linear(
            self.river_disc, s.cx, s.cy, span, size, grid), 0.0)
        s.lake_lv = np.maximum(self._sample_planet_linear(
            self.lake_level, s.cx, s.cy, span, size, grid), 0.0)
        s.brook_alpha = np.zeros_like(s.river_alpha)
        s.net = self.net
        s.lake_level = self.lake_level
        s.cloud1 = np.clip(self._sample_planet_linear(self.cloud1, s.cx, s.cy, span, size, grid),
                           0.0, 1.0)
        s.cloud2 = np.clip(self._sample_planet_linear(self.cloud2, s.cx, s.cy, span, size, grid),
                           0.0, 1.0)
        s.log_norm = self.log_norm
        s._derive_grids()
        self._share_world_state(s)
        self._stream_cache = (key, s)
        return s

    # ---- a cheap re-sampled window that SHARES this planet's history ---------
    def view(self, cx, cy, zoom):
        """Return a new WorldSlice looking at window (cx, cy, side 1/zoom).
        Fields are recomputed for the window; the (global) history timeline,
        faction cores and cloud sheets are reused from this planet by reference,
        so panning/zooming never re-runs the M3 CA."""
        zoom = max(1.0, float(zoom))
        span = 1.0 / zoom
        if span >= 0.999:                          # whole planet -> this slice
            return self
        size = self.size
        # PIXEL-SNAP the viewport: quantize the window center to the drainage
        # lattice at this zoom, so panning translates the exact same world
        # samples instead of re-sampling at shifted phases. Without this every
        # sub-pixel pan rebuilds a slightly different D8 network and the rivers
        # shimmer; with it the drainage (and all fields) just slide. The lattice
        # is the STRIDED grid the local D8 runs on, so one pan step = one whole
        # drainage cell = a pure translation of the network.
        st = max(1, size // 128)                   # D8 stride (cost + stability)
        step = (span / size) * st
        cx = round(cx / step) * step
        cy = round(cy / step) * step
        # drag frames usually snap to the same lattice cell -> reuse the window
        key = (round(cx / step), round(cy / step), round(zoom, 6))
        cached = getattr(self, "_view_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        s = WorldSlice.__new__(WorldSlice)
        s.size, s.seed = size, self.seed
        s.planet_size = self.size
        s.cx, s.cy, s.span = float(cx % 1.0), float(cy % 1.0), span
        s.elev = elevation_window(size, self.seed, s.cx, s.cy, span)
        s.moist = moisture_window(size, self.seed, s.cx, s.cy, span)
        # rivers: rasterize the planet's vector TREE into this window — the
        # same world-space geometry at every pan/zoom (slides rigidly, no
        # shimmer), refined with deterministic meanders at this zoom, and its
        # valleys carved into the refined elevation so terrain agrees.
        alpha, disc, carve = _stroke_field(self.net, cx, cy, span, size,
                                           self.planet_size, self.seed)
        s.elev = np.clip(s.elev - carve, 0.0, 1.0)
        s.river_alpha, s.river_disc = alpha, disc
        s.lake_lv = self._sample_planet(self.lake_level, cx, cy, span, size)
        s.net = self.net
        s.lake_level = self.lake_level
        s.brook_alpha = np.zeros_like(alpha)
        s.accum = disc
        if span <= 0.14:
            # deep zoom: the planet-res drainage has run out of detail, so run
            # REAL local drainage on the refined, carved window elevation. The
            # carved trunks / lakes / sea are the drains it empties into.
            s.brook_alpha, acc_local = _local_streams(
                s.elev, s.moist, alpha, s.lake_lv, cx, cy, span, self.seed)
            # flow layer: lift local cell counts so a just-visible brook reads
            # like the smallest vector river (display scaling only)
            lift = _net_threshold(self.planet_size) / BROOK_MIN
            s.accum = np.maximum(disc, acc_local * lift)
        s.cloud1 = _cloud_sheet(self.seed + 4001, s.cx, s.cy, span, size, 4)
        s.cloud2 = _cloud_sheet(self.seed + 8009, s.cx, s.cy, span, size, 3)
        s.log_norm = self.log_norm             # planet-fixed flow normalization
        s._derive_grids()
        # share the pre-integrated history + the live eco sim (both global, not
        # window-local) by reference, so zooming never rebuilds or forks them
        self._share_world_state(s)
        self._view_cache = (key, s)
        return s

    # ---- downsample a fine field to the coarse HIST grid (nearest) ----------
    def _coarse(self, field):
        idx = (np.arange(HIST_SIZE) * self.size) // HIST_SIZE
        return field[np.ix_(idx, idx)].astype(np.float32)

    def strided(self, st):
        s = WorldSlice.__new__(WorldSlice)
        size = self.size
        for k, v in self.__dict__.items():
            # stride any axis at render resolution (elev, lat, xn, ...); leave
            # the coarse HIST timeline (no axis == size) shared by reference
            if isinstance(v, np.ndarray) and any(d == size for d in v.shape):
                sl = tuple(slice(None, None, st) if d == size else slice(None)
                           for d in v.shape)
                setattr(s, k, v[sl])
            else:
                setattr(s, k, v)
        s.size = s.elev.shape[0]
        # striding a 1px stroke field would leave dotted rivers; re-rasterize
        # the vector network at the thumbnail's own resolution instead
        if getattr(self, "net", None):
            a, d, _ = _stroke_field(self.net, self.cx, self.cy, self.span,
                                    s.size, self.planet_size, self.seed)
            s.river_alpha, s.river_disc = a, d
            s.brook_alpha = np.zeros_like(a)
        return s


def build_world(seed, size, civ_count=3):
    """The heavy, seed-only part: fields, hydrology, cloud sheets, M3 history."""
    elev = elevation_field(size, seed).astype(np.float32)
    moist = moisture_field(size, seed).astype(np.float32)
    hyd = compute_hydrology(elev, int(seed))   # fill -> D8 -> accum -> lakes
    cloud1 = _cloud_sheet(seed + 4001, 0.5, 0.5, 1.0, size, 4)
    cloud2 = _cloud_sheet(seed + 8009, 0.5, 0.5, 1.0, size, 3)
    ws = WorldSlice(elev, moist, hyd,
                    cloud1, cloud2, int(civ_count), int(seed))
    ws.eco = EcoSim(ws, int(seed))           # stateful vitality sim (M4)
    return ws


def default_river_threshold(size):
    """350 is tuned for 512²; scale down for smaller worlds."""
    return round(350 * (size / 512) ** 1.5)


def frame_params(t, sea_level, tide_amp, season_amp):
    sea_eff = sea_level + tide_amp * np.sin(2 * np.pi * t / TIDE_PERIOD)
    season_off = season_amp * np.sin(2 * np.pi * t / YEAR_DAYS)
    sun_x = t % 1.0
    return sea_eff, season_off, sun_x


def _terrain_gray(biome_id):
    """Dim grayscale terrain base for the society layers, from precomputed ids."""
    base = BIOME_LUT[biome_id]
    gray = base @ np.array([0.30, 0.59, 0.11], np.float32)
    return np.repeat(gray[..., None], 3, axis=2) * 0.55 + 18


def _city_dots(ws, img, t):
    size = ws.size
    span, left, top = ws.span, (ws.cx - ws.span / 2), (ws.cy - ws.span / 2)
    for (yn, xn, f, t0) in ws.civ_cores:
        if t <= t0:
            continue
        # world -> window pixel, wrapping the torus; skip cores off-window
        fx = ((xn - left) % 1.0) / span
        fy = ((yn - top) % 1.0) / span
        if fx >= 1.0 or fy >= 1.0:
            continue
        cx, cy = int(fx * size), int(fy * size)
        r = max(1, size // 200) + (span < 0.999)     # a touch bigger when zoomed
        img[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = (250, 250, 235)


def state(ws, t, sea_level, river_thr, season_amp, tide_amp, day_night):
    """Compute the per-tile world state ONCE — the data every view (and, later,
    Godot) reads. All field math lives here; the RGB renderers below are thin
    skins over this dict. Shared fields (temperature, biome ids, vegetation,
    clouds, normals, shadow masks) are computed a single time instead of being
    re-derived per layer."""
    sea_eff, season_off, sun_x = frame_params(t, sea_level, tide_amp, season_amp)
    e = ws.elev
    tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
    clouds = clouds_field(ws, t)
    st = {
        "ws": ws, "t": t, "sea_level": sea_level, "river_thr": river_thr,
        "tide_amp": tide_amp, "day_night": day_night,
        "sea_eff": sea_eff, "season_off": season_off, "sun_x": sun_x,
        "e": e, "land": e >= sea_eff, "tf": tf,
        "biome_id": biome_ids(e, tf, ws.moist, sea_level, tide_amp,
                              getattr(ws, "lake_lv", None)),
        "veg": flora_field(ws, tf, sea_eff),
        "moist": ws.moist, "log_accum": ws.log_accum,
        "clouds": clouds,
    }
    st.update(_lighting_fields(ws, sun_x, season_off, day_night, clouds))
    lake_lv = getattr(ws, "lake_lv", None)
    if lake_lv is not None:
        st["veg"] = np.where(e < lake_lv, 0.0, st["veg"]).astype(np.float32)
    # the stateful ecosystem's DEVIATION from the seekable baseline: health in
    # [0,1] (1 = undamaged) plus the burn/salt scar. The biotic layers below are
    # baseline x health, so an undisturbed world matches the pure fields and a
    # flooded/burned one carries its scars. (Absent eco -> health 1, no scar.)
    eco = getattr(ws, "eco", None)
    if eco is not None:
        es = eco.sample(ws)
        clim = es["clim"] + 1e-3
        st["veg_health"] = np.clip(es["veg"] / clim, 0, 1)
        st["fauna_health"] = np.clip(es["fauna"] / (0.6 * clim), 0, 1)
        st["civ_health"] = np.clip(es["civ"], 0, 1)
        st["scorch"] = np.clip(es["scorch"], 0, 1)
    else:
        one = np.ones_like(e)
        st["veg_health"] = st["fauna_health"] = one
        st["civ_health"] = st["scorch"] = np.zeros_like(e)
    return st


def colorize(st, layer):
    """Pure skin: a state dict + a layer name -> uint8 RGB. No field math here."""
    ws, t = st["ws"], st["t"]
    e, land, sea_eff = st["e"], st["land"], st["sea_eff"]
    sun_x, tf = st["sun_x"], st["tf"]
    sea_level, tide_amp = st["sea_level"], st["tide_amp"]
    river_thr, day_night = st["river_thr"], st["day_night"]

    if layer == "elevation":
        img = np.empty(e.shape + (3,), np.float32)
        sea_m = e < sea_eff
        f = np.clip(e / max(sea_eff, 1e-6), 0, 1)
        img[..., 0] = 16 + 70 * f
        img[..., 1] = 34 + 106 * f
        img[..., 2] = 78 + 108 * f
        g = np.clip((e - sea_eff) / max(1 - sea_eff, 1e-6), 0, 1)
        land_rgb = color_ramp(g, [0.0, 0.55, 1.0],
                              [(88, 140, 80), (168, 150, 96), (245, 245, 248)])
        img[~sea_m] = land_rgb[~sea_m]
    elif layer == "temperature":
        img = color_ramp(tf, *TMP_RAMP)
    elif layer == "moisture":
        img = color_ramp(ws.moist, *MOI_RAMP)
    elif layer == "flow":
        img = color_ramp(ws.log_accum, *FLOW_RAMP)
    elif layer == "clouds":
        base = BIOME_LUT[st["biome_id"]] * 0.45
        cov = st["clouds"][..., None]
        img = base * (1 - cov) + np.array([242, 246, 250], np.float32) * cov
        img *= (0.30 + 0.70 * st["sunlight"])[..., None]
    elif layer == "flora":
        # LIVING vegetation = climatic baseline x ecosystem health, then scars.
        veg = st["veg"] * st["veg_health"]
        img = color_ramp(veg, *FLORA_RAMP)
        sc = st["scorch"][..., None]
        img = img * (1 - sc) + np.array([70, 54, 44], np.float32) * sc   # burn/salt scar
        img[~land] = (26, 42, 74)
    elif layer == "fauna":
        # LIVING game = baseline biomass x ecosystem health, minus settlement pressure.
        fh = st["fauna_health"]
        herb, pred = fauna_field(st["veg"], t)
        herb, pred = herb * fh, pred * fh
        civ_p, _ = civ_population(ws, t)
        herb = herb * (1 - 0.7 * np.clip(civ_p, 0, 1))     # settlers hunt/clear game
        img = np.empty(e.shape + (3,), np.float32)
        img[..., 0] = 40 + 205 * np.clip(pred * 1.6, 0, 1)
        img[..., 1] = 40 + 175 * np.clip(herb, 0, 1)
        img[..., 2] = 45 + 30 * np.clip(herb, 0, 1)
        img[~land] = (20, 32, 58)
    elif layer == "civ":
        # territory/population from the M3 history, DIMMED where the living
        # ecosystem has collapsed (shortage/flood) and charred where scorched.
        img = _terrain_gray(st["biome_id"])
        pop, fid = civ_population(ws, t)
        has = fid >= 0
        if has.any():
            tint = CIV_COLORS[np.clip(fid, 0, len(CIV_COLORS) - 1)]
            a = (np.clip(pop * 1.4, 0, 0.9) * (0.3 + 0.7 * st["civ_health"]))[..., None]
            img = np.where(has[..., None], img * (1 - a) + tint * a, img)
        sc = st["scorch"][..., None]
        img = img * (1 - sc) + np.array([54, 40, 34], np.float32) * sc
        _city_dots(ws, img, t)
    elif layer == "history":
        # the chronicle: territory + where the world is thriving / in conflict / in shortage
        img = _terrain_gray(st["biome_id"])
        pop, fid, stress, unrest = _sample_history(ws, t)
        has = fid >= 0
        if has.any():
            tint = CIV_COLORS[np.clip(fid, 0, len(CIV_COLORS) - 1)]
            a = np.clip(pop * 1.3, 0, 0.85)[..., None]
            img = np.where(has[..., None], img * (1 - a) + tint * a, img)
        conflict = np.clip(unrest * (pop > 0.02), 0, 1)[..., None]     # red fronts
        img = img * (1 - conflict) + np.array([235, 60, 45], np.float32) * conflict
        short = np.clip(stress * land * (0.4 + pop), 0, 1)[..., None]  # violet shortage
        img = img * (1 - short) + np.array([150, 70, 200], np.float32) * short
        _city_dots(ws, img, t)
    elif layer == "light":
        l = st["sunlight"]
        sh = (1.0 - st["terrain_shadow"]) * st["sun_up"]
        ch = (1.0 - st["cloud_shadow"]) * st["sun_up"]
        img = np.empty(e.shape + (3,), np.float32)
        img[..., 0] = (44 + 211 * l) * (1.0 - 0.18 * ch)
        img[..., 1] = (40 + 204 * l) * (1.0 - 0.26 * ch)
        img[..., 2] = 62 + 154 * l + 34 * sh
    else:  # biome / composite
        # geography is cut by the SLIDER sea level (+ tidal band), so the sand
        # stays put; the instantaneous tide only sweeps a waterline across it.
        img = BIOME_LUT[st["biome_id"]].copy()
        lo = sea_level - tide_amp                       # permanent low-tide line
        wet = (e >= lo) & (e < sea_eff)                 # intertidal sand, now wet
        img[wet] = (70, 130, 180)
        if layer == "composite":
            img *= ws.shade[..., None]
            l = st["sunlight"]
            img[..., 0] *= l
            img[..., 1] *= l * 0.96 + 0.04
            img[..., 2] *= l * 0.84 + 0.16
            ra = getattr(ws, "river_alpha", None)
            if ra is not None:
                # the slider fades tributaries below the threshold instead of
                # hard-cutting them, so the dendritic tree stays readable
                gate = np.clip(ws.river_disc / max(river_thr, 1.0), 0, 1) ** 0.6
                a = ra * gate
                ba = getattr(ws, "brook_alpha", None)
                if ba is not None:
                    a = np.maximum(a, ba)
                lake_lv = getattr(ws, "lake_lv", 0.0)
                a = np.where(land & ~(e < lake_lv), a, 0.0)[..., None]
                depth = np.clip(np.log1p(ws.river_disc) / ws.log_norm,
                                0, 1)[..., None]
                wat = (np.array([86, 148, 205], np.float32) * (1 - depth)
                       + np.array([30, 80, 150], np.float32) * depth)
                img = img * (1 - a) + wat * a
    return np.clip(img, 0, 255).astype(np.uint8)


def render(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night):
    """Convenience: compute state for one frame and skin a single layer."""
    return colorize(state(ws, t, sea_level, river_thr, season_amp, tide_amp,
                          day_night), layer)


def render_rgba_bytes(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night):
    """Same as render(), packed as (width, RGBA bytes) for canvas blitting."""
    img = render(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night)
    h, w, _ = img.shape
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = img
    rgba[..., 3] = 255
    return w, rgba.tobytes()
