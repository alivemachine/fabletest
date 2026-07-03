"""
world_core.py — the pure render core, shared by every interface.

world(seed, x, y, t) -> layers. This module owns everything between
worldgen.py's static fields and pixels on a screen.

The layer stack, each reading only the layers above it (per DESIGN.md):

    fields    elevation, temperature(t), moisture           noise + gradient
    climate   clouds(t)                                      advected noise
    water     D8 flow accumulation                           flow algorithm
    biome     lookup(elevation, temperature, moisture)       table
    ecology   flora(t)  -> fauna(t)                          field + Lotka-Volterra
    society   civilization(t)                                habitability + logistic stock

Design decision: every time-dependent layer here is the FAR FORM — a pure,
SEEKABLE function of t (a stock/statistic), never an integrated simulation.
That keeps render(t) stateless so any frame, at any t, exports on its own.
The NEAR FORM (herds, cities, wars as live agents that integrate over time)
is the M4 Resolver and lives elsewhere; see README / the design notes.

Consumers: world_viewer.py (desktop matplotlib) and web/index.html (same code
in the browser via Pyodide). No matplotlib, no PIL, no I/O — numpy in, RGB out.
"""

import numpy as np

from worldgen import (elevation_field, moisture_field, compute_rivers,
                      fractal_noise, BIOME_COLORS)

# ---------------------------------------------------------------------------
# Time model (M2). t is measured in sim DAYS.
# ---------------------------------------------------------------------------
YEAR_DAYS = 96.0          # one year = 96 days
TIDE_PERIOD = 0.52        # ~semi-diurnal tide
FAUNA_PERIOD = 32.0       # predator-prey limit cycle length (days)
CIV_TAU = YEAR_DAYS * 0.30  # settlement growth time constant

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


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def biome_ids(e, t, m, sea):
    """Vectorized version of worldgen.classify_biomes -> int16 ids."""
    water = e < sea
    conds = [
        water & (e < sea - 0.12),
        water & (e >= sea - 0.05),
        water,
        e > 0.92,
        e > 0.82,
        e < sea + 0.015,
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


def daylight_row(xn, sun_x, depth):
    """Per-longitude light factor: 1 at noon, floor at midnight."""
    c = np.cos(2 * np.pi * (xn - sun_x))
    s = np.clip((c + 0.15) / 0.45, 0, 1)
    s = s * s * (3 - 2 * s)
    floor = 1 - 0.72 * depth
    return floor + (1 - floor) * s


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
    """Herbivore & predator biomass as a Lotka-Volterra LIMIT CYCLE.

    Instead of integrating the ODE (which would make t non-seekable) we ride
    its closed orbit directly: both stocks circle their equilibrium, predators
    lagging prey by a quarter period. Amplitude scales with local carrying
    capacity = flora. This is the far-form 'stock' the design doc asks for.
    """
    phase = 2 * np.pi * t / FAUNA_PERIOD
    prey_osc = 0.62 + 0.38 * np.sin(phase)
    pred_osc = 0.55 + 0.38 * np.sin(phase - np.pi / 2)   # quarter-cycle lag
    herbivore = flora * prey_osc
    predator = flora * flora * pred_osc                  # predators need dense prey
    return herbivore.astype(np.float32), predator.astype(np.float32)


def civ_population(ws, t):
    """Per-cell settled population in [0,1] and its faction id (-1 = none).

    Territory, founding day, and carrying capacity are baked into the world at
    build time (annual-mean habitability). Here we only apply logistic growth
    in t: cities appear at their founding day and fill toward capacity. Pure,
    seekable — jump to any t and you get that year's population directly.
    """
    frac = 1.0 / (1.0 + np.exp(-(t - ws.civ_t0) / CIV_TAU))
    frac = np.where(ws.faction_id >= 0, frac, 0.0)
    return (ws.habitability * frac).astype(np.float32), ws.faction_id


def clouds_field(ws, t):
    """Cloud cover in [0,1]: two noise sheets advected by the wind, gated by
    moisture, piled up on windward slopes (orographic lift). Advection is an
    integer roll derived from t, so it is a pure, seekable function of t."""
    size = ws.elev.shape[0]
    ox1 = int((t * size / 6.0)) % size          # fast low sheet, blows east
    ox2 = int((t * size / 11.0)) % size         # slow high sheet
    oy2 = int((t * size / 40.0)) % size
    c1 = np.roll(ws.cloud1, ox1, axis=1)
    c2 = np.roll(np.roll(ws.cloud2, ox2, axis=1), oy2, axis=0)
    sheet = 0.6 * c1 + 0.4 * c2
    density = sheet * (0.45 + 0.55 * ws.moist) + 0.35 * ws.orographic
    return smoothstep((density - 0.42) / 0.35).astype(np.float32)


# ===========================================================================
class WorldSlice:
    """Static per-resolution data + grids (full res, or strided for thumbs)."""

    def __init__(self, elev, moist, accum, cloud1, cloud2, civ_count, seed):
        self.elev, self.moist, self.accum = elev, moist, accum
        self.cloud1, self.cloud2 = cloud1, cloud2
        size = elev.shape[0]
        yn = (np.arange(size, dtype=np.float32) / size)[:, None]
        self.lat = 1 - np.abs(yn - 0.5) * 2
        self.lat_signed = (0.5 - yn) * 2
        self.xn = np.arange(size, dtype=np.float32) / size
        gy, gx = np.gradient(elev)
        self.shade = np.clip(1 - (gx + gy) * 2.2, 0.75, 1.25).astype(np.float32)
        self.log_accum = np.log1p(accum) / np.log1p(accum.max())
        # windward (east-facing) slope, for orographic cloud lift
        self.orographic = np.clip(gx * 6.0, 0, 1).astype(np.float32)
        self._build_civ(civ_count, seed)

    # ---- society: habitability, cores, territory (annual-mean, built once) --
    def _build_civ(self, civ_count, seed):
        size = self.elev.shape[0]
        e, m = self.elev, self.moist
        sea0 = 0.42
        tf0 = temperature_t(e, self.lat, self.lat_signed, sea0, 0.0)  # annual mean
        flora0 = flora_field(self, tf0, sea0)
        # habitability = food (flora) + fresh water (rivers + coast) + mild climate
        water = 0.7 * self.log_accum
        coast = smoothstep(1 - np.abs(e - sea0) / 0.06) * (e >= sea0)
        water = np.clip(water + 0.4 * coast, 0, 1)
        temperate = np.clip(1 - np.abs(tf0 - 0.62) / 0.45, 0, 1)
        land = e >= sea0
        H = (0.42 * flora0 + 0.30 * water + 0.28 * temperate) * land
        H *= np.clip(1 - (e - 0.75) / 0.25, 0, 1)      # avoid high country
        self.civ_habitability_raw = H.astype(np.float32)

        faction_id = np.full((size, size), -1, np.int16)
        civ_t0 = np.zeros((size, size), np.float32)
        habitability = np.zeros((size, size), np.float32)
        cores = []
        if civ_count > 0:
            rng = np.random.default_rng(seed ^ 0x50C1A1)
            order = np.argsort(-H.reshape(-1))
            min_sep = size * 0.13
            for idx in order[:20000]:
                if H.reshape(-1)[idx] < 0.34:
                    break
                cy, cx = divmod(int(idx), size)
                ok = True
                for (py, px, *_1) in cores:
                    dy = min((cy - py) % size, (py - cy) % size)
                    dx = min((cx - px) % size, (px - cx) % size)
                    if dy * dy + dx * dx < min_sep * min_sep:
                        ok = False
                        break
                if ok:
                    cores.append([cy, cx])
                    if len(cores) >= civ_count:
                        break
            # assign every land cell to its nearest core (torus), within radius
            if cores:
                yy = np.arange(size)
                xx = np.arange(size)
                best_d2 = np.full((size, size), np.inf, np.float32)
                best_f = np.full((size, size), -1, np.int16)
                t0s = np.sort(rng.uniform(0, 1.6 * YEAR_DAYS, len(cores)))
                r_max = size * 0.20
                for f, (cy, cx) in enumerate(cores):
                    dy = np.minimum((yy - cy) % size, (cy - yy) % size).astype(np.float32)
                    dx = np.minimum((xx - cx) % size, (cx - xx) % size).astype(np.float32)
                    d2 = dy[:, None] ** 2 + dx[None, :] ** 2
                    take = d2 < best_d2
                    best_d2 = np.where(take, d2, best_d2)
                    best_f = np.where(take, f, best_f)
                dist = np.sqrt(best_d2)
                terr = (dist < r_max) & land & (H > 0.18)
                falloff = np.clip(1 - dist / r_max, 0, 1)
                faction_id = np.where(terr, best_f, -1).astype(np.int16)
                habitability = np.where(terr, H * (0.35 + 0.65 * falloff), 0).astype(np.float32)
                for f, (cy, cx) in enumerate(cores):
                    civ_t0[best_f == f] = t0s[f]
                    cores[f] = [cy / size, cx / size, f, float(t0s[f])]
        self.faction_id = faction_id
        self.civ_t0 = civ_t0
        self.habitability = habitability
        self.civ_cores = cores          # list of [yn, xn, faction, t0]

    def strided(self, st):
        s = WorldSlice.__new__(WorldSlice)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(s, k, v[::st, ::st] if v.ndim == 2 else v[::st])
            else:
                setattr(s, k, v)     # carry scalars / core list by reference
        return s


def build_world(seed, size, civ_count=3):
    """The heavy, seed-only part: fields, D8 flow, cloud sheets, civ cores."""
    elev = elevation_field(size, seed).astype(np.float32)
    moist = moisture_field(size, seed).astype(np.float32)
    _, accum = compute_rivers(elev, 0.5)     # accumulation is sea-level-free
    cloud1 = fractal_noise(size, seed + 4001, base_period=4, octaves=5).astype(np.float32)
    cloud2 = fractal_noise(size, seed + 8009, base_period=3, octaves=4).astype(np.float32)
    return WorldSlice(elev, moist, accum.astype(np.float32),
                      cloud1, cloud2, int(civ_count), int(seed))


def default_river_threshold(size):
    """350 is tuned for 512²; scale down for smaller worlds."""
    return round(350 * (size / 512) ** 1.5)


def frame_params(t, sea_level, tide_amp, season_amp):
    sea_eff = sea_level + tide_amp * np.sin(2 * np.pi * t / TIDE_PERIOD)
    season_off = season_amp * np.sin(2 * np.pi * t / YEAR_DAYS)
    sun_x = t % 1.0
    return sea_eff, season_off, sun_x


def render(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night):
    """Pure: (world slice, layer name, sim time, knobs) -> uint8 RGB image."""
    sea_eff, season_off, sun_x = frame_params(t, sea_level, tide_amp, season_amp)
    e = ws.elev
    land = e >= sea_eff

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
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        img = color_ramp(tf, *TMP_RAMP)
    elif layer == "moisture":
        img = color_ramp(ws.moist, *MOI_RAMP)
    elif layer == "flow":
        img = color_ramp(ws.log_accum, *FLOW_RAMP)
    elif layer == "clouds":
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        base = BIOME_LUT[biome_ids(e, tf, ws.moist, sea_eff)] * 0.45
        cov = clouds_field(ws, t)[..., None]
        img = base * (1 - cov) + np.array([242, 246, 250], np.float32) * cov
        l = daylight_row(ws.xn, sun_x, day_night)[None, :, None]
        img *= 0.35 + 0.65 * l
    elif layer == "flora":
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        veg = flora_field(ws, tf, sea_eff)
        img = color_ramp(veg, *FLORA_RAMP)
        img[~land] = (26, 42, 74)
    elif layer == "fauna":
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        veg = flora_field(ws, tf, sea_eff)
        herb, pred = fauna_field(veg, t)
        civ_p, _ = civ_population(ws, t)
        herb = herb * (1 - 0.7 * np.clip(civ_p, 0, 1))     # settlers hunt/clear game
        img = np.empty(e.shape + (3,), np.float32)
        img[..., 0] = 40 + 205 * np.clip(pred * 1.6, 0, 1)  # predators redden it
        img[..., 1] = 40 + 175 * np.clip(herb, 0, 1)        # prey biomass = green
        img[..., 2] = 45 + 30 * np.clip(herb, 0, 1)
        img[~land] = (20, 32, 58)
    elif layer == "civ":
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        base = BIOME_LUT[biome_ids(e, tf, ws.moist, sea_eff)]
        gray = base @ np.array([0.30, 0.59, 0.11], np.float32)
        img = np.repeat(gray[..., None], 3, axis=2) * 0.55 + 18
        pop, fid = civ_population(ws, t)
        has = fid >= 0
        if has.any():
            tint = CIV_COLORS[np.clip(fid, 0, len(CIV_COLORS) - 1)]
            a = np.clip(pop, 0, 0.9)[..., None]
            img = np.where(has[..., None], img * (1 - a) + tint * a, img)
        size = e.shape[0]
        for (yn, xn, f, t0) in ws.civ_cores:
            if t <= t0:
                continue
            cy, cx = int(yn * size), int(xn * size)
            r = max(1, size // 200)
            y0, y1 = max(0, cy - r), min(size, cy + r + 1)
            x0, x1 = max(0, cx - r), min(size, cx + r + 1)
            img[y0:y1, x0:x1] = (250, 250, 235)
    elif layer == "light":
        l = daylight_row(ws.xn, sun_x, day_night)[None, :]
        img = np.empty(e.shape + (3,), np.float32)
        img[..., 0] = 255 * l
        img[..., 1] = 248 * l
        img[..., 2] = 80 + 145 * l
    else:  # biome / composite
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        ids = biome_ids(e, tf, ws.moist, sea_eff)
        img = BIOME_LUT[ids]
        if layer == "composite":
            img = img.copy()
            img[land] *= ws.shade[land, None]
            rivers = (ws.accum > river_thr) & land
            img[rivers] = (70, 130, 200)
            l = daylight_row(ws.xn, sun_x, day_night)[None, :]
            img[..., 0] *= l
            img[..., 1] *= l * 0.96 + 0.04
            img[..., 2] *= l * 0.82 + 0.18
    return np.clip(img, 0, 255).astype(np.uint8)


def render_rgba_bytes(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night):
    """Same as render(), packed as (width, RGBA bytes) for canvas blitting."""
    img = render(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night)
    h, w, _ = img.shape
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = img
    rgba[..., 3] = 255
    return w, rgba.tobytes()
