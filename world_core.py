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
    society   civilization(t), history(t)                    history CA (M3)

Seekability, the one invariant: render(t) is STATELESS. Fields that are
algebraic in t (clouds, flora, fauna) are pure functions. The M3 history —
which genuinely needs state, because "a plague starts on day 40 and spreads"
is by definition non-seekable — is INTEGRATED ONCE at build time into a
keyframed timeline; render(t) then only INTERPOLATES that timeline. So the
simulation is stateful but its consumption is not, and the exporter still
samples any t on demand.

Consumers: world_viewer.py (desktop matplotlib) and web/index.html (same code
in the browser via Pyodide). No matplotlib, no PIL, no I/O — numpy in, RGB out.
"""

import numpy as np

from worldgen import (elevation_field, moisture_field, compute_rivers,
                      fractal_noise, elevation_window, moisture_window,
                      noise_window, BIOME_COLORS)

# ---------------------------------------------------------------------------
# Time model (M2). t is measured in sim DAYS.
# ---------------------------------------------------------------------------
YEAR_DAYS = 96.0          # one year = 96 days
TIDE_PERIOD = 0.52        # ~semi-diurnal tide
FAUNA_PERIOD = 32.0       # predator-prey limit cycle length (days)

# ---------------------------------------------------------------------------
# History CA (M3) parameters.
# ---------------------------------------------------------------------------
HIST_SIZE = 48            # coarse simulation grid (cells per side)
WEEK_DAYS = 7.0           # one CA step = one week
HIST_YEARS = 24.0         # how much history to pre-integrate
KF_WEEKS = 3              # record a keyframe every N weeks
POP_MAX = 1.2             # population value that maps to a full uint8

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
    ("vitality", "Vitality"),
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


def biome_ids(e, t, m, sea, tide=0.0):
    """Vectorized version of worldgen.classify_biomes -> int16 ids.

    `sea` is the GEOGRAPHIC sea level (the slider, no instantaneous tide), so
    the coastline/beach terrain only moves when you change the slider. `tide`
    is the tidal amplitude: the beach is the intertidal band it sweeps, so sand
    is drawn as wide as the tide reaches. The instantaneous waterline (which
    covers/uncovers that sand each cycle) is applied as a wet overlay in
    render(), not here — the sand geography itself stays put."""
    lo = sea - tide                      # lowest the water ever falls (geo)
    beach_top = sea + tide + 0.015       # intertidal band + a little dry sand
    water = e < lo
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
# Society — the M3 history simulation (integrated once, sampled as a timeline).
# ===========================================================================
def _neigh4(a):
    """4-neighbor mean on the torus (works for 2D or leading-axis stacks)."""
    ax = a.ndim - 2
    return 0.25 * (np.roll(a, 1, ax) + np.roll(a, -1, ax)
                   + np.roll(a, 1, ax + 1) + np.roll(a, -1, ax + 1))


def civ_population(ws, t):
    """Per-cell settled population in [0,1] and faction id (-1 = none), read
    from the pre-integrated history timeline and upsampled to render size."""
    pop, own, _stress, _unrest = _sample_history(ws, t)
    return pop, own


def _window_indices(ws, n=HIST_SIZE):
    """np.ix_ that maps each render pixel of `ws` to its coarse (n x n) cell via
    world coordinates, honouring the slice's viewport (pan/zoom)."""
    size = ws.size
    step = (np.arange(size, dtype=np.float32) / size - 0.5) * ws.span
    ci = (((ws.cx + step) % 1.0) * n).astype(np.int64) % n
    ri = (((ws.cy + step) % 1.0) * n).astype(np.int64) % n
    return np.ix_(ri, ci)


def _sample_history(ws, t):
    """Interpolate the coarse history timeline at day t and upsample to the
    render grid. Returns (pop, faction_id, stress, unrest), all render-sized."""
    size = ws.elev.shape[0]
    if not getattr(ws, "has_history", False):
        z = np.zeros((size, size), np.float32)
        return z, np.full((size, size), -1, np.int16), z, z

    days = ws.hist_days
    tc = float(np.clip(t, days[0], days[-1]))
    i = int(np.searchsorted(days, tc, side="right") - 1)
    i = max(0, min(i, len(days) - 2))
    span = days[i + 1] - days[i]
    fr = 0.0 if span <= 0 else (tc - days[i]) / span

    pop_c = ((1 - fr) * ws.hist_pop[i] + fr * ws.hist_pop[i + 1]) * (POP_MAX / 255.0)
    stress_c = ((1 - fr) * ws.hist_stress[i] + fr * ws.hist_stress[i + 1]) / 255.0
    unrest_c = ((1 - fr) * ws.hist_unrest[i] + fr * ws.hist_unrest[i + 1]) / 255.0
    own_c = ws.hist_own[i] if fr < 0.5 else ws.hist_own[i + 1]

    # map each render pixel to its coarse history cell via WORLD coordinates,
    # so the timeline lines up with the (possibly zoomed) window on screen.
    up = _window_indices(ws)
    return (pop_c[up].astype(np.float32), own_c[up].astype(np.int16),
            stress_c[up].astype(np.float32), unrest_c[up].astype(np.float32))


# ===========================================================================
class WorldSlice:
    """Static per-resolution data + grids (full res, or strided for thumbs)."""

    def __init__(self, elev, moist, accum, cloud1, cloud2, civ_count, seed,
                 cx=0.5, cy=0.5, span=1.0):
        self.size = elev.shape[0]
        self.seed = int(seed)
        # viewport on the unit torus: window centered at (cx, cy) of side span.
        # span == 1 -> the whole planet (the default, backward-compatible view).
        self.cx, self.cy, self.span = float(cx), float(cy), float(span)
        self.elev, self.moist, self.accum = elev, moist, accum
        self.cloud1, self.cloud2 = cloud1, cloud2
        self._derive_grids()
        self.has_history = False
        self._build_history(civ_count, seed)

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
        gy, gx = np.gradient(self.elev)
        self.shade = np.clip(1 - (gx + gy) * 2.2, 0.75, 1.25).astype(np.float32)
        amax = float(self.accum.max())
        self.log_accum = (np.log1p(self.accum) / np.log1p(amax)
                          if amax > 0 else np.zeros_like(self.elev))
        self.orographic = np.clip(gx * 6.0, 0, 1).astype(np.float32)

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
        s = WorldSlice.__new__(WorldSlice)
        s.size, s.seed = size, self.seed
        s.cx, s.cy, s.span = float(cx), float(cy), span
        s.elev = elevation_window(size, self.seed, cx, cy, span)
        s.moist = moisture_window(size, self.seed, cx, cy, span)
        # global D8 drainage can't be windowed yet (needs upstream boundary
        # conditions) -> no rivers inside a zoom for now; fields still resolve.
        s.accum = np.zeros_like(s.elev)
        c1 = noise_window(self.seed + 4001, cx, cy, span, size, base_period=4)
        c2 = noise_window(self.seed + 8009, cx, cy, span, size, base_period=3)
        rng = np.ptp(c1) or 1.0
        s.cloud1 = ((c1 - c1.min()) / rng).astype(np.float32)
        rng2 = np.ptp(c2) or 1.0
        s.cloud2 = ((c2 - c2.min()) / rng2).astype(np.float32)
        s._derive_grids()
        # share the pre-integrated history + the live eco sim (both global, not
        # window-local) by reference, so zooming never rebuilds or forks them
        s.has_history = self.has_history
        for k in ("hist_days", "hist_pop", "hist_own", "hist_stress",
                  "hist_unrest", "civ_cores", "eco"):
            if hasattr(self, k):
                setattr(s, k, getattr(self, k))
        return s

    # ---- downsample a fine field to the coarse HIST grid (nearest) ----------
    def _coarse(self, field):
        idx = (np.arange(HIST_SIZE) * self.size) // HIST_SIZE
        return field[np.ix_(idx, idx)].astype(np.float32)

    # ---- M3: seed factions, generate events, integrate the timeline --------
    def _build_history(self, civ_count, seed):
        if civ_count <= 0:
            self.civ_cores = []
            return
        H = HIST_SIZE
        e = self._coarse(self.elev)
        m = self._coarse(self.moist)
        sea0 = 0.42
        yn = (np.arange(H, dtype=np.float32) / H)[:, None]
        lat = np.repeat(1 - np.abs(yn - 0.5) * 2, H, axis=1)
        land = e >= sea0
        temp = np.clip(lat - np.clip(e - sea0, 0, 1) * 0.9, 0, 1)
        warmth = np.clip((temp - 0.26) / 0.55, 0, 1)
        wet = smoothstep((m - 0.1) / 0.8)
        flora0 = warmth * (0.30 + 0.70 * wet) * land
        water = np.clip(self._coarse(self.log_accum) * 0.7, 0, 1)
        temperate = np.clip(1 - np.abs(temp - 0.62) / 0.45, 0, 1)
        hab = (0.42 * flora0 + 0.30 * water + 0.28 * temperate) * land
        hab *= np.clip(1 - (e - 0.75) / 0.25, 0, 1)
        cap0 = np.clip(0.15 + 0.95 * hab, 0.0, 1.2) * land       # food capacity
        passable = land * np.clip(1 - np.clip((e - 0.7) / 0.3, 0, 1) * 0.7, 0.1, 1)

        # ---- pick faction cores: habitability maxima, spaced apart ----------
        rng = np.random.default_rng((seed ^ 0x50C1A1) & 0x7FFFFFFF)
        order = np.argsort(-hab.reshape(-1))
        cores, min_sep = [], H * 0.16
        for idx in order:
            if hab.reshape(-1)[idx] < 0.30:
                break
            cy, cx = divmod(int(idx), H)
            if all(min((cy - py) % H, (py - cy) % H) ** 2
                   + min((cx - px) % H, (px - cx) % H) ** 2 >= min_sep ** 2
                   for py, px in cores):
                cores.append((cy, cx))
                if len(cores) >= civ_count:
                    break
        if not cores:
            self.civ_cores = []
            return
        nf = len(cores)

        # ---- generate deterministic shock events over the horizon ----------
        weeks = int(HIST_YEARS * YEAR_DAYS / WEEK_DAYS)
        yy = np.arange(H)[:, None]
        xx = np.arange(H)[None, :]

        def blob(cy, cx, radius):
            dy = np.minimum((yy - cy) % H, (cy - yy) % H)
            dx = np.minimum((xx - cx) % H, (cx - xx) % H)
            return np.exp(-(dy * dy + dx * dx) / (2 * radius * radius))

        events = []
        arid = m < 0.4
        for _ in range(int(HIST_YEARS * 0.7)):        # pests / blights
            ly = int(rng.integers(0, H)); lx = int(rng.integers(0, H))
            events.append(("pest", blob(ly, lx, H * 0.10),
                           int(rng.integers(0, weeks)),
                           int(rng.integers(4, 12)), float(rng.uniform(0.20, 0.5))))
        for _ in range(int(HIST_YEARS * 0.25)):       # droughts (arid, long)
            cand = np.argwhere(arid)
            ly, lx = cand[rng.integers(len(cand))] if len(cand) else (H // 2, H // 2)
            events.append(("drought", blob(int(ly), int(lx), H * 0.18),
                           int(rng.integers(0, weeks)),
                           int(rng.integers(24, 44)), float(rng.uniform(0.45, 0.7))))
        polar = (lat < 0.45)
        for _ in range(max(1, int(HIST_YEARS * 0.12))):  # cold spells / ice
            events.append(("ice", polar.astype(np.float32),
                           int(rng.integers(0, weeks)),
                           int(rng.integers(20, 40)), float(rng.uniform(0.4, 0.65))))

        def event_mult(week):
            mult = np.ones((H, H), np.float32)
            for _kind, shape, t0, dur, strength in events:
                if t0 <= week < t0 + dur:
                    env = np.sin(np.pi * (week - t0) / dur)   # ramp up then down
                    mult *= 1 - (1 - strength) * env * shape
            return mult

        # ---- integrate the coarse CA week by week --------------------------
        rP, col, src_gain = 0.09, 0.012, 0.55
        decay, diff, mig, war_mort = 0.986, 0.34, 0.12, 0.07
        own_thresh, tiny = 0.02, 1e-3

        P = np.zeros((H, H), np.float32)
        I = np.zeros((nf, H, H), np.float32)
        for f, (cy, cx) in enumerate(cores):
            P[cy, cx] = 0.06
            I[f, cy, cx] = 1.0
        t0s = np.sort(rng.uniform(0, 1.6 * YEAR_DAYS, nf)) if nf else np.array([])

        kf_pop, kf_own, kf_stress, kf_unrest, kf_days = [], [], [], [], []
        for w in range(weeks):
            emult = event_mult(w)
            cap = np.maximum(cap0 * emult, 1e-3)

            maxI = I.max(0)
            own = np.where(land & (maxI > own_thresh), I.argmax(0), -1)

            # population: logistic toward capacity + slow colonization of land
            owned = own >= 0
            P += (rP * P + col * owned) * (1 - P / cap)
            P = np.clip(P, 0, POP_MAX)

            # influence: sourced by population, diffuses, decays, blocked by sea
            for f in range(nf):
                I[f] += src_gain * P * (own == f)
            I = decay * ((1 - diff) * I + diff * _neigh4(I)) * passable[None]

            # re-resolve ownership and border contest -> war casualties
            maxI = I.max(0)
            if nf >= 2:
                secondI = np.sort(I, axis=0)[-2]
            else:
                secondI = np.zeros_like(maxI)
            owned = land & (maxI > own_thresh)
            own = np.where(owned, I.argmax(0), -1)
            contest = np.where(maxI > tiny, secondI / (maxI + 1e-6), 0.0) * owned
            P *= 1 - war_mort * contest

            # migration + drop population the state can no longer reach
            P = (1 - mig) * P + mig * _neigh4(P)
            P *= (maxI > tiny)

            unrest = np.clip(np.clip(P / cap - 1, 0, 1) + contest, 0, 1)
            stress = np.clip(1 - emult, 0, 1)

            if w % KF_WEEKS == 0 or w == weeks - 1:
                kf_days.append(w * WEEK_DAYS)
                kf_pop.append((np.clip(P / POP_MAX, 0, 1) * 255).astype(np.uint8))
                kf_own.append(own.astype(np.int8))
                kf_stress.append((stress * 255).astype(np.uint8))
                kf_unrest.append((unrest * 255).astype(np.uint8))

        self.has_history = True
        self.hist_days = np.array(kf_days, np.float32)
        self.hist_pop = np.stack(kf_pop)
        self.hist_own = np.stack(kf_own).astype(np.int16)
        self.hist_stress = np.stack(kf_stress)
        self.hist_unrest = np.stack(kf_unrest)
        # normalized founding coords for city markers (yn, xn, faction, founded-day)
        self.civ_cores = [(cy / H, cx / H, f, float(t0s[f]))
                          for f, (cy, cx) in enumerate(cores)]

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
        return s


# ===========================================================================
# EcoSim — the STATEFUL near-form substrate (M4). Everything else in this file
# is a pure, seekable function of t. This one is not: it integrates forward and
# has MEMORY, so events leave lasting consequences. A flood salts the soil; a
# drought burns the forest; recovery is slow and only spreads inward from
# surviving neighbours, so an isolated dead zone stays dead until life reaches
# it again — which is how a world can fail to come back. Driven live by the
# sliders (sea level, seasons); reset() returns it to the pristine world. It is
# deliberately NOT scrubbable backward (that is what "consequences" means) — it
# only runs forward or resets. Coarse (HIST_SIZE) and cheap.
# ===========================================================================
class EcoSim:
    def __init__(self, ws, seed=0):
        H = HIST_SIZE
        self.H = H
        self.e = ws._coarse(ws.elev).astype(np.float32)
        self.m = ws._coarse(ws.moist).astype(np.float32)
        yn = (np.arange(H, dtype=np.float32) / H)[:, None]
        self.lat = np.repeat(1 - np.abs(yn - 0.5) * 2, H, axis=1).astype(np.float32)
        self.lat_signed = np.repeat((0.5 - yn) * 2, H, axis=1).astype(np.float32)
        self.sea0 = 0.42
        self.reset()

    def _climate(self, season_off):
        temp = np.clip(self.lat + season_off * self.lat_signed
                       - np.clip(self.e - self.sea_ref, 0, 1) * 0.9, 0, 1)
        warmth = np.clip((temp - 0.26) / 0.55, 0, 1)
        wet = smoothstep((self.m - 0.1) / 0.8)
        return warmth, wet

    def reset(self):
        """Back to the pristine, climax-state world (day 0)."""
        self.sea_ref = float(self.sea0)
        self.t = 0.0
        land = self.e >= self.sea0
        warmth, wet = self._climate(0.0)
        cap = warmth * (0.30 + 0.70 * wet) * land
        self.fert = (np.clip(0.35 + 0.65 * cap, 0.05, 1) * land).astype(np.float32)
        self.veg = (cap * 0.9).astype(np.float32)
        self.fauna = (cap * 0.5).astype(np.float32)
        self.civ = ((cap > 0.55) * 0.35).astype(np.float32)   # some seed settlements
        self.scorch = np.zeros((self.H, self.H), np.float32)

    def step(self, dt_days, sea_level, season_off):
        """Advance the ecosystem by dt_days under the current sliders."""
        if dt_days <= 0:
            return
        n = int(min(48, max(1, np.ceil(dt_days / 4.0))))    # sub-step for stability
        h = min(dt_days / n, 6.0)
        for _ in range(n):
            self._micro(h, float(sea_level), float(season_off))
        self.t += dt_days

    def _micro(self, h, sea_level, season_off):
        e = self.e
        warmth, wet = self._climate(season_off)
        under = e < sea_level
        land = ~under
        drowned = under & (e >= self.sea_ref)          # land the sea just took
        cap = warmth * (0.30 + 0.70 * wet) * self.fert * (1 - self.scorch) * land
        dry = np.clip(warmth - 0.75 * wet - 0.05, 0, 1)  # hot & dry -> fire/desert

        veg, fauna, civ, fert, scorch = (self.veg, self.fauna, self.civ,
                                         self.fert, self.scorch)

        # --- flood: drown the biota, salt the soil ---
        veg = np.where(under, veg * np.exp(-0.6 * h), veg)
        fauna = np.where(under, fauna * np.exp(-0.5 * h), fauna)
        civ = np.where(under, civ * np.exp(-0.7 * h), civ)
        fert = np.where(drowned, fert - 0.008 * h, fert)
        scorch = np.where(drowned, scorch + 0.020 * h, scorch)

        # --- fire / drought: needs both heat-dryness AND fuel (vegetation), so
        #     it strikes grass/savanna in hot summers, not bare desert ---
        burn = np.clip(dry - 0.40, 0, 1) * veg * land
        ignite = burn * (burn > 0.03)
        veg = veg - 0.70 * ignite * h
        fauna = fauna - 0.45 * ignite * h
        scorch = scorch + 0.40 * ignite * h
        desert = land & (dry > 0.45) & (veg < 0.10)     # bare hot ground erodes
        fert = np.where(desert, fert - 0.004 * h, fert)

        # --- recovery on dry land: growth needs a seed (self or neighbour), so
        #     wiped, isolated cells cannot restart until life spreads back in ---
        cap_pos = cap > 0.02
        vseed = 0.015 + 0.85 * veg + 0.5 * _neigh4(veg)
        veg = np.where(land, veg + 0.045 * h * vseed
                       * np.clip(1 - veg / (cap + 1e-3), 0, 1) * cap_pos, veg)
        fcap = 0.9 * veg
        fseed = 0.02 + 0.85 * fauna + 0.5 * _neigh4(fauna)
        fauna = np.where(land, fauna + 0.06 * h * fseed
                         * np.clip(1 - fauna / (fcap + 1e-3), 0, 1) * (fcap > 0.02), fauna)
        # slow soil rebuild (needs life nearby) and scar fade — the "much time
        # must pass" knobs, on a timescale of years not days
        fert = fert + h * (0.0016 * veg + 0.0009 * _neigh4(veg)) * (1 - fert) * land
        scorch = scorch - 0.0025 * h

        # --- civilization: grows on food, collapses without it, recolonises
        #     only from surviving neighbours ---
        food = 0.5 * veg + 0.5 * fauna
        cseed = 0.55 * civ + 0.35 * _neigh4(civ)
        ok = land & (food > 0.28)
        civ = civ + np.where(ok, 0.02 * h * (0.04 + cseed) * (1 - civ), 0)
        civ = civ - np.where(~ok, 0.05 * h * civ, 0)
        civ = np.where(food < 0.14, civ * np.exp(-0.12 * h), civ)  # famine wipeout
        fauna = fauna - 0.03 * h * civ * fauna          # hunting pressure
        veg = veg - 0.01 * h * civ * veg                # land clearing

        self.veg = np.clip(veg, 0, 1).astype(np.float32)
        self.fauna = np.clip(fauna, 0, 1).astype(np.float32)
        self.civ = np.clip(civ, 0, 1).astype(np.float32)
        self.fert = np.clip(fert, 0.02, 1).astype(np.float32)
        self.scorch = np.clip(scorch, 0, 1).astype(np.float32)
        # the coastline the ecosystem is adapted to drifts toward the imposed
        # level (slowly), so a held sea level becomes the new normal
        self.sea_ref += 0.02 * h * (sea_level - self.sea_ref)

    def sample(self, ws):
        """Upsample the coarse state to the render window (honours pan/zoom)."""
        up = _window_indices(ws)
        return {k: getattr(self, k)[up] for k in
                ("veg", "fauna", "fert", "civ", "scorch")}


def build_world(seed, size, civ_count=3):
    """The heavy, seed-only part: fields, D8 flow, cloud sheets, M3 history."""
    elev = elevation_field(size, seed).astype(np.float32)
    moist = moisture_field(size, seed).astype(np.float32)
    _, accum = compute_rivers(elev, 0.5)     # accumulation is sea-level-free
    cloud1 = fractal_noise(size, seed + 4001, base_period=4, octaves=5).astype(np.float32)
    cloud2 = fractal_noise(size, seed + 8009, base_period=3, octaves=4).astype(np.float32)
    ws = WorldSlice(elev, moist, accum.astype(np.float32),
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


def _terrain_gray(ws, tf, sea, tide=0.0):
    """Dim grayscale terrain base for the society layers."""
    base = BIOME_LUT[biome_ids(ws.elev, tf, ws.moist, sea, tide)]
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
        base = BIOME_LUT[biome_ids(e, tf, ws.moist, sea_level, tide_amp)] * 0.45
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
        img[..., 0] = 40 + 205 * np.clip(pred * 1.6, 0, 1)
        img[..., 1] = 40 + 175 * np.clip(herb, 0, 1)
        img[..., 2] = 45 + 30 * np.clip(herb, 0, 1)
        img[~land] = (20, 32, 58)
    elif layer == "civ":
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        img = _terrain_gray(ws, tf, sea_level, tide_amp)
        pop, fid = civ_population(ws, t)
        has = fid >= 0
        if has.any():
            tint = CIV_COLORS[np.clip(fid, 0, len(CIV_COLORS) - 1)]
            a = np.clip(pop * 1.4, 0, 0.9)[..., None]
            img = np.where(has[..., None], img * (1 - a) + tint * a, img)
        _city_dots(ws, img, t)
    elif layer == "history":
        # the chronicle: territory + where the world is thriving / at war / starving
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        img = _terrain_gray(ws, tf, sea_level, tide_amp)
        pop, fid, stress, unrest = _sample_history(ws, t)
        has = fid >= 0
        if has.any():
            tint = CIV_COLORS[np.clip(fid, 0, len(CIV_COLORS) - 1)]
            a = np.clip(pop * 1.3, 0, 0.85)[..., None]
            img = np.where(has[..., None], img * (1 - a) + tint * a, img)
        war = np.clip(unrest * (pop > 0.02), 0, 1)[..., None]          # red fronts
        img = img * (1 - war) + np.array([235, 60, 45], np.float32) * war
        fam = np.clip(stress * land * (0.4 + pop), 0, 1)[..., None]    # violet famine
        img = img * (1 - fam) + np.array([150, 70, 200], np.float32) * fam
        _city_dots(ws, img, t)
    elif layer == "vitality":
        # the LIVING world: the stateful ecosystem, where events leave scars.
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        img = _terrain_gray(ws, tf, sea_level, tide_amp)
        eco = getattr(ws, "eco", None)
        if eco is not None:
            s = eco.sample(ws)
            veg = np.clip(s["veg"] / 0.55, 0, 1)               # healthy veg -> full green
            scorch = np.clip(s["scorch"], 0, 1)[..., None]
            fert = np.clip(s["fert"], 0, 1)[..., None]
            civ = np.clip((s["civ"] - 0.35) / 0.65, 0, 1)[..., None]
            # barren land tints toward tan (drier/poorer soil = paler)
            bare = ((1 - veg))[..., None]
            img = img * (1 - 0.45 * bare) + np.array([168, 140, 96], np.float32) * (0.45 * bare) * (0.5 + 0.5 * fert)
            # vegetation green is the dominant signal
            g = (0.9 * veg)[..., None]
            img = img * (1 - g) + np.array([48, 152, 56], np.float32) * g
            # burn/salt scars: charred ground
            img = img * (1 - scorch) + np.array([54, 40, 34], np.float32) * scorch
            # civilization: a restrained gold accent only where it is dense
            img = img * (1 - 0.7 * civ) + np.array([240, 222, 128], np.float32) * (0.7 * civ)
        # the instantaneous waterline covers whatever is currently below it
        under = (e < sea_eff)[..., None]
        img = img * (1 - under) + np.array([40, 80, 130], np.float32) * under
        _city_dots(ws, img, t)
    elif layer == "light":
        l = daylight_row(ws.xn, sun_x, day_night)[None, :]
        img = np.empty(e.shape + (3,), np.float32)
        img[..., 0] = 255 * l
        img[..., 1] = 248 * l
        img[..., 2] = 80 + 145 * l
    else:  # biome / composite
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        # geography is cut by the SLIDER sea level (+ tidal band), so the sand
        # stays put; the instantaneous tide only sweeps a waterline across it.
        ids = biome_ids(e, tf, ws.moist, sea_level, tide_amp)
        img = BIOME_LUT[ids].copy()
        lo = sea_level - tide_amp                       # permanent low-tide line
        wet = (e >= lo) & (e < sea_eff)                 # intertidal sand, now wet
        img[wet] = (70, 130, 180)
        if layer == "composite":
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
