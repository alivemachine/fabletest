"""
worldgen.py  —  the world's FIELD PRODUCERS (data, not pixels).

The whole philosophy: the world is a PURE FUNCTION of (seed, x, y). Nothing here
is stored. This module produces the raw per-tile fields the rest of the engine
reads — elevation, moisture, D8 flow — sampled from tileable, windowable noise
so the map wraps seamlessly and can be zoomed to any depth. Rendering lives in
world_core.py (state() -> colorize()); this file emits only numbers.

    windowed noise ->  elevation, moisture
    elevation      ->  flow direction (D8) ->  rivers & lakes
"""

import numpy as np

# ---------------------------------------------------------------------------
# LAYER 0 — WINDOWED value noise. Tileable fractal value-noise sampled over an
# ARBITRARY window [cx±span/2, cy±span/2] of the unit torus, at any zoom. This
# is what makes the world a pure function of *continuous* (x, y): the 512² grid
# is just one sampling of it.
#
# Why a hash instead of a materialized lattice: zooming in means adding
# higher-frequency octaves whose lattice would be millions of cells wide — far
# too big to allocate. Instead each integer lattice corner's value is a pure
# hash of (seed, octave, i, j), so we only ever evaluate the corners the window
# actually touches. O(window pixels), independent of zoom depth.
#
# Coherence guarantee: octave o is divided by a FIXED total (sum of all octave
# amplitudes = 2), never by "how many octaves we happened to sum". So the low
# octaves contribute identically whether you view the whole planet or dive into
# one bay — the coastline stays exactly where it was; zoom only ADDS detail.
# ---------------------------------------------------------------------------

def _corner_hash(seed, octave, ix, iy):
    """Deterministic value in [0,1) at integer lattice corner (ix, iy)."""
    with np.errstate(over="ignore"):
        h = (ix.astype(np.int64) * np.int64(374761393)
             + iy.astype(np.int64) * np.int64(668265263)
             + np.int64(int(seed) & 0x7FFFFFFF) * np.int64(2246822519)
             + np.int64(octave) * np.int64(3266489917))
        h = (h ^ (h >> np.int64(15))) * np.int64(2654435761)
        h = h ^ (h >> np.int64(13))
    return (h & np.int64(0xFFFFFF)).astype(np.float32) / np.float32(0xFFFFFF)


def _octave_window(seed, octave, u, v, period):
    """One tileable octave sampled at world coords u (cols) and v (rows).
    u is (1,size), v is (size,1); lattice wraps mod `period` (integer)."""
    fu, fv = u * period, v * period
    iu, iv = np.floor(fu).astype(np.int64), np.floor(fv).astype(np.int64)
    fx, fy = (fu - iu).astype(np.float32), (fv - iv).astype(np.float32)
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    iu0, iu1 = iu % period, (iu + 1) % period
    iv0, iv1 = iv % period, (iv + 1) % period
    c00 = _corner_hash(seed, octave, iu0, iv0)
    c10 = _corner_hash(seed, octave, iu1, iv0)
    c01 = _corner_hash(seed, octave, iu0, iv1)
    c11 = _corner_hash(seed, octave, iu1, iv1)
    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy


def noise_window(seed, cx, cy, span, size, base_period=3):
    """Tileable fractal value noise over the window centered at (cx, cy) with
    side `span` (span=1 -> whole torus), sampled on a size² grid. Octaves are
    added until they hit one pixel (Nyquist at this zoom), so zooming in reveals
    finer terrain. Returns float32 (NOT range-normalized -- see _planet_norm)."""
    half = (np.arange(size, dtype=np.float64) / size - 0.5) * span
    u = ((cx + half) % 1.0)[None, :]
    v = ((cy + half) % 1.0)[:, None]
    field = np.zeros((size, size), np.float32)
    amp, period, o = 1.0, int(base_period), 0
    max_period = size / max(span, 1e-9)          # finest octave worth sampling
    while period <= max_period and o < 22:
        field += amp * _octave_window(seed, o, u, v, period)
        amp *= 0.5
        period *= 2
        o += 1
    return field / 2.0     # fixed total (sum 0.5^o -> 2): zoom-coherent


# ---------------------------------------------------------------------------
# LAYER 1 — ELEVATION.  Domain-warped for less "blobby" coastlines.
# ---------------------------------------------------------------------------

# normalization is computed ONCE per seed from a coarse planet-scale sample and
# reused for every window, so elevation/moisture keep a fixed meaning (a given
# sea level cuts the same coastline) no matter how far you pan or zoom.
_NORM_CACHE = {}


def _raw_elevation(seed, cx, cy, span, size):
    warp = noise_window(seed + 99, cx, cy, span, size, base_period=3)
    base = noise_window(seed, cx, cy, span, size, base_period=3)
    return 0.75 * base + 0.25 * warp                 # tileable domain blend


def _planet_norm(seed):
    """(elo, ehi, mlo, mhi) for this seed, from a coarse whole-planet sample."""
    key = int(seed)
    if key not in _NORM_CACHE:
        e = _raw_elevation(seed, 0.5, 0.5, 1.0, 160)
        m = noise_window(seed + 555, 0.5, 0.5, 1.0, 160, base_period=4)
        _NORM_CACHE[key] = (float(e.min()), float(e.max()),
                            float(m.min()), float(m.max()))
    return _NORM_CACHE[key]


def elevation_window(size, seed, cx=0.5, cy=0.5, span=1.0):
    raw = _raw_elevation(seed, cx, cy, span, size)
    elo, ehi, _, _ = _planet_norm(seed)
    return np.clip((raw - elo) / (ehi - elo + 1e-9), 0, 1).astype(np.float32)


def moisture_window(size, seed, cx=0.5, cy=0.5, span=1.0):
    raw = noise_window(seed + 555, cx, cy, span, size, base_period=4)
    _, _, mlo, mhi = _planet_norm(seed)
    return np.clip((raw - mlo) / (mhi - mlo + 1e-9), 0, 1).astype(np.float32)


def elevation_field(size, seed):
    return elevation_window(size, seed, 0.5, 0.5, 1.0)


def moisture_field(size, seed):
    return moisture_window(size, seed, 0.5, 0.5, 1.0)


# ---------------------------------------------------------------------------
# BIOME palette — the colours world_core's biome LUT is built from. (The
# classification itself is vectorized in world_core.biome_ids.)
# ---------------------------------------------------------------------------

BIOME_COLORS = {
    "deep_ocean":  (30,  60, 120),
    "ocean":       (45,  85, 155),
    "shallow":     (70, 130, 180),
    "beach":       (210, 200, 150),
    "desert":      (222, 200, 120),
    "savanna":     (180, 190,  90),
    "grassland":   (120, 180,  90),
    "forest":      ( 60, 140,  70),
    "jungle":      ( 30, 110,  55),
    "taiga":       ( 90, 140, 110),
    "tundra":      (170, 180, 170),
    "snow":        (235, 240, 245),
    "mountain":    (130, 125, 120),
    "high_peak":   (200, 200, 205),
}

# ---------------------------------------------------------------------------
# WATER — planetary hydrology. Real drainage, in four deterministic steps:
#
#   1. JITTER    a tiny per-cell hash offset breaks D8's 8-direction grid
#                bias (without it, steepest descent on smooth noise locks
#                onto long straight 45° runs — the "diagonal rivers" bug).
#   2. FILL      depression filling (a vectorized priority-flood): every
#                pit is raised to its spill level with an epsilon gradient
#                toward the ocean, so EVERY land cell provably drains to
#                the sea. Cells raised above the original terrain are
#                standing water — lakes, with outlets, for free.
#   3. D8        steepest-descent flow direction on the filled surface.
#   4. ACCUM     flow accumulation as a vectorized fixed-point iteration
#                (acc = 1 + inflow, repeated until stable) — no Python
#                per-cell loop, so it stays fast under Pyodide too.
# ---------------------------------------------------------------------------

SEA_REF = 0.42        # geographic reference sea level the drainage is built for

_D8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _flow_jitter(seed, size):
    """Deterministic per-cell offset in [-0.5, 0.5) used to de-bias D8."""
    iy, ix = np.mgrid[0:size, 0:size]
    return _corner_hash(seed ^ 0xA5F00D, 31, ix, iy).astype(np.float64) - 0.5


def _roll_min8(f):
    m = None
    for dy, dx in _D8:
        r = np.roll(np.roll(f, dy, 0), dx, 1)
        m = r if m is None else np.minimum(m, r)
    return m


def fill_depressions(elev, water, eps=1e-6, max_iters=None):
    """Raise every closed depression to its spill level (+eps per step toward
    the ocean), by iterative morphological reconstruction: start from the
    ocean seeds and repeatedly relax  f = max(elev, min(f, neighbors+eps)).
    Equivalent to priority-flood, but each pass is 8 vectorized rolls instead
    of a per-cell heap, which matters in the browser. Torus-aware."""
    e = elev.astype(np.float64)
    f = np.where(water, e, np.inf)
    if not water.any():                       # waterworldless seed: lowest cell
        f.flat[int(np.argmin(e))] = float(e.min())
    if max_iters is None:
        max_iters = 8 * e.shape[0]
    for _ in range(max_iters):
        nf = np.maximum(e, np.minimum(f, _roll_min8(f) + eps))
        if np.array_equal(nf, f):
            break
        f = nf
    return f


def d8_parents(f):
    """Flat index of each cell's steepest-descent neighbor (self = sink)."""
    size = f.shape[0]
    flat = f.reshape(-1)
    n = flat.size
    yy, xx = np.divmod(np.arange(n), size)
    parent = np.arange(n)
    best = flat.copy()
    for dy, dx in _D8:
        nidx = ((yy + dy) % size) * size + ((xx + dx) % size)
        nval = flat[nidx]
        better = nval < best
        parent = np.where(better, nidx, parent)
        best = np.where(better, nval, best)
    return parent


def flow_accumulation(parent, max_iters=None):
    """Cells drained through each cell (incl. itself). Fixed-point iteration
    acc = 1 + Σ children's acc; converges in (longest flow path) passes, each
    pass one vectorized bincount."""
    n = parent.size
    node = np.arange(n)
    src = node[parent != node]
    dst = parent[src]
    acc = np.ones(n, np.float64)
    if max_iters is None:
        max_iters = 8 * int(np.sqrt(n))
    for _ in range(max_iters):
        new = 1.0 + np.bincount(dst, weights=acc[src], minlength=n)
        if np.array_equal(new, acc):
            break
        acc = new
    return acc


def _dilate_max8(a):
    m = a
    for dy, dx in _D8:
        m = np.maximum(m, np.roll(np.roll(a, dy, 0), dx, 1))
    return m


def compute_hydrology(elev, seed, sea_level=SEA_REF):
    """Full planetary drainage for a heightfield. Returns a dict:
        accum      (size,size) float32  cells drained through each cell
        parent     (n,)        int64    D8 downstream flat index (self=sink)
        lake_level (size,size) float32  standing-water surface (0 = no lake)
        sea        (size,size) bool     below reference sea level
    The drainage is anchored to `sea_level` (the geographic reference); the
    UI's sea-level slider only moves the rendered waterline, as before."""
    size = elev.shape[0]
    # jitter amplitude ~ the finest noise octave's cell-scale relief: strong
    # enough that steepest descent wanders and merges (dendritic) instead of
    # locking into parallel 45° runs on broad smooth slopes
    jit_amp = 0.5 * 0.5 ** np.floor(np.log2(size / 3.0))
    ej = elev.astype(np.float64) + _flow_jitter(seed, size) * jit_amp
    water = ej < sea_level
    filled = fill_depressions(ej, water)
    parent = d8_parents(filled)
    accum = flow_accumulation(parent).reshape(size, size).astype(np.float32)
    depth = filled - ej
    lake = (~water) & (depth > 2.5 * jit_amp)      # deeper than the jitter floor
    lake_level = np.where(lake, filled, 0.0).astype(np.float32)
    # dilate one cell so the refined-elevation shoreline test (e < level) can
    # decide the water's edge inside neighboring cells too, not the raster grid
    lake_level = _dilate_max8(lake_level)
    return {"accum": accum, "parent": parent,
            "lake_level": lake_level, "sea": water}


def compute_rivers(elevation, sea_level, river_threshold=350):
    """Back-compat wrapper: boolean river mask + accumulation field."""
    h = compute_hydrology(elevation, 0, sea_level)
    rivers = (h["accum"] > river_threshold) & (elevation >= sea_level)
    return rivers, h["accum"]
