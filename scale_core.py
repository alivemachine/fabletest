"""scale_core.py — the multi-scale tile core (granularity levels).

The successor experiment to world_core's continuous zoom: here the SCALE
RANGE is the whole point. The world is a pure function of (seed, x, y)
sampled at any *tile size*:

    view(seed, cx, cy, tile_m, nx, ny)  ->  per-tile property grids

Coordinates are METERS (float64) on a square planet surface of side
PLANET_M (the continuous fields wrap as a torus). The invariant the whole
design hangs on:

    A view always renders the SAME NUMBER OF TILES. Resolution is which
    GRANULARITY LEVEL the tile properties come from — never tile count.

Formalism:

* A **granularity level** L(k) is a rung on a fixed ladder of tile sizes,
  tile_m(k) = TILE0_M / BRANCH**k  (TILE0_M = PLANET_M / GRID, BRANCH = 8).
  Zoom is continuous; the ladder only names the rungs (planet, country,
  city, district, block, building, figure, desk, insect, grain) and gives
  each entity kind the scale at which it "owns" tiles.

* Every **entity kind** has an intrinsic physical size (meters). Its
  footprint at the current scale is size/tile_m tiles:
    - footprint < ~half a tile  -> the kind is AGGREGATED: it exists only
      as a tile statistic ("≈ 214 trees", "ants: myriads").
    - footprint >= ~half a tile -> the kind is EXPANDED: individuals are
      placed deterministically (hashed lattices) and stamped onto tiles,
      growing 1 tile -> 2×2 -> 3×3 ... as you keep zooming.
  Zooming out re-aggregates; nothing is stored; the same ant is at the
  same millimeter every visit.

* Placement is CONDITIONED downward through the ladder: buildings only
  exist inside a settlement's influence, garden tables only near
  buildings, cups only on tables — each check is a pure hashed-lattice
  query, so any window at any depth is O(window), independent of zoom.

numpy in, plain dict/JSON out; no UI, no I/O. The browser harness
(web/scales.html) and any future Godot client read the same functions.
"""

import json
import math

import numpy as np

# ---------------------------------------------------------------------------
# the scale ladder
# ---------------------------------------------------------------------------

PLANET_M = 20_000_000.0          # planet side, meters (half Earth circumference)
GRID = 64                        # a view is always GRID tiles across (longest side)
BRANCH = 8.0                     # tiles per level step (8x smaller each rung)
TILE0_M = PLANET_M / GRID        # 312.5 km — the planet-level tile
MIN_TILE_M = 0.002               # 2 mm — a grain of sand gets a tile
SEA = 0.5                        # elevation value of the waterline

LEVELS = [
    # (name, what resolves into its own tiles at this rung)
    ("planet",   "continents, oceans"),
    ("country",  "territories, mountain ranges, forests"),
    ("city",     "cities occupy tiles"),
    ("district", "urban plan: districts, croplands, parks"),
    ("block",    "blocks, large woods, villages"),
    ("building", "buildings, gardens, big trees"),
    ("figure",   "people, animals, furniture"),
    ("desk",     "desk objects: cups, books, flowers"),
    ("insect",   "ants, pebbles, leaves"),
    ("grain",    "sand grains, crumbs"),
]


def level_tile_m(k):
    return TILE0_M / BRANCH ** k


def level_of(tile_m):
    """Continuous granularity level for a tile size (0 = planet rung)."""
    return math.log(TILE0_M / max(tile_m, 1e-9)) / math.log(BRANCH)


def level_name(tile_m):
    k = int(round(max(0.0, min(len(LEVELS) - 1, level_of(tile_m)))))
    return LEVELS[k][0], k


def fmt_m(v):
    """Human-readable length."""
    if v >= 1000.0:
        return f"{v / 1000.0:.4g} km"
    if v >= 1.0:
        return f"{v:.3g} m"
    if v >= 0.01:
        return f"{v * 100.0:.3g} cm"
    return f"{v * 1000.0:.3g} mm"


# ---------------------------------------------------------------------------
# hashing — int64 lattices, uint64 mixing; survives the full scale range
# ---------------------------------------------------------------------------

_U = np.uint64
_M1 = _U(0xFF51AFD7ED558CCD)
_M2 = _U(0xC4CEB9FE1A85EC53)
_P1 = _U(0x9E3779B97F4A7C15)
_P2 = _U(0xC2B2AE3D27D4EB4F)
_P3 = _U(0x165667B19E3779F9)
_S33 = _U(33)
_INV64 = 1.0 / 2.0 ** 64


def _mix(h):
    with np.errstate(over="ignore"):
        h = (h ^ (h >> _S33)) * _M1
        h = (h ^ (h >> _S33)) * _M2
    return h ^ (h >> _S33)


def _hash_u64(seed, salt, ix, iy):
    """uint64 hash of (seed, salt, ix, iy); ix/iy may be int64 arrays.
    uint64 arithmetic wraps by design (that IS the hash)."""
    s = _U((int(seed) * 0x632BE59B + int(salt) * 0x9E3779B9) & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        a = np.atleast_1d(np.asarray(ix)).astype(np.int64).view(np.uint64) * _P1
        b = np.atleast_1d(np.asarray(iy)).astype(np.int64).view(np.uint64) * _P2
        h = _mix(a ^ b ^ (s * _P3))
    return h.reshape(np.asarray(ix).shape)


def _hash01(seed, salt, ix, iy):
    return (_hash_u64(seed, salt, ix, iy) * _INV64).astype(np.float64)


# ---------------------------------------------------------------------------
# fields — value-noise fbm with a FIXED normalization so every octave
# contributes identically at every zoom (detail is added, never rescaled)
# ---------------------------------------------------------------------------

_OCT0_WL = PLANET_M / 4.0        # wavelength of octave 0 (4 cells around torus)
_MAX_OCT = 44                    # bottoms out below MIN_TILE_M
_PERS = 0.55
_FIXED_TOTAL = 1.0 / (1.0 - _PERS)   # sum of ALL possible octave amplitudes


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def _vnoise(seed, salt, x, y, k):
    """One value-noise octave at wavelength _OCT0_WL/2**k, wrapping the torus."""
    wl = _OCT0_WL / (1 << k) if k < 63 else _OCT0_WL / 2.0 ** k
    period = np.int64(4) << k                     # lattice cells around the torus
    u = x / wl
    v = y / wl
    iu = np.floor(u)
    iv = np.floor(v)
    fu = _smooth(u - iu)
    fv = _smooth(v - iv)
    iu = iu.astype(np.int64) % period
    iv = iv.astype(np.int64) % period
    iu1 = (iu + 1) % period
    iv1 = (iv + 1) % period
    s2 = salt * 64 + k
    n00 = _hash01(seed, s2, iu, iv)
    n10 = _hash01(seed, s2, iu1, iv)
    n01 = _hash01(seed, s2, iu, iv1)
    n11 = _hash01(seed, s2, iu1, iv1)
    top = n00 + (n10 - n00) * fu
    bot = n01 + (n11 - n01) * fu
    return top + (bot - top) * fv


def fbm(seed, salt, x, y, res_m, persistence=_PERS):
    """Fractal sum down to wavelength ~res_m, in [0,1]-ish around 0.5.

    Octave k contributes amp p^k / FIXED_TOTAL regardless of how many
    octaves are summed, and the finest octave FADES IN as its wavelength
    crosses res_m — so zooming adds detail continuously, and the coarse
    shape (the coastline you saw from orbit) never moves.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    total = np.zeros(np.broadcast(x, y).shape, np.float64)
    amp = 1.0
    for k in range(_MAX_OCT):
        wl = _OCT0_WL / 2.0 ** k
        if wl < res_m:
            break
        w = min(1.0, wl / res_m - 1.0) if wl < 2.0 * res_m else 1.0
        if w > 0.0:
            total += amp * w * (_vnoise(seed, salt, x, y, k) - 0.5)
        amp *= persistence
    return 0.5 + total / _FIXED_TOTAL


_SALT_ELEV, _SALT_MOIST, _SALT_TEMP, _SALT_GROUND = 11, 22, 33, 44


def elevation01(seed, x, y, res_m):
    return fbm(seed, _SALT_ELEV, x, y, res_m, persistence=0.58)


def moisture01(seed, x, y, res_m):
    return fbm(seed, _SALT_MOIST, x, y, max(res_m, 200.0), persistence=0.52)


def temperature_c(seed, x, y, res_m, elev01):
    v = np.mod(np.asarray(y, np.float64) / PLANET_M, 1.0)
    lat = 1.0 - np.abs(v - 0.5) * 2.0            # 1 at the equator band
    tnoise = (fbm(seed, _SALT_TEMP, x, y, max(res_m, 5000.0)) - 0.5) * 14.0
    alt = np.maximum(elev01 - SEA, 0.0) / (1.0 - SEA)
    return -22.0 + 52.0 * lat - 26.0 * alt + tnoise


def elevation_m(elev01):
    e = np.asarray(elev01, np.float64)
    up = (e - SEA) / (1.0 - SEA) * 4500.0
    dn = (e - SEA) / SEA * 5500.0
    return np.where(e >= SEA, up, dn)


def detail01(seed, x, y, tile_m):
    """Scale-relative ground detail: the fbm octaves whose wavelengths sit
    between ~2 and ~32 tiles, renormalized to unit amplitude. Coherent under
    pan (absolute lattices); fades octaves in/out under zoom. Cosmetic —
    it feeds shading and per-tile tint, not classification."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    total = np.zeros(np.broadcast(x, y).shape, np.float64)
    wsum = 0.0
    lo, hi = 2.0 * tile_m, 32.0 * tile_m
    for k in range(_MAX_OCT):
        wl = _OCT0_WL / 2.0 ** k
        if wl < lo:
            break
        if wl > hi * 2.0:
            continue
        w = min(1.0, wl / lo - 1.0) if wl < 2.0 * lo else 1.0
        w *= min(1.0, max(0.0, hi * 2.0 / wl - 1.0)) if wl > hi else 1.0
        if w <= 0.0:
            continue
        total += w * (_vnoise(seed, _SALT_GROUND, x, y, k) - 0.5)
        wsum += w * 0.5
    if wsum <= 0.0:
        return np.full(total.shape, 0.5)
    return 0.5 + 0.5 * total / wsum              # ~[0,1]


# ---------------------------------------------------------------------------
# biomes
# ---------------------------------------------------------------------------

BIOMES = [
    ("deep ocean",   (16, 38, 74)),
    ("ocean",        (26, 62, 110)),
    ("sea ice",      (196, 214, 226)),
    ("beach",        (206, 186, 138)),
    ("snow",         (232, 238, 242)),
    ("tundra",       (144, 150, 128)),
    ("taiga",        (58, 94, 74)),
    ("forest",       (52, 110, 62)),
    ("rainforest",   (26, 96, 54)),
    ("grassland",    (124, 152, 78)),
    ("savanna",      (166, 156, 84)),
    ("desert",       (204, 172, 108)),
    ("bare rock",    (128, 120, 112)),
]
BIOME_ID = {name: i for i, (name, _) in enumerate(BIOMES)}
_BIOME_LUT = np.array([c for _, c in BIOMES], np.float64)


def biome_ids(e, t_c, m):
    """Vectorized (elevation01, tempC, moisture01) -> biome id grid."""
    e = np.asarray(e)
    ids = np.full(e.shape, BIOME_ID["grassland"], np.int32)
    ids[m < 0.42] = BIOME_ID["savanna"]
    ids[m < 0.30] = BIOME_ID["desert"]
    ids[(m >= 0.42) & (t_c > 24.0)] = BIOME_ID["savanna"]
    ids[(m >= 0.48) & (t_c <= 24.0)] = BIOME_ID["forest"]
    ids[(m >= 0.62) & (t_c > 18.0)] = BIOME_ID["rainforest"]
    ids[(t_c < 4.0)] = BIOME_ID["taiga"]
    ids[(t_c < -4.0)] = BIOME_ID["tundra"]
    ids[(t_c < -12.0)] = BIOME_ID["snow"]
    ids[e > 0.84] = BIOME_ID["bare rock"]
    ids[(e > 0.84) & (t_c < 0.0)] = BIOME_ID["snow"]
    ids[e < SEA + 0.0035] = BIOME_ID["beach"]
    water = e < SEA
    ids[water] = BIOME_ID["ocean"]
    ids[water & (e < SEA - 0.09)] = BIOME_ID["deep ocean"]
    ids[water & (t_c < -8.0)] = BIOME_ID["sea ice"]
    return ids


def vegetation01(e, t_c, m):
    """Vegetation potential [0,1] — drives tree/bush/ant densities."""
    land = (np.asarray(e) >= SEA).astype(np.float64)
    warm = np.clip((t_c + 6.0) / 18.0, 0.0, 1.0)
    wet = np.clip((m - 0.25) / 0.5, 0.0, 1.0)
    high = np.clip((0.86 - np.asarray(e)) / 0.1, 0.0, 1.0)
    return land * warm * wet * high


# ---------------------------------------------------------------------------
# countries — voronoi over a hashed site lattice (naming + territory layer)
# ---------------------------------------------------------------------------

_COUNTRY_CELL = 2_400_000.0

_SYL_A = ["ka", "ve", "lo", "mi", "ra", "su", "tan", "or", "bel", "du",
          "fen", "gal", "hol", "is", "jor", "kel", "lum", "mor", "nar", "os"]
_SYL_B = ["dia", "ria", "land", "mark", "via", "stan", "gard", "holm", "wick",
          "ora", "una", "ath", "esse", "ovo", "ium", "ary", "eth", "onia", "ale", "or"]


def _gen_name(h, kind="country"):
    h = int(h)
    a = _SYL_A[h % len(_SYL_A)]
    b = _SYL_A[(h >> 5) % len(_SYL_A)]
    c = _SYL_B[(h >> 10) % len(_SYL_B)]
    base = (a + b + c) if (h >> 15) % 3 else (a + c)
    return base.capitalize()


def country_at(seed, x, y):
    """(country_id, name) per point — nearest hashed voronoi site."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ic = np.floor(x / _COUNTRY_CELL).astype(np.int64)
    jc = np.floor(y / _COUNTRY_CELL).astype(np.int64)
    best_d = np.full(x.shape, np.inf)
    best_id = np.zeros(x.shape, np.int64)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            ii, jj = ic + di, jc + dj
            jx = _hash01(seed, 501, ii, jj)
            jy = _hash01(seed, 502, ii, jj)
            sx = (ii + 0.15 + 0.7 * jx) * _COUNTRY_CELL
            sy = (jj + 0.15 + 0.7 * jy) * _COUNTRY_CELL
            d = (x - sx) ** 2 + (y - sy) ** 2
            hid = _hash_u64(seed, 503, ii, jj).astype(np.int64) & np.int64(0x7FFFFFFF)
            take = d < best_d
            best_d = np.where(take, d, best_d)
            best_id = np.where(take, hid, best_id)
    return best_id


# ---------------------------------------------------------------------------
# settlements — the shared lattice both the urban field and the drawn
# city/town/village entities read, so they always agree
# ---------------------------------------------------------------------------

_SETTLE = [
    # name, size_m (footprint), lattice cell_m, base probability
    ("city",    6000.0, 90000.0, 0.55),
    ("town",    1500.0, 26000.0, 0.45),
    ("village",  380.0,  9000.0, 0.40),
]
_SETTLE_SALT = {"city": 601, "town": 611, "village": 621}


def _habitability(seed, xs, ys):
    """Coarse-resolution suitability for settlement (0..1)."""
    e = elevation01(seed, xs, ys, 2000.0)
    m = moisture01(seed, xs, ys, 2000.0)
    t = temperature_c(seed, xs, ys, 2000.0, e)
    land = (e >= SEA + 0.002) & (e < 0.8)
    ok_t = np.clip(1.0 - np.abs(t - 14.0) / 22.0, 0.0, 1.0)
    ok_m = np.clip((m - 0.22) / 0.3, 0.0, 1.0)
    return land * ok_t * ok_m


def _settle_site(seed, kind_name, ii, jj):
    """Deterministic site of settlement kind in lattice cell (ii, jj).
    Returns (exists, sx, sy, id01) — all vectorized."""
    name, size_m, cell_m, prob = next(s for s in _SETTLE if s[0] == kind_name)
    salt = _SETTLE_SALT[kind_name]
    r = _hash01(seed, salt, ii, jj)
    jx = _hash01(seed, salt + 1, ii, jj)
    jy = _hash01(seed, salt + 2, ii, jj)
    sx = (ii + 0.2 + 0.6 * jx) * cell_m
    sy = (jj + 0.2 + 0.6 * jy) * cell_m
    hab = _habitability(seed, sx, sy)
    exists = r < prob * hab
    return exists, sx, sy, _hash_u64(seed, salt + 3, ii, jj)


_OFFS = np.array([(di, dj) for dj in (-1, 0, 1) for di in (-1, 0, 1)], np.int64)


def urban_at(seed, xs, ys, tile_m=0.0):
    """Urban influence [0,1] + nearest settlement info at points.

    Returns (urban, best_kind_idx, best_name_hash, best_dist_m);
    best_kind_idx: -1 none, else index into _SETTLE.

    Cost discipline: sites are computed once per UNIQUE lattice cell the
    points touch (at deep zoom that is one cell), then broadcast back —
    so the habitability fields behind each site are evaluated O(unique
    cells), not O(points × 9 neighbors)."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    shape = np.broadcast(xs, ys).shape
    xf = np.broadcast_to(xs, shape).ravel()
    yf = np.broadcast_to(ys, shape).ravel()
    n = xf.size
    urban = np.zeros(n)
    bk = np.full(n, -1, np.int64)
    bh = np.zeros(n, np.uint64)
    bd = np.full(n, np.inf)
    for kidx, (kname, size_m, cell_m, _prob) in enumerate(_SETTLE):
        radius = size_m * 0.62
        if radius < tile_m * 0.25:
            continue                        # influence is sub-tile: invisible
        ic = np.floor(xf / cell_m).astype(np.int64)
        jc = np.floor(yf / cell_m).astype(np.int64)
        uc, inv = np.unique(np.stack([ic, jc], 1), axis=0, return_inverse=True)
        ci = (uc[:, 0:1] + _OFFS[None, :, 0]).ravel()
        cj = (uc[:, 1:2] + _OFFS[None, :, 1]).ravel()
        ex, sx, sy, hid = _settle_site(seed, kname, ci, cj)
        ex = ex.reshape(-1, 9)[inv]
        sxp = sx.reshape(-1, 9)[inv]
        syp = sy.reshape(-1, 9)[inv]
        hidp = hid.reshape(-1, 9)[inv]
        d = np.hypot(xf[:, None] - sxp, yf[:, None] - syp)
        urban = np.maximum(urban, (np.clip(1.0 - d / radius, 0.0, 1.0) * ex).max(1))
        dm = np.where(ex > 0, d, np.inf)
        col = dm.argmin(1)
        rows = np.arange(n)
        dmin = dm[rows, col]
        take = (dmin < bd) & (dmin < radius * 2.5)
        bd = np.where(take, dmin, bd)
        bk = np.where(take, kidx, bk)
        bh = np.where(take, hidp[rows, col], bh)
    return (urban.reshape(shape), bk.reshape(shape),
            bh.reshape(shape), bd.reshape(shape))


# ---------------------------------------------------------------------------
# entity kinds — the ladder's inhabitants (exterior world only)
# ---------------------------------------------------------------------------
# Each kind: intrinsic size (m), lattice cell (m), color, category,
# interactable flag, and a density rule evaluated at candidate positions.
# Aggregated below ~half-tile footprint; expanded above it.

CAT_COLORS = {
    "structure": (168, 150, 134),
    "vegetal":   (70, 140, 70),
    "animal":    (206, 120, 60),
    "mineral":   (140, 140, 148),
    "object":    (196, 170, 92),
    "water":     (80, 130, 190),
    "player":    (240, 220, 80),
}

# Each kind's GEOMETRY is a fixed integer ratio (w, d, h): footprint width x
# depth x HEIGHT, in units of size/w meters. Footprints on the tile grid are
# always the ratio times an integer multiplier s, so a cubic house reads
# 1x1 -> 2x2 -> 3x3 under zoom and a 2:1 field reads 2x1 -> 4x2 -> 6x3 —
# the ratio is conserved, and (kind, s) names a FINITE set of sprite
# canvases for texture generation. Height occupies tiles ABOVE the base
# (screen-north), so a tall tree covers its base plus h*s rows up.
#
# "desc" is the sprite prompt body (one individual); "group" is the phrase
# used when the kind is aggregated into a ground tile's cover texture.
# rule tags are dispatched in _density().
KINDS = [
    dict(name="city",     cat="structure", size=6000.0, cell=90000.0, color=(184, 178, 170), inter=False, rule="settlement", ratio=(1, 1, 0),
         desc="a dense old-town city district seen from above: hundreds of tightly packed medieval houses with clay-tile and slate roofs, winding streets, small market squares, a surrounding stone wall",
         group="a distant walled city, a dense mass of tiny rooftops and streets"),
    dict(name="town",     cat="structure", size=1500.0, cell=26000.0, color=(178, 170, 158), inter=False, rule="settlement", ratio=(1, 1, 0),
         desc="a market town seen from above: a few dozen timber-framed houses with clay roofs gathered around a central square, dirt roads radiating outward",
         group="a small market town of clustered clay rooftops"),
    dict(name="village",  cat="structure", size=380.0,  cell=9000.0,  color=(172, 160, 142), inter=False, rule="settlement", ratio=(1, 1, 0),
         desc="a small farming village seen from above: a handful of thatched cottages along a single dirt lane, fenced yards and vegetable gardens",
         group="a farming hamlet of thatched cottages"),
    dict(name="crop field", cat="vegetal", size=160.0,  cell=420.0,   color=(188, 178, 96),  inter=False, rule="farmland", ratio=(2, 1, 0),
         desc="a cultivated crop field seen from above: neat parallel furrows of ripe golden wheat, a low hedge border and a narrow footpath along one edge",
         group="a patchwork of cultivated wheat fields"),
    dict(name="wood",     cat="vegetal",   size=90.0,   cell=260.0,   color=(44, 96, 52),    inter=False, rule="woodland", ratio=(1, 1, 0),
         desc="a small dense wood seen from above: overlapping broadleaf canopies in several greens, deep understory shadow between the crowns",
         group="closed broadleaf forest canopy, overlapping treetops"),
    dict(name="building", cat="structure", size=13.0,   cell=44.0,    color=(158, 138, 120), inter=True,  rule="building", ratio=(1, 1, 1),
         desc="a small rustic stone-and-timber house with a steep pitched clay-tile roof, a heavy wooden door, two shuttered windows and a small stone chimney",
         group="a group of small rustic stone-and-timber houses with clay-tile roofs"),
    dict(name="boulder",  cat="mineral",   size=3.2,    cell=64.0,    color=(134, 130, 126), inter=False, rule="mountain", ratio=(1, 1, 1),
         desc="a massive weathered granite boulder with deep cracks, patches of pale lichen and moss at its base",
         group="a scatter of massive granite boulders"),
    dict(name="tree",     cat="vegetal",   size=7.0,    cell=16.0,    color=(38, 118, 48),   inter=False, rule="tree", ratio=(1, 1, 2),
         desc="a mature broadleaf oak tree: a thick gnarled trunk and a wide irregular canopy built from layered clusters of leaves, small patches of sky showing through",
         group="broadleaf tree canopy seen from above"),
    dict(name="bush",     cat="vegetal",   size=1.7,    cell=7.0,     color=(70, 130, 58),   inter=False, rule="veg", ratio=(1, 1, 1),
         desc="a rounded leafy shrub with dense small leaves and a few dark berries",
         group="low shrubland thicket"),
    dict(name="garden table", cat="object", size=1.6,   cell=40.0,    color=(150, 108, 62),  inter=True,  rule="garden", ratio=(2, 2, 1),
         desc="a rustic wooden garden table on four sturdy legs, made of weathered gray-brown planks with visible grain",
         group="rustic wooden garden furniture"),
    dict(name="deer",     cat="animal",    size=1.4,    cell=90.0,    color=(172, 122, 72),  inter=True,  rule="wildlife", ratio=(2, 1, 2),
         desc="a red deer standing in profile: slender legs, warm brown coat, pale rump, small antlers, alert ears",
         group="a grazing herd of red deer"),
    dict(name="rock",     cat="mineral",   size=0.9,    cell=9.0,     color=(142, 140, 138), inter=False, rule="rocky", ratio=(1, 1, 1),
         desc="a rough gray rock with angular facets and a dusting of dirt at its base",
         group="a rocky scatter of gray stones"),
    dict(name="rabbit",   cat="animal",    size=0.35,   cell=60.0,    color=(196, 176, 150), inter=True,  rule="wildlife", ratio=(1, 1, 1),
         desc="a small brown rabbit sitting upright with long ears and a white tail",
         group="rabbits grazing in the grass"),
    dict(name="grass tuft", cat="vegetal", size=0.28,   cell=1.1,     color=(96, 148, 60),   inter=False, rule="grass", ratio=(1, 1, 1),
         desc="a tuft of tall wild grass, a dozen blades bending slightly in the wind, small seed heads",
         group="a meadow of tall wild grass"),
    dict(name="book",     cat="object",    size=0.22,   cell=0.9,     color=(120, 60, 50),   inter=True,  rule="on_table", ratio=(3, 2, 1),
         desc="a leather-bound book lying closed: embossed dark red cover, brass clasp, yellowed page edges",
         group="scattered leather-bound books"),
    dict(name="flower",   cat="vegetal",   size=0.15,   cell=1.4,     color=(214, 108, 160), inter=True,  rule="meadow", ratio=(1, 1, 2),
         desc="a single wildflower on a thin green stem: five bright pink petals around a yellow center, two small leaves",
         group="a wildflower meadow in bloom"),
    dict(name="cup",      cat="object",    size=0.09,   cell=0.55,    color=(226, 222, 214), inter=True,  rule="on_table", ratio=(1, 1, 1),
         desc="a glazed ceramic cup, off-white with a thin blue rim and a small round handle",
         group="ceramic crockery on a table"),
    dict(name="mushroom", cat="vegetal",   size=0.08,   cell=2.4,     color=(206, 160, 96),  inter=True,  rule="damp", ratio=(1, 1, 1),
         desc="a small forest mushroom with a domed ochre cap speckled with pale spots, on a short pale stem",
         group="a cluster of forest mushrooms"),
    dict(name="anthill",  cat="structure", size=0.45,   cell=42.0,    color=(150, 118, 84),  inter=True,  rule="veg", ratio=(2, 2, 1),
         desc="a mound anthill built of fine dirt grains, several dark entrance holes, a few ants on its surface",
         group="ant mounds of fine dirt"),
    dict(name="pebble",   cat="mineral",   size=0.045,  cell=0.35,    color=(150, 146, 140), inter=False, rule="ground", ratio=(1, 1, 1),
         desc="a smooth rounded river pebble, gray with faint mineral banding",
         group="a gravel bed of smooth pebbles"),
    dict(name="leaf",     cat="vegetal",   size=0.05,   cell=0.45,    color=(120, 110, 44),  inter=False, rule="veg", ratio=(1, 1, 0),
         desc="a single fallen dry leaf with curled brittle edges, ochre and brown",
         group="fallen leaf litter"),
    dict(name="ant",      cat="animal",    size=0.011,  cell=0.09,    color=(60, 40, 30),    inter=True,  rule="ant", ratio=(2, 1, 1),
         desc="a black worker ant seen from above: three glossy body segments, six articulated legs, bent antennae",
         group="a swarming trail of black ants"),
    dict(name="sand grain", cat="mineral", size=0.0035, cell=0.016,   color=(190, 174, 140), inter=False, rule="ground", ratio=(1, 1, 1),
         desc="a single grain of quartz sand: an irregular translucent crystal with rounded edges, warm beige",
         group="coarse quartz sand"),
]
for _i, _k in enumerate(KINDS):
    _k["kid"] = _i + 1
KIND_BY_ID = {k["kid"]: k for k in KINDS}
PLAYER_KID = 1000
PLAYER_SIZE = 0.6
PLAYER_KIND = dict(
    name="player", cat="animal", size=PLAYER_SIZE, cell=None, kid=PLAYER_KID,
    color=(240, 220, 80), inter=True, rule="player", ratio=(1, 1, 3),
    desc=("the player character standing and facing south: a small adventurer "
          "in a hooded travel cloak over simple leather armor, a satchel on "
          "the hip and a wooden walking staff in one hand"),
    group="the player character")
KIND_BY_ID[PLAYER_KID] = PLAYER_KIND

# footprint multipliers are exact up to 8, then snap to a sparse ladder —
# (kind, s) stays a FINITE texture catalog even when an object fills the view
_S_LADDER = [10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192]


def _snap_s(x):
    """Quantize the ideal footprint multiplier (size / (tile_m * ratio_w))."""
    s = int(round(x))
    if s < 1:
        return 0                                  # sub-tile: aggregated
    if s <= 8:
        return s
    return min(_S_LADDER, key=lambda v: abs(v - x))

_MAX_CELLS = 120_000      # safety cap on lattice candidates per kind per view
_MAX_STAMPS = 6000        # safety cap on individuals stamped per kind per view


def _env_at(seed, xs, ys, res_m):
    """Habitat fields at candidate points, at coarse (cheap) resolution."""
    res = max(res_m, 24.0)
    e = elevation01(seed, xs, ys, res)
    m = moisture01(seed, xs, ys, res)
    t = temperature_c(seed, xs, ys, res, e)
    veg = vegetation01(e, t, m)
    urban, _, _, _ = urban_at(seed, xs, ys)
    land = (e >= SEA).astype(np.float64)
    return dict(e=e, m=m, t=t, veg=veg, urban=urban, land=land)


def _table_proximity(seed, xs, ys):
    """1 where the point sits on a garden table's top, else 0."""
    table = next(k for k in KINDS if k["name"] == "garden table")
    cell = table["cell"]
    ic = np.floor(np.asarray(xs) / cell).astype(np.int64)
    jc = np.floor(np.asarray(ys) / cell).astype(np.int64)
    on = np.zeros(np.asarray(xs).shape)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            ex, sx, sy, _hid = _kind_site(seed, table, ic + di, jc + dj)
            d = np.maximum(np.abs(xs - sx), np.abs(ys - sy))
            on = np.maximum(on, ((d < table["size"] * 0.5) & (ex > 0)).astype(np.float64))
    return on


def _density(seed, kind, xs, ys, env):
    """Probability that this kind's lattice cell is occupied, per point."""
    rule = kind["rule"]
    land, veg, urban = env["land"], env["veg"], env["urban"]
    if rule == "settlement":
        return None                                 # handled by _settle_site
    if rule == "building":
        return land * np.clip(urban, 0.0, 1.0) ** 1.4 * 0.9 + land * 0.004
    if rule == "farmland":
        band = 4.0 * urban * (1.0 - urban)          # ring around settlements
        return land * np.clip(band, 0.0, 1.0) * veg * 0.6
    if rule == "woodland":
        return land * np.clip(veg - 0.45, 0.0, 1.0) * 1.3 * (1.0 - urban)
    if rule == "garden":
        return land * np.clip(urban - 0.25, 0.0, 1.0) * 0.5
    if rule == "on_table":
        return _table_proximity(seed, xs, ys) * (0.9 if kind["name"] == "cup" else 0.5)
    if rule == "tree":
        return land * veg * 0.8 * (1.0 - urban * 0.85)
    if rule == "veg":
        return land * veg * 0.45
    if rule == "grass":
        return land * veg * 0.75
    if rule == "meadow":
        return land * veg * np.clip(1.0 - env["m"], 0.2, 1.0) * 0.4
    if rule == "damp":
        return land * np.clip(env["m"] - 0.55, 0.0, 1.0) * veg * 1.2
    if rule == "wildlife":
        return land * veg * 0.35 * (1.0 - urban)
    if rule == "mountain":
        return land * np.clip(env["e"] - 0.68, 0.0, 1.0) * 2.2
    if rule == "rocky":
        return land * (0.05 + np.clip(env["e"] - 0.6, 0.0, 1.0) * 0.8)
    if rule == "ground":
        return land * 0.35
    if rule == "ant":
        return land * (0.06 + veg * 0.35)
    return land * 0.1


def _kind_site(seed, kind, ii, jj):
    """Deterministic individual of `kind` in lattice cell (ii, jj):
    (exists01, x, y, id_hash). Settlements delegate to the shared lattice."""
    if kind["rule"] == "settlement":
        ex, sx, sy, hid = _settle_site(seed, kind["name"], ii, jj)
        return ex.astype(np.float64), sx, sy, hid
    salt = 7000 + kind["kid"] * 16
    r = _hash01(seed, salt, ii, jj)
    jx = _hash01(seed, salt + 1, ii, jj)
    jy = _hash01(seed, salt + 2, ii, jj)
    cell = kind["cell"]
    sx = (ii + 0.1 + 0.8 * jx) * cell
    sy = (jj + 0.1 + 0.8 * jy) * cell
    env = _env_at(seed, sx, sy, max(kind["size"] * 8.0, 24.0))
    p = _density(seed, kind, sx, sy, env)
    return (r < p).astype(np.float64), sx, sy, _hash_u64(seed, salt + 3, ii, jj)


def player_pos(seed):
    """Deterministic spawn: first hashed candidate on habitable land."""
    for k in range(4096):
        x = float(_hash01(seed, 9001, np.int64(k), np.int64(1))) * PLANET_M
        y = float(_hash01(seed, 9002, np.int64(k), np.int64(2))) * PLANET_M
        hab = float(_habitability(seed, np.float64(x), np.float64(y)))
        if hab > 0.35:
            return x, y
    return PLANET_M * 0.5, PLANET_M * 0.5


# ---------------------------------------------------------------------------
# the view — sample a window of nx × ny tiles at tile_m meters per tile
# ---------------------------------------------------------------------------

_state = {}


def _place_kind(seed, kind, x0, y0, x1, y1, tile_m):
    """All individuals of `kind` intersecting the window, as arrays
    (sx, sy, ids) — or None if the kind is aggregated at this scale."""
    win = max(x1 - x0, y1 - y0)
    if _snap_s(kind["size"] / (tile_m * kind["ratio"][0])) < 1 \
            or kind["size"] > win * 3.0:
        return None
    cell = kind["cell"]
    pad = kind["size"]
    i0 = math.floor((x0 - pad) / cell)
    i1 = math.floor((x1 + pad) / cell)
    j0 = math.floor((y0 - pad) / cell)
    j1 = math.floor((y1 + pad) / cell)
    ncell = (i1 - i0 + 1) * (j1 - j0 + 1)
    if ncell > _MAX_CELLS or ncell <= 0:
        return None
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1, dtype=np.int64),
                         np.arange(j0, j1 + 1, dtype=np.int64))
    ex, sx, sy, hid = _kind_site(seed, kind, ii.ravel(), jj.ravel())
    keep = ex > 0.5
    if not keep.any():
        return np.empty(0), np.empty(0), np.empty(0, np.uint64)
    sx, sy, hid = sx[keep], sy[keep], hid[keep]
    if sx.size > _MAX_STAMPS:
        sx, sy, hid = sx[:_MAX_STAMPS], sy[:_MAX_STAMPS], hid[:_MAX_STAMPS]
    return sx, sy, hid


def _stamp(g, kind, s, sx, sy, hid, tile_m, gx0, gy0):
    """Rasterize individuals as QUANTIZED sprite blocks.

    Footprint is ratio * s tiles (ratio conserved under zoom); height
    occupies h = ratio_h * s tile rows ABOVE the base (screen-north), the
    way an isometric sprite rises up the screen. The whole sprite canvas
    is w x (d + h) tiles — exactly the canvas its texture is generated at.
    Anchors snap to the absolute world tile lattice so panning slides and
    textures land on tile boundaries. Within a kind, southern entities are
    stamped last (painter's order); across kinds, large -> small."""
    ny, nx = g["kid"].shape
    rw, rd, rh = kind["ratio"]
    w, d, h = rw * s, rd * s, rh * s
    order = np.argsort(sy, kind="stable")            # north first
    for n in order:
        il = int(round(sx[n] / tile_m - w / 2.0)) - gx0
        jb = int(round(sy[n] / tile_m - d / 2.0)) - gy0   # top row of the BASE
        jt = jb - h                                       # top row incl height
        i0, i1 = max(0, il), min(nx, il + w)
        j0, j1 = max(0, jt), min(ny, jt + d + h)
        if i0 >= i1 or j0 >= j1:
            continue
        eid = np.int32(int(hid[n]) & 0x7FFFFFFF)
        var = float((int(hid[n]) & 0xFF) / 255.0)
        vv = np.arange(j0, j1) - jt                       # row within sprite
        uu = np.arange(i0, i1) - il                       # col within sprite
        g["kid"][j0:j1, i0:i1] = kind["kid"]
        g["eid"][j0:j1, i0:i1] = eid
        g["var"][j0:j1, i0:i1] = var
        g["sca"][j0:j1, i0:i1] = s
        g["spr_v"][j0:j1, i0:i1] = vv[:, None]
        g["spr_u"][j0:j1, i0:i1] = uu[None, :]
        g["part"][j0:j1, i0:i1] = np.where(vv[:, None] >= h, 1, 2)


def view(seed, cx, cy, tile_m, nx=GRID, ny=GRID):
    """Sample the world: nx × ny tiles of tile_m meters centered at (cx, cy).

    Snaps the window to the tile lattice (pan slides, never shimmers),
    computes the field + entity grids, stores them for describe()/render().
    Returns the snapped (cx, cy, tile_m)."""
    tile_m = float(min(max(tile_m, MIN_TILE_M), PLANET_M / max(nx, ny)))
    cx = round(float(cx) / tile_m) * tile_m % PLANET_M
    cy = round(float(cy) / tile_m) * tile_m % PLANET_M
    nx, ny = int(nx), int(ny)

    xs = cx + (np.arange(nx, dtype=np.float64) - nx / 2 + 0.5) * tile_m
    ys = cy + (np.arange(ny, dtype=np.float64) - ny / 2 + 0.5) * tile_m
    X, Y = np.meshgrid(np.mod(xs, PLANET_M), np.mod(ys, PLANET_M))

    e = elevation01(seed, X, Y, tile_m)
    m = moisture01(seed, X, Y, tile_m)
    t = temperature_c(seed, X, Y, tile_m, e)
    bio = biome_ids(e, t, m)
    veg = vegetation01(e, t, m)
    det = detail01(seed, X, Y, tile_m)
    country = country_at(seed, X, Y)
    urban, ubk, ubh, ubd = urban_at(seed, X, Y, tile_m)

    g = dict(kid=np.zeros((ny, nx), np.int32),
             eid=np.zeros((ny, nx), np.int32),
             var=np.zeros((ny, nx), np.float64),
             sca=np.zeros((ny, nx), np.int32),
             spr_u=np.zeros((ny, nx), np.int32),
             spr_v=np.zeros((ny, nx), np.int32),
             part=np.zeros((ny, nx), np.int8))
    x0, y0 = cx - nx / 2 * tile_m, cy - ny / 2 * tile_m
    x1, y1 = cx + nx / 2 * tile_m, cy + ny / 2 * tile_m
    gx0 = int(round(x0 / tile_m))
    gy0 = int(round(y0 / tile_m))
    win = max(x1 - x0, y1 - y0)
    for kind in sorted(KINDS, key=lambda k: -k["size"]):
        placed = _place_kind(seed, kind, x0, y0, x1, y1, tile_m)
        if placed is None:
            continue
        sx, sy, hid = placed
        if sx.size:
            s = _snap_s(kind["size"] / (tile_m * kind["ratio"][0]))
            _stamp(g, kind, s, sx, sy, hid, tile_m, gx0, gy0)

    # the player is one more entity, at a fixed world position; like every
    # kind it stops being an entity when it outgrows the window (background)
    px, py = _state.get("player") or _state.setdefault("player", player_pos(seed))
    s = _snap_s(PLAYER_SIZE / tile_m)
    reach = PLAYER_SIZE * (1 + PLAYER_KIND["ratio"][2])
    if (s >= 1 and PLAYER_SIZE <= win * 3.0
            and x0 - reach <= px <= x1 + reach and y0 - reach <= py <= y1 + reach):
        _stamp(g, PLAYER_KIND, s, np.array([px]), np.array([py]),
               np.array([1], np.uint64), tile_m, gx0, gy0)

    # ---- ground texture keys: quantize what each tile CONTAINS -------------
    # coverage fraction of an aggregated kind = density * (size/cell)^2 —
    # scale-free, so the same spot keeps the same cover class while zooming.
    cover_kid = np.zeros((ny, nx), np.int32)
    cover_frac = np.zeros((ny, nx))
    env = dict(e=e, m=m, t=t, veg=veg, urban=urban,
               land=(e >= SEA).astype(np.float64))
    for kind in KINDS:
        if kind["rule"] in ("settlement", "on_table"):
            continue
        if _snap_s(kind["size"] / (tile_m * kind["ratio"][0])) >= 1:
            continue                                  # expanded, not cover
        if tile_m < kind["cell"]:
            continue                                  # rarer than one per tile
        p = _density(seed, kind, X, Y, env)
        frac = p * (kind["size"] / kind["cell"]) ** 2
        take = frac > cover_frac
        cover_frac = np.where(take, frac, cover_frac)
        cover_kid = np.where(take, kind["kid"], cover_kid)
    dens = np.zeros((ny, nx), np.int32)               # 0 bare 1 sparse 2 mid 3 dense
    dens[cover_frac >= 0.02] = 1
    dens[cover_frac >= 0.08] = 2
    dens[cover_frac >= 0.25] = 3
    cover_kid[dens == 0] = 0
    # one int32 texture id per tile: entity sprites (kind, s) or ground
    # (biome, cover kind, density) — np.unique of this grid IS the catalog
    tex = np.where(
        g["kid"] > 0,
        (1 << 28) | (g["kid"] << 12) | g["sca"],
        (2 << 28) | (bio.astype(np.int32) << 12) | (cover_kid << 4) | dens)

    _state.update(seed=int(seed), cx=cx, cy=cy, tile_m=tile_m, nx=nx, ny=ny,
                  e=e, m=m, t=t, bio=bio, veg=veg, det=det, country=country,
                  urban=urban, ubk=ubk, ubh=ubh, ubd=ubd,
                  kid=g["kid"], eid=g["eid"], var=g["var"], sca=g["sca"],
                  spr_u=g["spr_u"], spr_v=g["spr_v"], part=g["part"],
                  cover_kid=cover_kid, cover_dens=dens, tex=tex, X=X, Y=Y)
    return cx, cy, tile_m


# ---------------------------------------------------------------------------
# colorize layers + describe (the harness API)
# ---------------------------------------------------------------------------

LAYER_NAMES = ["composite", "biome", "elevation", "temperature", "moisture",
               "territory", "category", "entities", "texture"]


def _ramp(v, stops, colors):
    v = np.clip(np.asarray(v, np.float64), 0.0, 1.0)
    out = np.zeros(v.shape + (3,))
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        w = np.clip((v - a) / max(b - a, 1e-9), 0.0, 1.0)[..., None]
        seg = (v >= a)[..., None]
        col = (1 - w) * np.array(colors[i]) + w * np.array(colors[i + 1])
        out = np.where(seg, col, out)
    return out


def _shade(s):
    """Hillshade from the window's elevation + detail (unit-free)."""
    h = s["e"] * 26.0 + s["det"] * 1.1
    gy, gx = np.gradient(h)
    sh = 1.0 + (gx * 0.7 - gy * 0.5)
    return np.clip(sh, 0.62, 1.25)


def colorize(layer):
    s = _state
    bio = s["bio"]
    water = s["e"] < SEA
    if layer == "biome":
        rgb = _BIOME_LUT[bio]
    elif layer == "elevation":
        rgb = _ramp((s["e"] - 0.2) / 0.75,
                    [0.0, SEA * 0.79, SEA * 0.8, 0.6, 0.8, 1.0],
                    [(10, 20, 60), (60, 130, 180), (60, 110, 60),
                     (150, 140, 90), (130, 110, 100), (245, 245, 245)])
    elif layer == "temperature":
        rgb = _ramp((s["t"] + 25.0) / 60.0, [0.0, 0.5, 1.0],
                    [(58, 103, 196), (232, 228, 216), (200, 80, 46)])
    elif layer == "moisture":
        rgb = _ramp(s["m"], [0.0, 0.5, 1.0],
                    [(201, 163, 90), (214, 214, 196), (62, 143, 122)])
    elif layer == "territory":
        h = s["country"]
        rgb = np.stack([(h % 83) / 83.0, (h % 57) / 57.0, (h % 101) / 101.0],
                       axis=-1) * 140 + 80
        rgb[water] = (30, 44, 66)
    elif layer == "category":
        rgb = np.full(bio.shape + (3,), 46.0)
        rgb[water] = (24, 40, 66)
        for k in KINDS:
            mask = s["kid"] == k["kid"]
            rgb[mask] = CAT_COLORS[k["cat"]]
        rgb[s["kid"] == PLAYER_KID] = CAT_COLORS["player"]
    elif layer == "entities":
        rgb = np.full(bio.shape + (3,), 30.0)
        rgb[water] = (18, 28, 44)
        for k in KINDS:
            mask = s["kid"] == k["kid"]
            rgb[mask] = k["color"]
        rgb[s["kid"] == PLAYER_KID] = CAT_COLORS["player"]
    elif layer == "texture":
        # one color per texture KEY — the quantization made visible: tiles
        # sharing a color would share a generated texture
        h = _mix(s["tex"].astype(np.int64).view(np.uint64))
        rgb = np.stack([(h & _U(0xFF)).astype(np.float64),
                        ((h >> _U(8)) & _U(0xFF)).astype(np.float64),
                        ((h >> _U(16)) & _U(0xFF)).astype(np.float64)],
                       axis=-1) * 0.6 + 60.0
    else:                                            # composite
        rgb = _BIOME_LUT[bio].copy()
        tint = (s["det"][..., None] - 0.5) * 34.0
        rgb = rgb + tint
        rgb = rgb * _shade(_state)[..., None]
        # urban wash so settlements read at coarse zoom
        uw = np.clip(s["urban"], 0, 1)[..., None] * ~water[..., None]
        rgb = rgb * (1 - 0.35 * uw) + np.array((186, 178, 168)) * (0.35 * uw)
        for k in KINDS:
            mask = s["kid"] == k["kid"]
            if mask.any():
                v = (s["var"][mask, None] - 0.5) * 26.0
                rgb[mask] = np.clip(np.array(k["color"], np.float64) + v, 0, 255)
        pm = s["kid"] == PLAYER_KID
        rgb[pm] = CAT_COLORS["player"]
        # the part of a sprite that RISES above its base reads lighter, and
        # brightens toward the top row — cheap "tall things lean up-screen"
        up = s["part"] == 2
        if up.any():
            h_rows = np.maximum(s["spr_v"].max() + 1, 1)
            lift = 1.10 + 0.10 * (1.0 - s["spr_v"] / h_rows)
            rgb[up] = np.clip(rgb[up] * lift[up, None], 0, 255)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_rgba(layer):
    """(nx, ny, RGBA bytes) of the current view for the given layer."""
    rgb = colorize(layer)
    ny, nx = rgb.shape[:2]
    buf = np.empty((ny, nx, 4), np.uint8)
    buf[..., :3] = rgb
    buf[..., 3] = 255
    return nx, ny, buf.tobytes()


# ---------------------------------------------------------------------------
# texture keys & prompts — what a tile contains, quantized into a FINITE
# catalog of image-generation prompts (fed to the RunPod/ComfyUI pipeline)
# ---------------------------------------------------------------------------
# Two families of keys:
#   obj|<kind>|s<multiplier>            one sprite per (kind, footprint step);
#                                       ratio-conserving footprints mean a
#                                       cubic house is only ever n x n, so the
#                                       whole zoom range needs ~19 sizes/kind
#   ground|<biome>|<cover>|<density>    seamless terrain tile, optionally
#                                       carrying a GROUP texture (houses,
#                                       canopy, ants...) when its dominant
#                                       sub-tile content is dense enough

TILE_PX = 32          # pixels per tile when a texture is generated
DENS_NAMES = ["bare", "sparse", "scattered", "dense"]

_BIOME_GROUND = {
    "deep ocean": "deep open ocean water: dark blue swell, faint foam streaks",
    "ocean": "coastal sea water: mid-blue with light wave crests and subtle depth variation",
    "sea ice": "cracked pale sea ice floes over dark water, thin snow drifts",
    "beach": "wet golden beach sand with ripple marks and a few tiny shells",
    "snow": "a fresh snow field, soft drifts, faint blue shadows in the hollows",
    "tundra": "mossy tundra: low lichen mats, patches of bare frozen soil and small stones",
    "taiga": "boreal forest floor: dark conifer needles, moss patches, thin roots",
    "forest": "temperate forest floor: brown leaf litter, soft dark soil, scattered twigs",
    "rainforest": "lush tropical undergrowth: broad wet leaves, deep green shadow, damp earth",
    "grassland": "a green meadow of short mixed grass with small color variations",
    "savanna": "dry golden savanna grass with cracked bare-earth patches",
    "desert": "wind-rippled dry desert sand and dust, a few small stones",
    "bare rock": "bare fractured gray rock and scree, sharp shards and dust",
}

_STYLE_OBJ_PREFIX = (
    "16-bit pixel art game sprite, classic SNES JRPG tileset style, isometric "
    "three-quarter top-down view, crisp single-pixel dark outline around the "
    "silhouette, limited 32-color palette with warm mid tones, flat ambient "
    "light with the sun from the north-west and soft shadows falling to the "
    "south-east, no anti-aliasing, careful hand-placed dithering for shading, ")

_STYLE_OBJ_SUFFIX = (
    " Fully transparent background, single isolated object, no ground plane, "
    "no drop shadow outside the silhouette, no text, no watermark, no border, "
    "no photorealism, no blur, no depth of field; keep the palette, outline "
    "weight, lighting direction and level of detail exactly consistent with "
    "the other sprites of this same tileset, so that every variation looks "
    "like an asset from the same video game.")

_STYLE_GROUND_PREFIX = (
    "16-bit pixel art terrain tile, classic SNES JRPG overworld tileset "
    "style, top-down three-quarter view, seamless and perfectly tileable on "
    "all four edges, limited 32-color palette with warm mid tones, flat "
    "ambient light from the north-west, no anti-aliasing, careful hand-placed "
    "dithering for texture, ")

_STYLE_GROUND_SUFFIX = (
    " The tile must repeat perfectly: no element may touch or cross the tile "
    "border, no vignette, no lighting gradient across the tile, no text, no "
    "watermark, no photorealism, no blur; keep the palette, dithering style "
    "and level of detail exactly consistent with the other terrain tiles of "
    "this same tileset, so that every variation tiles seamlessly next to the "
    "others.")

_KIND_BY_NAME = {k["name"]: k for k in KINDS}
_KIND_BY_NAME["player"] = PLAYER_KIND


def texture_key(i, j):
    """The texture key string for tile (i, j) of the current view."""
    return _decode_tex(int(_state["tex"][j, i]))


def _decode_tex(t):
    fam = t >> 28
    if fam == 1:
        kid = (t >> 12) & 0xFFFF
        s = t & 0xFFF
        return f"obj|{KIND_BY_ID[kid]['name']}|s{s}"
    bio_id = (t >> 12) & 0xFFFF
    ck = (t >> 4) & 0xFF
    dens = t & 0xF
    cover = KIND_BY_ID[ck]["name"] if ck else "bare"
    return f"ground|{BIOMES[bio_id][0]}|{cover}|{DENS_NAMES[dens]}"


def texture_prompt(key):
    """The full image-generation prompt for a texture key. Long and precise
    on purpose: the style prefix pins the look, the body describes content
    and exact tile proportions, the suffix pins consistency and negatives —
    so variations of one key and neighbors in the catalog match each other."""
    parts = key.split("|")
    if parts[0] == "obj":
        kind = _KIND_BY_NAME[parts[1]]
        s = int(parts[2][1:])
        rw, rd, rh = kind["ratio"]
        w, d, h = rw * s, rd * s, rh * s
        cw, ch = w * TILE_PX, (d + h) * TILE_PX
        body = (f"{kind['desc']}. A {kind['cat']} object. It occupies a "
                f"footprint of {w} by {d} ground tiles"
                + (f" and rises {h} tiles tall" if h else ", lying flat")
                + f"; draw it to exactly fill a {cw} by {ch} pixel canvas "
                  f"({w} tiles wide, {d + h} tiles high at {TILE_PX} pixels "
                  f"per tile), its base resting on the bottom edge of the "
                  f"canvas, centered horizontally, the whole silhouette "
                  f"visible and nothing cropped.")
        return _STYLE_OBJ_PREFIX + body + _STYLE_OBJ_SUFFIX
    _g, biome, cover, dens = parts
    body = _BIOME_GROUND[biome]
    if cover != "bare":
        kind = _KIND_BY_NAME[cover]
        body += (f", {dens}ly covered by {kind['group']}, drawn small and "
                 f"seen from directly above as part of the ground")
    body += (f". One seamless ground tile of {TILE_PX} by {TILE_PX} pixels "
             f"(also valid rendered larger), uniform content edge to edge.")
    return _STYLE_GROUND_PREFIX + body + _STYLE_GROUND_SUFFIX


def texture_catalog():
    """Distinct texture keys of the CURRENT view with tile counts — what
    would have to exist in the store to skin this screen. The finiteness of
    this list under continuous zoom is the whole point of the quantization."""
    vals, counts = np.unique(_state["tex"], return_counts=True)
    out = [{"key": _decode_tex(int(v)), "tiles": int(c)}
           for v, c in sorted(zip(vals, counts), key=lambda vc: -vc[1])]
    return {"total": len(out), "keys": out}


def texture_catalog_json(with_prompts=False):
    cat = texture_catalog()
    if with_prompts:
        for k in cat["keys"]:
            k["prompt"] = texture_prompt(k["key"])
    return json.dumps(cat)


def _aggregates(seed, x, y, tile_m):
    """Closed-form estimates of what lives INSIDE one tile — the kinds too
    small to own tiles here. This is the 'zoom out aggregates' direction."""
    out = []
    for kind in sorted(KINDS, key=lambda k: -k["size"]):
        if _snap_s(kind["size"] / (tile_m * kind["ratio"][0])) >= 1:
            continue                                  # expanded, not aggregated
        if tile_m / kind["cell"] < 1.0:
            continue
        if kind["rule"] == "settlement":
            continue
        env = _env_at(seed, np.float64(x), np.float64(y),
                      max(kind["size"] * 8.0, 24.0))
        if kind["rule"] == "on_table":
            p = float(_table_proximity(seed, np.float64(x), np.float64(y)))
            p *= 0.9 if kind["name"] == "cup" else 0.5
        else:
            p = float(_density(seed, kind, np.float64(x), np.float64(y), env))
        est = p * (tile_m / kind["cell"]) ** 2
        if est < 0.5:
            continue
        if est > 2e5:
            label = "myriads"
        elif est > 1000:
            label = f"~{est / 1000:.0f}k"
        else:
            label = f"~{est:.0f}"
        out.append({"kind": kind["name"], "cat": kind["cat"], "count": label})
        if len(out) >= 7:
            break
    return out


def describe(i, j):
    """Inspector record for tile (i=column, j=row) of the current view."""
    s = _state
    i = int(min(max(i, 0), s["nx"] - 1))
    j = int(min(max(j, 0), s["ny"] - 1))
    x = float(s["X"][j, i])
    y = float(s["Y"][j, i])
    tile_m = s["tile_m"]
    lname, lidx = level_name(tile_m)
    e01 = float(s["e"][j, i])
    rec = {
        "tile": [i, j],
        "world_m": [round(x, 4), round(y, 4)],
        "tile_size": fmt_m(tile_m),
        "level": f"L{lidx} {lname}",
        "level_f": round(level_of(tile_m), 2),
        "biome": BIOMES[int(s["bio"][j, i])][0],
        "elevation": f"{elevation_m(e01):.5g} m",
        "temperature": f"{float(s['t'][j, i]):.1f} °C",
        "moisture": round(float(s["m"][j, i]), 2),
        "vegetation": round(float(s["veg"][j, i]), 2),
    }
    if e01 >= SEA:
        rec["territory"] = _gen_name(int(s["country"][j, i]))
        if float(s["urban"][j, i]) > 0.02 and int(s["ubk"][j, i]) >= 0:
            skind = _SETTLE[int(s["ubk"][j, i])][0]
            sname = _gen_name(int(s["ubh"][j, i]) & 0x7FFFFFFF)
            rec["settlement"] = f"{sname} ({skind}, {fmt_m(float(s['ubd'][j, i]))} from center)"
    kid = int(s["kid"][j, i])
    if kid > 0:
        k = KIND_BY_ID[kid]
        sc = int(s["sca"][j, i])
        rw, rd, rh = k["ratio"]
        w, d, h = rw * sc, rd * sc, rh * sc
        height_m = k["size"] * rh / rw
        rec["entity"] = {
            "name": k["name"], "category": k["cat"],
            "size": fmt_m(k["size"]), "interactable": k["inter"],
            "id": int(s["eid"][j, i]),
            "height": fmt_m(height_m) if rh else "flat",
            "tiles": f"{w}×{d} base" + (f" + {h} up" if h else ""),
            "part": "base" if int(s["part"][j, i]) == 1 else "upper",
            "sprite_cell": [int(s["spr_u"][j, i]), int(s["spr_v"][j, i])],
        }
    tkey = texture_key(i, j)
    rec["texture"] = {"key": tkey, "prompt": texture_prompt(tkey)}
    rec["contains"] = _aggregates(s["seed"], x, y, tile_m)
    return rec


def describe_json(i, j):
    return json.dumps(describe(i, j))


def find_player(seed):
    px, py = player_pos(seed)
    _state["player"] = (px, py)
    return px, py


def reset(seed):
    """New world: clear cached per-seed state (the player spawn)."""
    _state.clear()
    _state["player"] = player_pos(seed)
    return _state["player"]
