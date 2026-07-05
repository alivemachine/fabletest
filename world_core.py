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

from worldgen import (elevation_field, moisture_field, compute_hydrology,
                      elevation_window, moisture_window, noise_window,
                      BIOME_COLORS, SEA_REF, _corner_hash)


# Fixed per-seed cloud normalization: sampled ONCE from a coarse whole-torus
# view and reused for every window — exactly like worldgen._planet_norm does for
# elevation/moisture. Without a fixed range each window rescales to its own local
# min/max, so the SAME world point is a different brightness in the thumbnail,
# the main view, and every pan — that is the seam/brightness-pumping bug.
_CLOUD_NORM = {}


def _cloud_norm(seed, base_period):
    key = (int(seed), int(base_period))
    hit = _CLOUD_NORM.get(key)
    if hit is None:
        f = noise_window(seed, 0.5, 0.5, 1.0, 96, base_period=base_period)
        hit = (float(f.min()), float(np.ptp(f) or 1.0))
        _CLOUD_NORM[key] = hit
    return hit


def _cloud_sheet(seed, cx, cy, span, size, base_period):
    """A [0,1] cloud-cover sheet, windowed like the terrain fields and using a
    FIXED per-seed normalization at every zoom — so adjacent windows agree at
    their seams and the brightness never pumps as you pan or dive in."""
    f = noise_window(seed, cx, cy, span, size, base_period=base_period)
    lo, ptp = _cloud_norm(seed, base_period)
    return np.clip((f - lo) / ptp, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# RIVERS. The planet's drainage (worldgen.compute_hydrology) is traced ONCE
# into a TREE of vector polylines — junction to junction, each vertex carrying
# its discharge. Every view then renders that same world-space geometry:
#
#   width    real hydraulic geometry (w ≈ k·√drainage-area), so rivers are
#            hairlines from orbit and only resolve to many pixels wide at
#            genuinely deep zoom — never inflated "worms";
#   shape    Chaikin-smoothed at build (kills the D8 staircase), then refined
#            per view by deterministic midpoint displacement seeded from the
#            segment's WORLD coordinates: zooming adds meanders exactly the
#            way terrain adds octaves, identically for every window;
#   valleys  the network CARVES the heightfield (in every window, at every
#            zoom), so terrain and rivers agree and hillshade shows drainage;
#   detail   past planet resolution, a window-local D8 runs on the refined,
#            carved elevation — the carved trunks act as drains, so the small
#            streams that appear are REAL drainage of the refined terrain,
#            not decoration.
# ===========================================================================

PLANET_KM = 4000.0      # map width; fixes the physical meaning of one cell
RIVER_W_KM = 0.0035     # hydraulic width: w[km] ≈ this · sqrt(drainage[km²])
CARVE_DEPTH = 0.012     # valley depth (elevation units) for a threshold river
MEANDER = 0.30          # midpoint displacement as a fraction of segment length
BROOK_MIN = 170         # brook-grid cells a local stream must drain to be drawn
BROOK_TILE_SPAN = 0.125  # brooks are DEFINED once at this fixed world scale:
                         # drainage has a real minimum catchment, so branches
                         # stop multiplying past it — deeper zoom magnifies
                         # the same brooks instead of inventing new ones
BROOK_TILE_MC = 40       # margin cells of D8 context beyond each tile edge


def _net_threshold(planet_size):
    """Min accumulation (cells) traced into the vector network. Scales with
    cell count so it is the same PHYSICAL drainage area at any resolution."""
    return max(12.0, 48.0 * (planet_size / 256.0) ** 2)


def _width_world(acc, planet_size):
    """Channel width in world units from drainage area, via w ≈ k·√A."""
    return RIVER_W_KM * np.sqrt(np.maximum(acc, 1.0)) / planet_size


def _chaikin(x, y, a, passes=2):
    """Corner-cutting smoothing; endpoints pinned so junctions stay connected."""
    for _ in range(passes):
        if x.size < 3:
            break
        def one(v):
            out = np.empty(2 * v.size, v.dtype)
            out[0], out[-1] = v[0], v[-1]
            out[1:-1:2] = 0.75 * v[:-1] + 0.25 * v[1:]
            out[2:-1:2] = 0.25 * v[:-1] + 0.75 * v[1:]
            return out
        x, y, a = one(x), one(y), one(a)
    return x, y, a


def _extract_network(parent, accum, size, thr, sea):
    """Trace the D8 forest into junction-to-junction polylines. Returns a list
    of (x, y, acc) arrays in UNWRAPPED world coords (a polyline may run past
    [0,1) across the torus seam; the rasterizer tests wrapped copies)."""
    n = size * size
    acc = accum.reshape(-1)
    sea_f = sea.reshape(-1)
    chan = (acc > thr) & (~sea_f)
    node = np.arange(n)
    moving = chan & (parent != node)
    inflow = np.bincount(parent[moving], minlength=n)   # channel tributaries in
    junc = chan & (inflow >= 2)
    starts = np.where((chan & (inflow == 0)) | junc)[0]
    edges = []
    for s0 in starts:
        cur = int(s0)
        xs = [(cur % size + 0.5) / size]
        ys = [(cur // size + 0.5) / size]
        aa = [float(acc[cur])]
        for _ in range(4 * n):
            nxt = int(parent[cur])
            if nxt == cur:
                break
            ddx = ((nxt % size) - (cur % size) + size // 2) % size - size // 2
            ddy = ((nxt // size) - (cur // size) + size // 2) % size - size // 2
            xs.append(xs[-1] + ddx / size)
            ys.append(ys[-1] + ddy / size)
            aa.append(float(acc[nxt]))
            cur = nxt
            if sea_f[cur] or junc[cur] or not chan[cur]:
                break
        if len(xs) >= 2:
            x, y, a = _chaikin(np.asarray(xs), np.asarray(ys),
                               np.asarray(aa, np.float64))
            edges.append((x.astype(np.float32), y.astype(np.float32),
                          a.astype(np.float32)))
    return edges


def _refine_polyline(x, y, a, levels, planet_size, seed):
    """Deterministic fractal meanders: midpoints displaced perpendicular by a
    hash of their WORLD position, recursively. The same world curve falls out
    of every window and zoom; deeper zoom just reveals more levels."""
    for lvl in range(levels):
        dx, dy = np.diff(x), np.diff(y)
        q = np.float64(planet_size) * (2 << lvl)
        mx, my = x[:-1] + dx * 0.5, y[:-1] + dy * 0.5
        h = _corner_hash(seed + 13, 91 + lvl,
                         np.round(mx * q).astype(np.int64),
                         np.round(my * q).astype(np.int64)) - 0.5
        mx = mx - dy * h * MEANDER
        my = my + dx * h * MEANDER
        def weave(v, m):
            out = np.empty(v.size + m.size, v.dtype)
            out[0::2], out[1::2] = v, m
            return out
        x = weave(x, mx)
        y = weave(y, my)
        a = weave(a, 0.5 * (a[:-1] + a[1:]))
    return x, y, a


def _stroke_capsule(alpha, disc, x0, y0, x1, y1, rad, a_vis, value, size):
    """One wide river segment: antialiased coverage into `alpha`, discharge
    into `disc`."""
    x_lo = max(0, int(np.floor(min(x0, x1) - rad)))
    x_hi = min(size, int(np.ceil(max(x0, x1) + rad)) + 1)
    y_lo = max(0, int(np.floor(min(y0, y1) - rad)))
    y_hi = min(size, int(np.ceil(max(y0, y1) + rad)) + 1)
    if x_hi <= x_lo or y_hi <= y_lo:
        return
    ys = np.arange(y_lo, y_hi)[:, None].astype(np.float32)
    xs = np.arange(x_lo, x_hi)[None, :].astype(np.float32)
    dx, dy = x1 - x0, y1 - y0
    ll = dx * dx + dy * dy
    if ll < 1e-9:
        t = np.zeros((1, 1), np.float32)
    else:
        t = np.clip(((xs - x0) * dx + (ys - y0) * dy) / ll, 0.0, 1.0)
    dist = np.sqrt((xs - (x0 + t * dx)) ** 2 + (ys - (y0 + t * dy)) ** 2)
    cover = np.clip(rad - dist + 0.5, 0.0, 1.0)
    sub_a = alpha[y_lo:y_hi, x_lo:x_hi]
    np.maximum(sub_a, a_vis * cover, out=sub_a)
    sub_d = disc[y_lo:y_hi, x_lo:x_hi]
    np.maximum(sub_d, np.where(cover > 0.2, value, 0.0).astype(np.float32),
               out=sub_d)


def _splat_thin(alpha, disc, carve, segs, size):
    """Batch-draw thin (≲1.5px) stroke segments and ALL carve centerlines as
    resampled points with bilinear max-splat — one vectorized pass for the
    whole network instead of a Python loop per segment."""
    X0, Y0, X1, Y1, A, V, D, W = (np.concatenate(s) for s in zip(*segs))
    L = np.hypot(X1 - X0, Y1 - Y0)
    m = np.maximum(1, np.ceil(L / 0.6).astype(np.int32))
    eid = np.repeat(np.arange(m.size), m)
    base = np.repeat(np.cumsum(m) - m, m)
    t = (np.arange(eid.size) - base + 0.5) / m[eid]
    xs = X0[eid] + t * (X1 - X0)[eid]
    ys = Y0[eid] + t * (Y1 - Y0)[eid]
    av, vv = A[eid], V[eid]
    dv, thin = D[eid], (W[eid] <= 1.5)
    xi = np.floor(xs - 0.5).astype(np.int64)
    yi = np.floor(ys - 0.5).astype(np.int64)
    fx, fy = xs - 0.5 - xi, ys - 0.5 - yi
    af, df_, cf = alpha.reshape(-1), disc.reshape(-1), carve.reshape(-1)
    for ddx, ddy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        w = (fx if ddx else 1 - fx) * (fy if ddy else 1 - fy)
        gx, gy = xi + ddx, yi + ddy
        ok = (gx >= 0) & (gx < size) & (gy >= 0) & (gy < size)
        idx = gy[ok] * size + gx[ok]
        wk = w[ok]
        np.maximum.at(cf, idx, (wk * dv[ok]).astype(np.float32))
        tm = thin[ok]
        np.maximum.at(af, idx[tm], (wk[tm] * av[ok][tm]).astype(np.float32))
        hit = tm & (wk > 0.25)
        np.maximum.at(df_, idx[hit], vv[ok][hit].astype(np.float32))


def _shift_max8(a):
    """8-neighbor max WITHOUT torus wrap (windows are not periodic)."""
    p = np.pad(a, 1)
    m = a
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            m = np.maximum(m, p[dy:dy + a.shape[0], dx:dx + a.shape[1]])
    return m


def _stroke_field(net, cx, cy, span, size, planet_size, seed):
    """Render the vector network for window (cx,cy,span) at size².
    Returns (alpha, disc, carve):
      alpha  stroke coverage in [0,1] (already faded for sub-pixel widths)
      disc   discharge (planet cells) under the stroke, for gating/color
      carve  valley depth to SUBTRACT from the window's elevation."""
    alpha = np.zeros((size, size), np.float32)
    disc = np.zeros((size, size), np.float32)
    carve = np.zeros((size, size), np.float32)
    if not net:
        return alpha, disc, carve
    inv = size / span
    left, top = cx - span / 2.0, cy - span / 2.0
    ppp = inv / planet_size                     # screen px per planet cell
    # refine until segments are ~0.8px (build segments are ~0.25 planet cells),
    # so meanders start appearing as soon as a channel spans multiple pixels
    levels = int(np.clip(np.ceil(np.log2(max(ppp / 3.2, 1e-9))), 0, 6))
    thr = _net_threshold(planet_size)
    pad = 14.0 / inv
    thin_segs = []
    for bx, by, ba in net:
        o0 = int(np.ceil(left - pad - bx.max()))
        o1 = int(np.floor(left + span + pad - bx.min()))
        p0 = int(np.ceil(top - pad - by.max()))
        p1 = int(np.floor(top + span + pad - by.min()))
        if o1 < o0 or p1 < p0:
            continue
        rx, ry, ra = _refine_polyline(bx.astype(np.float64),
                                      by.astype(np.float64),
                                      ba.astype(np.float64),
                                      levels, planet_size, seed)
        w_px = (_width_world(ra, planet_size) * inv).astype(np.float64)
        dep = np.clip(CARVE_DEPTH * (ra / thr) ** 0.25, 0.0,
                      3.0 * CARVE_DEPTH)
        for ox in range(o0, o1 + 1):
            for oy in range(p0, p1 + 1):
                X = ((rx + ox - left) * inv)
                Y = ((ry + oy - top) * inv)
                sw = np.maximum(w_px[:-1], w_px[1:])
                sv = np.maximum(ra[:-1], ra[1:])
                sa = 0.32 + 0.68 * np.clip(sw / 0.9, 0.0, 1.0)
                sd = np.maximum(dep[:-1], dep[1:])
                mgn = 2.0 + sw
                vis = ((np.minimum(X[:-1], X[1:]) < size + mgn) &
                       (np.maximum(X[:-1], X[1:]) > -mgn) &
                       (np.minimum(Y[:-1], Y[1:]) < size + mgn) &
                       (np.maximum(Y[:-1], Y[1:]) > -mgn))
                if not vis.any():
                    continue
                thin_segs.append((X[:-1][vis], Y[:-1][vis],
                                  X[1:][vis], Y[1:][vis],
                                  sa[vis], sv[vis], sd[vis], sw[vis]))
                for k in np.where(vis & (sw > 1.5))[0]:
                    _stroke_capsule(alpha, disc, X[k], Y[k], X[k + 1],
                                    Y[k + 1], sw[k] * 0.5, 1.0, sv[k], size)
    if thin_segs:
        _splat_thin(alpha, disc, carve, thin_segs, size)
    # widen the carved centerline into a valley: world-anchored radius
    # (~1.5 planet cells on screen), exponential cross-profile
    r = int(np.clip(round(1.5 * ppp), 1, 12))
    for _ in range(r):
        carve = np.maximum(carve, 0.78 * _shift_max8(carve))
    return alpha, disc, carve


# ---------------------------------------------------------------------------
# Window-local drainage: past planet resolution the vector net has no more
# detail, so run the SAME hydrology (fill -> D8 -> accumulation) on the
# window's refined, valley-carved elevation. Trunk channels, lakes, the sea
# and the window border are the drains; anything ≥ BROOK_MIN cells becomes a
# visible brook. Pixel-snapped viewports (see WorldSlice.view) make this a
# pure function of the window, so panning slides it rigidly.
# ---------------------------------------------------------------------------

def _shift_min8(a):
    p = np.pad(a, 1, constant_values=np.inf)
    m = np.full_like(a, np.inf)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            m = np.minimum(m, p[dy:dy + a.shape[0], dx:dx + a.shape[1]])
    return m


def _box3(a):
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def _local_streams(elev, moist, trunk_alpha, lake_lv, cx, cy, span, seed,
                   max_iters=96):
    size = elev.shape[0]
    e = elev.astype(np.float64)
    # de-bias the local D8 exactly like the planet's: a per-cell hash jitter,
    # keyed by the cell's WORLD coordinate (the viewport is pixel-snapped, so
    # panning sees the same jitter and the brooks slide rigidly). Amplitude
    # tracks the LOCAL relief so broad uniform slopes wander too, instead of
    # locking into parallel 45° runs.
    q = size / span
    step = (np.arange(size, dtype=np.float64) / size - 0.5) * span
    qx = np.round((cx + step) * q).astype(np.int64)[None, :]
    qy = np.round((cy + step) * q).astype(np.int64)[:, None]
    gy, gx = np.gradient(e)
    relief = _box3(_box3(np.abs(gx) + np.abs(gy)))
    amp = 1.4 * relief + 0.25 * float(relief.mean())
    e = e + (_corner_hash(seed + 77, 37, qx, qy).astype(np.float64) - 0.5) * amp
    seeds = (trunk_alpha > 0.30) | (e < SEA_REF) | ((lake_lv > 0) & (e < lake_lv))
    seeds[0, :] = seeds[-1, :] = seeds[:, 0] = seeds[:, -1] = True
    f = np.where(seeds, e, np.inf)
    for _ in range(max_iters):                       # bounded local pit fill
        nf = np.maximum(e, np.minimum(f, _shift_min8(f) + 1e-7))
        if np.array_equal(nf, f):
            break
        f = nf
    f = np.where(np.isinf(f), e, f)
    # local D8 (non-torus)
    yy, xx = np.divmod(np.arange(size * size), size)
    parent = np.arange(size * size)
    best = f.reshape(-1).copy()
    p = np.pad(f, 1, constant_values=np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nval = p[1 + dy:1 + dy + size, 1 + dx:1 + dx + size].reshape(-1)
            nidx = np.clip(yy + dy, 0, size - 1) * size + np.clip(xx + dx, 0, size - 1)
            better = nval < best
            parent = np.where(better, nidx, parent)
            best = np.where(better, nval, best)
    node = np.arange(size * size)
    stop = seeds.reshape(-1)                    # water stops at its drains
    src = node[(parent != node) & ~stop]
    dst = parent[src]
    acc = np.ones(size * size, np.float64)
    for i in range(max_iters):
        new = 1.0 + np.bincount(dst, weights=acc[src], minlength=size * size)
        if i % 8 == 7 and np.array_equal(new, acc):
            break
        acc = new
    acc = acc.reshape(size, size)
    brook = (acc >= BROOK_MIN) & ~seeds
    a = np.zeros((size, size), np.float32)
    a[brook] = np.clip(0.20 + 0.13 * np.log2(acc[brook] / BROOK_MIN + 1.0),
                       0.0, 0.55)
    # drainage density follows rainfall: deserts carry few perennial brooks
    a *= np.clip((moist - 0.12) / 0.45, 0.06, 1.0).astype(np.float32)
    return a, acc.astype(np.float32), parent

# ---------------------------------------------------------------------------
# Time model (M2). t is measured in sim DAYS.
# ---------------------------------------------------------------------------
YEAR_DAYS = 96.0          # one year = 96 days
TIDE_PERIOD = 0.52        # ~semi-diurnal tide
FAUNA_PERIOD = 32.0       # predator-prey limit cycle length (days)
NORMAL_RELIEF_WORLD = 0.018
SHADOW_RELIEF_WORLD = 0.090
SUN_LEAN = 0.35           # fixed southward lean of the sun path (flat map):
                          # shadows always point one way and never vanish at
                          # noon, so the cast-shadow field stays usable as a
                          # global shadow map for sprite-stack lighting
SUN_SEASON_LEAN = 0.30    # how far season_off (+/-0.5) swings that lean
TERRAIN_SHADOW_STEPS = 56
CLOUD_WORLD_HEIGHT = 0.010
CLOUD_SHADOW_STRENGTH = 0.50

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


def biome_ids(e, t, m, sea, tide=0.0, lake_lv=None, g=None):
    """Vectorized version of worldgen.classify_biomes -> int16 ids.

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
    bid = np.select(conds, choices, default=BID["jungle"]).astype(np.int16)
    if g is not None:
        # sub-biome refinement: the ground channel splits each broad biome
        # into distinct, tile-able ground types (each its own color code the
        # game can key on). Same octave noise as terrain -> zoom-coherent.
        def sub(base, mask, name):
            bid[(bid == BID[base]) & mask] = BID[name]
        hi, vhi, lo = g > 0.54, g > 0.60, g < 0.44
        sub("grassland", vhi, "tall_grass")
        sub("grassland", hi & ~vhi, "meadow")
        sub("grassland", lo & (m > 0.55), "wheat_soil")
        sub("savanna", vhi, "acacia_scrub")
        sub("desert", vhi & (m > 0.19), "oasis")       # rare wet pockets first
        sub("desert", hi & (m > 0.13), "shrub_steppe")
        sub("desert", hi, "reg_rock")
        sub("desert", lo, "dunes")
        sub("forest", vhi, "glade")
        sub("forest", lo, "dark_forest")
        sub("jungle", g > 0.62, "jungle_clear")
        sub("tundra", hi, "rocky_tundra")
        sub("mountain", lo, "scree")
    return bid


def temperature_t(elev, lat, lat_signed, sea_eff, season_off):
    t = lat + season_off * lat_signed - np.clip(elev - sea_eff, 0, None) * 0.9
    return np.clip(t, 0, 1)


def _sample_offset(a, ox, oy, fill):
    """Sample `a` with a non-wrapping integer offset: out[y,x] = a[y+oy,x+ox]."""
    out = np.full_like(a, fill)
    h, w = a.shape
    if abs(ox) >= w or abs(oy) >= h:
        return out
    sx0, sx1 = max(0, ox), min(w, w + ox)
    sy0, sy1 = max(0, oy), min(h, h + oy)
    dx0, dx1 = max(0, -ox), min(w, w - ox)
    dy0, dy1 = max(0, -oy), min(h, h - oy)
    out[dy0:dy1, dx0:dx1] = a[sy0:sy1, sx0:sx1]
    return out


def _sun_dir(sun_x, season_off):
    """ONE sun direction (east/south/up) for the whole map at this instant.

    The map is a flat plane, so every point shares the same sun — there is no
    lit side / dark side. Day and night are a transition over TIME for the
    entire map at once: fraction-of-day .0 is midnight, .25 sunrise in the
    east, .5 noon, .75 sunset in the west. All spatial variation in the light
    comes from slopes (normal map), terrain cast shadows, and cloud shadows —
    together, the global shadow map."""
    hour = (sun_x - 0.5) * (2 * np.pi)            # 0 at noon
    sx = -np.sin(hour)                            # east at dawn -> west at dusk
    sz = np.cos(hour)                             # up: +1 noon, -1 midnight
    sy = SUN_LEAN + SUN_SEASON_LEAN * season_off
    inv = 1.0 / np.sqrt(sx * sx + sy * sy + sz * sz)
    return float(sx * inv), float(sy * inv), float(sz * inv)


def _terrain_shadow(height, sx, sy, sz, pixel_world):
    """Approximate terrain cast-shadow visibility in [0,1] for one sun ray."""
    if sz <= 1e-4:
        return np.zeros_like(height, np.float32)
    horiz = float(np.hypot(sx, sy))
    if horiz <= 1e-6:
        return np.ones_like(height, np.float32)
    dx, dy = sx / horiz, sy / horiz
    rise = pixel_world * sz / horiz
    soft = max(pixel_world * SHADOW_RELIEF_WORLD * 1.8, 1e-5)
    horizon = np.full_like(height, -1e9, np.float32)
    steps = min(max(height.shape) - 1,
                max(12, min(TERRAIN_SHADOW_STEPS,
                            int(12 + 34 * horiz / max(sz, 0.12)))))
    for step in range(1, steps + 1):
        ox = int(round(dx * step))
        oy = int(round(dy * step))
        if ox == 0 and oy == 0:
            continue
        sample = _sample_offset(height, ox, oy, -1e9)
        np.maximum(horizon, sample - step * rise, out=horizon)
    return smoothstep((height - horizon + soft) / (2 * soft)).astype(np.float32)


def _cloud_shadow(clouds, sx, sy, sz, pixel_world):
    """Project cloud cover onto the ground along the sun ray."""
    if sz <= 1e-4:
        return np.ones_like(clouds, np.float32)
    scale = CLOUD_WORLD_HEIGHT / max(sz * pixel_world, 1e-6)
    ox = int(round(sx * scale))
    oy = int(round(sy * scale))
    cover = _sample_offset(clouds, ox, oy, 0.0).astype(np.float32)
    cover = 0.65 * cover + 0.35 * _shift_max8(cover)
    return np.clip(1.0 - CLOUD_SHADOW_STRENGTH * cover,
                   1.0 - CLOUD_SHADOW_STRENGTH, 1.0).astype(np.float32)


def _lighting_fields(ws, sun_x, season_off, day_night, clouds, sea_eff):
    """Derived lighting payload: normals, sun visibility, and shadow masks.

    One sun direction lights the whole frame; ndotl (the normal map), the
    terrain cast shadows and the cloud shadows all use that same vector, so
    they agree by construction. Lighting sees the WATER SURFACE, not the
    seabed: elevation is clamped to sea level before deriving normals and
    shadow heights, so the sea is a flat plane, relief flattens exactly at
    the waterline, and mountains cast shadows onto the water."""
    sx, sy, sz = _sun_dir(sun_x, season_off)
    e_lit = np.maximum(ws.elev, np.float32(sea_eff))
    gy, gx = np.gradient(e_lit, ws.pixel_world, ws.pixel_world)
    nx = -gx * NORMAL_RELIEF_WORLD
    ny = -gy * NORMAL_RELIEF_WORLD
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    ndotl = np.clip((nx * sx + ny * sy + sz) * inv, 0, 1).astype(np.float32)
    height = (e_lit * SHADOW_RELIEF_WORLD).astype(np.float32)
    day = float(smoothstep((sz + 0.10) / 0.20))   # whole-map dusk/dawn fade
    terrain_vis = _terrain_shadow(height, sx, sy, max(sz, 0.0),
                                  ws.pixel_world)
    cloud_vis = _cloud_shadow(clouds, sx, sy, max(sz, 0.08), ws.pixel_world)
    direct = ndotl * terrain_vis * cloud_vis
    lit = day * (0.28 + 0.72 * direct)
    floor = 1.0 - 0.72 * day_night
    sunlight = floor + (1.0 - floor) * lit
    nz = np.ones_like(e_lit, np.float32)
    return {
        "normal": np.stack((nx * inv, ny * inv, nz * inv),
                           axis=2).astype(np.float32),
        "sun_dir": np.array([sx, sy, sz], np.float32),
        "sun_up": np.float32(max(sz, 0.0)),
        "terrain_shadow": terrain_vis,
        "cloud_shadow": cloud_vis,
        "sunlight": np.clip(sunlight, 0, 1).astype(np.float32),
    }


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


# Wind velocity in WORLD units per day (fraction of the torus width). Advection
# is a shift of the SAMPLE CENTER in world coordinates — not a roll of the window
# array — so clouds drift the same physical speed at every zoom and translate
# rigidly with the window as you pan. No window-edge wrap, no whole-pixel jumps,
# no racing across the frame at deep zoom.
CLOUD_WIND1_X = 1.0 / 6.0
CLOUD_WIND2_X = 1.0 / 11.0
CLOUD_WIND2_Y = 1.0 / 40.0
# Clouds are a soft, blurred cover mask, so the noise is sampled on a grid this
# many times coarser than the render and bilinear-upscaled. The dropped octaves
# are finer than the mask's own blur (invisible), and it cuts the per-frame
# noise-hash cost — the dominant animation cost — by ~this factor squared.
CLOUD_DOWNSCALE = 4
CLOUD_MIN_RES = 24


def _bilinear_upsample(a, size):
    """Upscale a small (h, h) float field to (size, size), bilinear, align-corners.
    Separable: interpolate rows then columns. Returns float32."""
    h = a.shape[0]
    if h >= size:
        return a.astype(np.float32)
    t = np.linspace(0.0, h - 1.0, size, dtype=np.float32)
    i0 = np.floor(t).astype(np.int64)
    i1 = np.minimum(i0 + 1, h - 1)
    f = (t - i0).astype(np.float32)
    rows = a[i0] * (1.0 - f)[:, None] + a[i1] * f[:, None]        # (size, h)
    return (rows[:, i0] * (1.0 - f)[None, :]
            + rows[:, i1] * f[None, :]).astype(np.float32)         # (size, size)


def clouds_field(ws, t):
    """Cloud cover in [0,1]: two noise sheets drifting with the wind at
    different speeds. Pure, seekable function of t, fully decoupled from the
    terrain below — no moisture gating, no orographic (slope) term, just a mask
    that moves. Re-sampled per frame at time-advected WORLD centers, so the
    cover stays coherent under pan/zoom in both the browser console and Godot.

    Computed on a downscaled grid and bilinear-upsampled: the mask is soft
    enough that the finest octaves never show, and the noise hash — the hot
    spot of the whole animation loop — runs on ~CLOUD_DOWNSCALE² fewer pixels."""
    size = ws.elev.shape[0]
    cs = max(CLOUD_MIN_RES, -(-size // CLOUD_DOWNSCALE))    # ceil(size / scale)
    cs = min(cs, size)
    c1 = _cloud_sheet(ws.seed + 4001, ws.cx - t * CLOUD_WIND1_X, ws.cy,
                      ws.span, cs, 4)
    c2 = _cloud_sheet(ws.seed + 8009, ws.cx - t * CLOUD_WIND2_X,
                      ws.cy - t * CLOUD_WIND2_Y, ws.span, cs, 3)
    sheet = 0.6 * c1 + 0.4 * c2
    mask = smoothstep((sheet - 0.46) / 0.30)
    return _bilinear_upsample(mask, size)


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


def _history_coarse(ws, t):
    """Interpolate the coarse HIST timeline at day t. Returns
    (pop, faction_id, stress, unrest) at HIST_SIZE x HIST_SIZE — the summary
    grids both the upsampled layers and the M4 settlement expand() read."""
    days = ws.hist_days
    tc = float(np.clip(t, days[0], days[-1]))
    i = int(np.searchsorted(days, tc, side="right") - 1)
    i = max(0, min(i, len(days) - 2))
    span = days[i + 1] - days[i]
    fr = 0.0 if span <= 0 else (tc - days[i]) / span
    pop = ((1 - fr) * ws.hist_pop[i] + fr * ws.hist_pop[i + 1]) * (POP_MAX / 255.0)
    stress = ((1 - fr) * ws.hist_stress[i] + fr * ws.hist_stress[i + 1]) / 255.0
    unrest = ((1 - fr) * ws.hist_unrest[i] + fr * ws.hist_unrest[i + 1]) / 255.0
    own = ws.hist_own[i] if fr < 0.5 else ws.hist_own[i + 1]
    return (pop.astype(np.float32), own,
            stress.astype(np.float32), unrest.astype(np.float32))


def _sample_history(ws, t):
    """Interpolate the coarse history timeline at day t and upsample to the
    render grid. Returns (pop, faction_id, stress, unrest), all render-sized."""
    size = ws.elev.shape[0]
    if not getattr(ws, "has_history", False):
        z = np.zeros((size, size), np.float32)
        return z, np.full((size, size), -1, np.int16), z, z

    pop_c, own_c, stress_c, unrest_c = _history_coarse(ws, t)

    # map each render pixel to its coarse history cell via WORLD coordinates,
    # so the timeline lines up with the (possibly zoomed) window on screen.
    up = _window_indices(ws)
    return (pop_c[up].astype(np.float32), own_c[up].astype(np.int16),
            stress_c[up].astype(np.float32), unrest_c[up].astype(np.float32))


# ---------------------------------------------------------------------------
# M4 expand() — society. A coarse history cell's summary (faction, population,
# stress) resolves under zoom into settlements: sites on a hashed lattice,
# buildings laid out along hashed street lanes. Like the rivers, NOTHING is
# stored — the same world-space geometry is re-derived for whatever window
# looks at it, so every visit to a village finds the same village.
# ---------------------------------------------------------------------------
SETTLE_GRID = 192        # settlement-site lattice (16 candidate sites/HIST cell)
SETTLE_SPAN = 0.35       # windows narrower than this resolve settlements
SETTLE_POP_MIN = 0.05    # coarse population where the first hamlet appears
SETTLE_MAX_SITES = 400   # budget governor: most-populous sites expand first
BUILD_WORLD = 0.00035    # building footprint, world units


def _settlements(ws, t, sea_level):
    """Rasterize the settlements the coarse history implies for this window.

    Returns (alpha, rgb), render-sized; alpha is 1 on building footprints.
    Sites exist where their HIST cell's population clears a hash-staggered
    threshold, so villages appear one by one as a region fills (and vanish if
    it empties). All geometry is keyed on (seed, lattice cell): deterministic,
    windowless, O(visible cells). Cached per (window, sim day)."""
    size = ws.size
    key = (round(ws.cx, 9), round(ws.cy, 9), round(ws.span, 9), size,
           int(t), round(float(sea_level), 3))
    cached = getattr(ws, "_settle_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    alpha = np.zeros((size, size), np.float32)
    rgb = np.zeros((size, size, 3), np.float32)
    if getattr(ws, "has_history", False) and ws.span < SETTLE_SPAN:
        # summary grids at the START of the current sim day (stable per day)
        pop_c, own_c, stress_c, _ = _history_coarse(ws, float(int(t)) + 0.5)
        G = SETTLE_GRID
        left, top = ws.cx - ws.span / 2, ws.cy - ws.span / 2
        pad = 1.0 / G
        i0 = int(np.floor((left - pad) * G))
        i1 = int(np.ceil((left + ws.span + pad) * G))
        j0 = int(np.floor((top - pad) * G))
        j1 = int(np.ceil((top + ws.span + pad) * G))
        gi = np.arange(i0, i1 + 1, dtype=np.int64)[None, :] % G   # cols (x)
        gj = np.arange(j0, j1 + 1, dtype=np.int64)[:, None] % G   # rows (y)
        h_ex = _corner_hash(ws.seed + 101, 0, gi, gj)
        h_jx = _corner_hash(ws.seed + 101, 1, gi, gj)
        h_jy = _corner_hash(ws.seed + 101, 2, gi, gj)
        wx = (gi.astype(np.float32) + 0.2 + 0.6 * h_jx) / G       # site world x
        wy = (gj.astype(np.float32) + 0.2 + 0.6 * h_jy) / G       # site world y
        hi = (wy * HIST_SIZE).astype(np.int64) % HIST_SIZE
        hj = (wx * HIST_SIZE).astype(np.int64) % HIST_SIZE
        pop = pop_c[hi, hj]
        own = own_c[hi, hj]
        stress = stress_c[hi, hj]
        # staggered founding: each site has its own hashed threshold, so a
        # filling cell lights its hamlets up one at a time, biggest cells first
        exists = (own >= 0) & (pop > SETTLE_POP_MIN * (0.6 + 2.8 * h_ex))
        ys, xs = np.nonzero(exists)
        if len(ys) > SETTLE_MAX_SITES:                 # budget governor
            order = np.argsort(-pop[ys, xs])[:SETTLE_MAX_SITES]
            ys, xs = ys[order], xs[order]
        px_w = size / ws.span                          # pixels per world unit
        bpx = max(1, int(round(BUILD_WORLD * px_w)))
        lake = np.asarray(getattr(ws, "lake_lv", 0.0))
        ra = getattr(ws, "river_alpha", None)
        wall = np.array([228, 214, 186], np.float32)   # sun-dried plaster

        def place(sxw, syw, p, fid, dim, hsite, hx, hy):
            """One settlement: 2-4 street lanes, two building rows per lane.
            Every quantity is hashed from (hsite, building index) — the same
            village on every visit. Buildings never stand in water or on a
            river."""
            fx = ((sxw - left) % 1.0) / ws.span
            fy = ((syw - top) % 1.0) / ws.span
            if fx >= 1.0 + pad / ws.span or fy >= 1.0 + pad / ws.span:
                return
            exi = min(size - 1, max(0, int(fx * size)))
            eyi = min(size - 1, max(0, int(fy * size)))
            lakev = float(lake[eyi, exi]) if lake.ndim == 2 else float(lake)
            if ws.elev[eyi, exi] < max(sea_level + 0.004, lakev):
                return                                 # site must stand on land
            n = 3 + int(min(p, 1.2) * 30)              # hamlet -> town
            lanes = 2 + (hsite % 3)
            rings = max(1, -(-n // (lanes * 2)))       # ceil
            spacing = (0.42 / G) / (rings + 1)
            base_a = 2 * np.pi * ((hsite >> 7) & 1023) / 1024.0
            tint = CIV_COLORS[min(max(fid, 0), len(CIV_COLORS) - 1)]
            k = 0
            for ring in range(1, rings + 1):
                for ln in range(lanes):
                    for side in (-1.0, 1.0):
                        if k >= n:
                            break
                        hb = float(_corner_hash(ws.seed + 505, k,
                                                np.int64(hx), np.int64(hy)))
                        a = base_a + ln * (2 * np.pi / lanes) + (hb - 0.5) * 0.3
                        d = ring * spacing * (0.8 + 0.4 * hb)
                        off = side * (1.1 + 0.9 * hb) * BUILD_WORLD
                        bx = sxw + np.cos(a) * d - np.sin(a) * off
                        by = syw + np.sin(a) * d + np.cos(a) * off
                        x0 = int(((bx - left) % 1.0) / ws.span * size)
                        y0 = int(((by - top) % 1.0) / ws.span * size)
                        k += 1
                        if x0 < 0 or y0 < 0 or x0 >= size or y0 >= size:
                            continue
                        lakev = (float(lake[y0, x0]) if lake.ndim == 2
                                 else float(lake))
                        if ws.elev[y0, x0] < max(sea_level + 0.003, lakev):
                            continue                   # never build in water
                        if ra is not None and ra[y0, x0] > 0.35:
                            continue                   # nor on the river
                        col = (0.42 * tint + 0.58 * wall) * (0.6 + 0.4 * hb) * dim
                        x1, y1 = min(size, x0 + bpx), min(size, y0 + bpx)
                        alpha[y0:y1, x0:x1] = 1.0
                        rgb[y0:y1, x0:x1] = col

        # --- roads (deep zoom only): lattice-adjacent settlements of the same
        # faction link east & south (each pair draws once) with a hashed bend;
        # roads ford rivers but never cross the sea, and buildings paint over
        # them. Drawn first so villages sit ON their roads.
        if ws.span < 0.15:
            road_col = np.array([124, 104, 78], np.float32)

            def road(ax, ay, bx, by, hh):
                mx = (ax + bx) / 2 + (hh - 0.5) * 0.30 / G
                my = (ay + by) / 2 + (0.5 - hh) * 0.30 / G
                for (p0x, p0y), (p1x, p1y) in (((ax, ay), (mx, my)),
                                               ((mx, my), (bx, by))):
                    seg = np.hypot(p1x - p0x, p1y - p0y)
                    ts = np.linspace(0, 1, max(2, int(seg * px_w * 1.6)))
                    xi = ((((p0x + (p1x - p0x) * ts) - left) % 1.0)
                          / ws.span * size).astype(np.int64)
                    yi = ((((p0y + (p1y - p0y) * ts) - top) % 1.0)
                          / ws.span * size).astype(np.int64)
                    ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
                    xi, yi = xi[ok], yi[ok]
                    if not len(xi):
                        continue
                    lv = lake[yi, xi] if lake.ndim == 2 else lake
                    ok = ws.elev[yi, xi] >= np.maximum(sea_level + 0.001, lv)
                    ok &= alpha[yi, xi] < 0.5          # don't repaint buildings
                    xi, yi = xi[ok], yi[ok]
                    alpha[yi, xi] = np.maximum(alpha[yi, xi], 0.75)
                    rgb[yi, xi] = road_col

            ex_set = {(int(y), int(x)) for y, x in zip(*np.nonzero(exists))}
            for y, x in sorted(ex_set):
                for dy, dx in ((0, 1), (1, 0)):
                    y2, x2 = y + dy, x + dx
                    if ((y2, x2) not in ex_set
                            or own[y, x] != own[y2, x2]):
                        continue
                    hh = float(_corner_hash(ws.seed + 707, dy * 2 + dx,
                                            gi[0, x], gj[y, 0]))
                    if hh > 0.72:                      # thin the mesh: not
                        continue                       # every neighbour links
                    road(float(wx[y, x]), float(wy[y, x]),
                         float(wx[y2, x2]), float(wy[y2, x2]), hh / 0.72)
            # each capital links to its nearest expanded sites (up to 3)
            for idx, (cyn, cxn, fid, t0) in enumerate(
                    getattr(ws, "civ_cores", [])):
                if t <= t0 or not ex_set:
                    continue
                cand = sorted(
                    ex_set,
                    key=lambda yx: (float(wx[yx]) - cxn) ** 2
                                   + (float(wy[yx]) - cyn) ** 2)[:3]
                for (y, x) in cand:
                    d2 = (float(wx[y, x]) - cxn) ** 2 + (float(wy[y, x]) - cyn) ** 2
                    if d2 > (2.5 / G) ** 2:
                        continue
                    road(float(cxn), float(cyn),
                         float(wx[y, x]), float(wy[y, x]), 0.3 + 0.4 * (idx % 2))

        for y, x in zip(ys, xs):
            hsite = ((int(gi[0, x]) * 73856093) ^ (int(gj[y, 0]) * 19349663)
                     ^ (ws.seed * 83492791)) & 0x7FFFFFFF
            place(float(wx[y, x]), float(wy[y, x]), float(pop[y, x]),
                  int(own[y, x]), 1.0 - 0.45 * float(stress[y, x]),
                  hsite, int(gi[0, x]), int(gj[y, 0]))

        # capitals: each faction's founding core is a GUARANTEED town at its
        # exact world position (the coarse cell around it may be mostly sea)
        for idx, (cyn, cxn, fid, t0) in enumerate(getattr(ws, "civ_cores", [])):
            if t <= t0:
                continue
            hi0 = int(cyn * HIST_SIZE) % HIST_SIZE
            hj0 = int(cxn * HIST_SIZE) % HIST_SIZE
            p = max(float(pop_c[hi0, hj0]), 0.55)      # a capital never decays
            dim = 1.0 - 0.45 * float(stress_c[hi0, hj0])
            hsite = ((997 + idx * 7919) ^ (ws.seed * 40503)) & 0x7FFFFFFF
            place(float(cxn), float(cyn), 1.4 * p, int(fid), dim,
                  hsite, 100000 + idx, 200000 + idx)
    ws._settle_cache = (key, alpha, rgb)
    return alpha, rgb


# ===========================================================================
class WorldSlice:
    """Static per-resolution data + grids (full res, or strided for thumbs)."""

    def __init__(self, elev, moist, hyd, civ_count, seed,
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
        # ground-detail channel for sub-biome classification: the same
        # windowed octave noise as the terrain, so patch borders gain fractal
        # detail with zoom instead of blurring (a wheat field keeps a crisp,
        # ever-finer edge at tile level)
        self.ground = noise_window(self.seed + 7777, self.cx, self.cy,
                                   self.span, size, base_period=24)
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

    # ---- brooks at a FIXED world scale ------------------------------------
    # The drainage below the vector tree is computed ONCE per world tile and
    # extracted as segments, so every window at every zoom rasterizes the
    # same geometry: branches stop multiplying at the tile scale's minimum
    # catchment (drainage density is physically finite), and panning/zooming
    # cannot rearrange them — deeper zoom just magnifies the same brooks.
    def _brook_tile(self, ti, tj):
        cache = getattr(self, "_brook_tiles", None)
        if cache is None:
            cache = self._brook_tiles = {}
        nt = int(round(1.0 / BROOK_TILE_SPAN))
        key = (int(ti) % nt, int(tj) % nt)
        hit = cache.get(key)
        if hit is not None:
            return hit
        tspan, tsize = BROOK_TILE_SPAN, self.size
        gsize = tsize + 2 * BROOK_TILE_MC
        gspan = tspan * gsize / tsize            # same cell size as the core
        tcx = (key[0] + 0.5) * tspan
        tcy = (key[1] + 0.5) * tspan
        elev = elevation_window(gsize, self.seed, tcx, tcy, gspan)
        alpha, _disc, carve = _stroke_field(self.net, tcx, tcy, gspan, gsize,
                                            self.planet_size, self.seed)
        elev = np.clip(elev - carve, 0.0, 1.0)
        moist = moisture_window(gsize, self.seed, tcx, tcy, gspan)
        lake = self._sample_planet(self.lake_level, tcx, tcy, gspan, gsize)
        a, acc, parent = _local_streams(elev, moist, alpha, lake,
                                        tcx, tcy, gspan, self.seed)
        # cell -> parent segments for brook cells OWNED by the tile core (the
        # margin only provides D8 context; neighbours own their own cells)
        idx = np.nonzero(a.reshape(-1) > 0)[0]
        par = parent[idx]
        y0, x0 = np.divmod(idx, gsize)
        y1, x1 = np.divmod(par, gsize)
        own = ((x0 >= BROOK_TILE_MC) & (x0 < BROOK_TILE_MC + tsize)
               & (y0 >= BROOK_TILE_MC) & (y0 < BROOK_TILE_MC + tsize))
        cell = tspan / tsize

        def rel(c):                              # tile-frame world coords
            return ((c - BROOK_TILE_MC + 0.5) * cell).astype(np.float32)

        val = (rel(x0[own]), rel(y0[own]), rel(x1[own]), rel(y1[own]),
               a.reshape(-1)[idx][own].astype(np.float32),
               acc.reshape(-1)[idx][own].astype(np.float32))
        cache[key] = val
        return val

    def _brook_window(self, cx, cy, span, size):
        """Rasterize the fixed-scale brook segments into a window: returns
        (alpha, acc), render-sized. Every zoom draws the SAME brooks."""
        alpha = np.zeros((size, size), np.float32)
        acc = np.zeros((size, size), np.float32)
        tspan = BROOK_TILE_SPAN
        left, top = cx - span / 2, cy - span / 2
        parts = []
        for tj in range(int(np.floor(top / tspan)),
                        int(np.floor((top + span) / tspan)) + 1):
            for ti in range(int(np.floor(left / tspan)),
                            int(np.floor((left + span) / tspan)) + 1):
                x0, y0, x1, y1, av, cv = self._brook_tile(ti, tj)
                if len(x0):
                    parts.append((x0 + ti * tspan, y0 + tj * tspan,
                                  x1 + ti * tspan, y1 + tj * tspan, av, cv))
        if not parts:
            return alpha, acc
        x0, y0, x1, y1, av, cv = (np.concatenate([p[i] for p in parts])
                                  for i in range(6))
        cell = tspan / self.size
        keep = ((np.minimum(x0, x1) < left + span + cell)
                & (np.maximum(x0, x1) > left - cell)
                & (np.minimum(y0, y1) < top + span + cell)
                & (np.maximum(y0, y1) > top - cell))
        if not keep.any():
            return alpha, acc
        x0, y0, x1, y1, av, cv = (v[keep] for v in (x0, y0, x1, y1, av, cv))
        # resample each segment at ~pixel steps and splat (crisp 1px lines)
        k = max(2, int(cell * size / span * 1.5) + 1)
        ts = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        xi = (((x0[None, :] + (x1 - x0)[None, :] * ts) - left)
              / span * size).astype(np.int64).ravel()
        yi = (((y0[None, :] + (y1 - y0)[None, :] * ts) - top)
              / span * size).astype(np.int64).ravel()
        aa = np.broadcast_to(av[None, :], (k, len(av))).ravel()
        cc = np.broadcast_to(cv[None, :], (k, len(cv))).ravel()
        ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
        xi, yi, aa, cc = xi[ok], yi[ok], aa[ok], cc[ok]
        np.maximum.at(alpha, (yi, xi), aa)
        np.maximum.at(acc, (yi, xi), cc)
        return alpha, acc

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
        # Bilinear planet sampling is only honest while the planet grid still
        # covers the output pixels (allow a 2x stretch). Past that, hand off
        # to the octave-refining window generator — otherwise every zoom up
        # to planet_size was serving BLUR instead of the finer octaves, and
        # tile-level detail was lost.
        if zoom * size > 2.0 * self.planet_size or size > self.size:
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
            # deep zoom: the planet-res drainage has run out of detail. The
            # brooks come from the FIXED-scale tile network (computed once
            # per world tile, cached), so every zoom level draws the SAME
            # branches — they stop multiplying at the tiles' minimum
            # catchment and cannot shift under pan/zoom.
            s.brook_alpha, acc_local = self._brook_window(cx, cy, span, size)
            # flow layer: lift local cell counts so a just-visible brook reads
            # like the smallest vector river (display scaling only)
            lift = _net_threshold(self.planet_size) / BROOK_MIN
            s.accum = np.maximum(disc, acc_local * lift)
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
        decay, diff, mig, conflict_mort = 0.986, 0.34, 0.12, 0.07
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

            # re-resolve ownership and contested borders -> attrition term
            maxI = I.max(0)
            if nf >= 2:
                secondI = np.sort(I, axis=0)[-2]
            else:
                secondI = np.zeros_like(maxI)
            owned = land & (maxI > own_thresh)
            own = np.where(owned, I.argmax(0), -1)
            contest = np.where(maxI > tiny, secondI / (maxI + 1e-6), 0.0) * owned
            P *= 1 - conflict_mort * contest

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
        # normalized founding coords (yn, xn, faction, founded-day) for the
        # city markers and the M4 capital settlements. The CA founds on
        # COARSE cells; snap each core to the nearest fine-grid pixel of
        # solid land so a coastal founding never leaves the capital standing
        # in the sea (or drowning at high tide).
        solid = np.argwhere(self.elev >= SEA_REF + 0.035)      # (y, x) pixels
        size = self.size
        self.civ_cores = []
        for f, (cy, cx) in enumerate(cores):
            py, px = (cy + 0.5) / H * size, (cx + 0.5) / H * size
            if len(solid):
                d2 = ((solid[:, 0] - py) ** 2 + (solid[:, 1] - px) ** 2)
                py, px = solid[np.argmin(d2)] + 0.5
            self.civ_cores.append((float(py) / size, float(px) / size,
                                   f, float(t0s[f])))
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


# ===========================================================================
# EcoSim — the STATEFUL near-form substrate (M4). Everything else in this file
# is a pure, seekable function of t. This one is not: it integrates forward and
# has MEMORY, so events leave lasting consequences. A flood salinates the soil;
# a drought dries the forest; recovery is slow and only spreads inward from
# neighbouring cells, so an isolated barren zone stays barren until life reaches
# it again — which is how a world can fail to recover. Driven live by the
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
        """Back to the pristine, climax-state world (day 0): full soil, and
        vegetation/fauna at the climate's potential, so health == 1 everywhere
        and the flora/fauna layers look exactly like their pure baseline until
        something happens to them."""
        self.sea_ref = float(self.sea0)
        self.t = 0.0
        land = (self.e >= self.sea0).astype(np.float32)
        warmth, wet = self._climate(0.0)
        clim = warmth * (0.30 + 0.70 * wet) * land          # climatic potential
        self.clim = clim.astype(np.float32)
        self.fert = land.copy()                             # climax soil = 1 on land
        self.veg = clim.astype(np.float32)                  # at full potential
        self.fauna = (0.6 * clim).astype(np.float32)
        self.civ = ((clim > 0.55) * 0.35).astype(np.float32)   # some seed settlements
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
        submerged = under & (e >= self.sea_ref)        # land the sea just covered
        clim = warmth * (0.30 + 0.70 * wet) * land       # climatic potential
        self.clim = clim.astype(np.float32)              # (for health readout)
        cap = clim * self.fert * (1 - self.scorch)
        dry = np.clip(warmth - 0.75 * wet - 0.05, 0, 1)  # hot & dry -> fire/desert

        veg, fauna, civ, fert, scorch = (self.veg, self.fauna, self.civ,
                                         self.fert, self.scorch)

        # --- flood: submerged biota declines, soil salinates ---
        veg = np.where(under, veg * np.exp(-0.6 * h), veg)
        fauna = np.where(under, fauna * np.exp(-0.5 * h), fauna)
        civ = np.where(under, civ * np.exp(-0.7 * h), civ)
        fert = np.where(submerged, fert - 0.008 * h, fert)
        scorch = np.where(submerged, scorch + 0.020 * h, scorch)

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
        #     cleared, isolated cells cannot restart until life spreads back in ---
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
        civ = np.where(food < 0.14, civ * np.exp(-0.12 * h), civ)  # food-collapse decline
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
                ("veg", "fauna", "fert", "civ", "scorch", "clim")}


def build_world(seed, size, civ_count=3):
    """The heavy, seed-only part: fields, hydrology, cloud sheets, M3 history."""
    elev = elevation_field(size, seed).astype(np.float32)
    moist = moisture_field(size, seed).astype(np.float32)
    hyd = compute_hydrology(elev, int(seed))   # fill -> D8 -> accum -> lakes
    ws = WorldSlice(elev, moist, hyd, int(civ_count), int(seed))
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
    # climate references the GEOGRAPHIC sea level: the tide moves the
    # waterline, not the snowline (mountains don't cool at high tide)
    tf = temperature_t(e, ws.lat, ws.lat_signed, sea_level, season_off)
    clouds = clouds_field(ws, t)
    st = {
        "ws": ws, "t": t, "sea_level": sea_level, "river_thr": river_thr,
        "tide_amp": tide_amp, "day_night": day_night,
        "sea_eff": sea_eff, "season_off": season_off, "sun_x": sun_x,
        "e": e, "land": e >= sea_eff, "tf": tf,
        "biome_id": biome_ids(e, tf, ws.moist, sea_level, tide_amp,
                              getattr(ws, "lake_lv", None),
                              getattr(ws, "ground", None)),
        "veg": flora_field(ws, tf, sea_level),
        "moist": ws.moist, "log_accum": ws.log_accum,
        "clouds": clouds,
    }
    st.update(_lighting_fields(ws, sun_x, season_off, day_night, clouds,
                               sea_eff))
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
        g = np.clip((e - sea_level) / max(1 - sea_level, 1e-6), 0, 1)
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
        # pure black & white cover mask: white = cloud, black = clear sky.
        # No terrain underlay, no lighting — this is the raw occlusion input
        # the shadow pass consumes.
        img = np.repeat((st["clouds"] * 255.0)[..., None], 3, axis=2)
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
        if ws.span < SETTLE_SPAN:
            sa, srgb = _settlements(ws, t, sea_level)
            a = sa[..., None]
            img = img * (1 - a) + srgb * a
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
            if ws.span < SETTLE_SPAN:
                # M4 expand(): settlements resolve out of the civ summary
                # under zoom, lit by the same sun as the terrain
                sa, srgb = _settlements(ws, t, sea_level)
                a = sa[..., None]
                img = img * (1 - a) + srgb * (0.25 + 0.75 * l)[..., None] * a
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
