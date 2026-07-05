"""
hydrology.py — RIVERS. The planet's drainage (worldgen.compute_hydrology) is
traced ONCE into a TREE of vector polylines — junction to junction, each vertex
carrying its discharge. Every view then renders that same world-space geometry:

  width    real hydraulic geometry (w ≈ k·√drainage-area), so rivers are
           hairlines from orbit and only resolve to many pixels wide at
           genuinely deep zoom — never inflated "worms";
  shape    Chaikin-smoothed at build (kills the D8 staircase), then refined
           per view by deterministic midpoint displacement seeded from the
           segment's WORLD coordinates: zooming adds meanders exactly the
           way terrain adds octaves, identically for every window;
  valleys  the network CARVES the heightfield (in every window, at every
           zoom), so terrain and rivers agree and hillshade shows drainage;
  detail   past planet resolution, a window-local D8 runs on the refined,
           carved elevation — the carved trunks act as drains, so the small
           streams that appear are REAL drainage of the refined terrain,
           not decoration.
"""

import numpy as np

from worldgen import SEA_REF, _corner_hash
from common import _shift_max8

PLANET_KM = 4000.0      # map width; fixes the physical meaning of one cell
RIVER_W_KM = 0.0035     # hydraulic width: w[km] ≈ this · sqrt(drainage[km²])
CARVE_DEPTH = 0.012     # valley depth (elevation units) for a threshold river
MEANDER = 0.30          # midpoint displacement as a fraction of segment length
BROOK_MIN = 170         # window cells a local stream must drain to be drawn


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
    return a, acc.astype(np.float32)
