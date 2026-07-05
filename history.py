"""
history.py — Society: the M3 history simulation. A coarse cellular automaton
(population, food capacity, faction influence, unrest; deterministic pest /
drought / ice shocks) is INTEGRATED ONCE at build time into a keyframed
timeline; rendering then only interpolates that timeline, so history stays
fully seekable even though the underlying process is stateful.
"""

import numpy as np

from common import YEAR_DAYS, smoothstep, _neigh4

HIST_SIZE = 48            # coarse simulation grid (cells per side)
WEEK_DAYS = 7.0           # one CA step = one week
HIST_YEARS = 24.0         # how much history to pre-integrate
KF_WEEKS = 3              # record a keyframe every N weeks
POP_MAX = 1.2             # population value that maps to a full uint8


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


# ---- M3: seed factions, generate events, integrate the timeline ------------
def build_history(ws, civ_count, seed):
    """Run the CA over the horizon and store its keyframes on `ws`
    (hist_days/pop/own/stress/unrest + civ_cores, has_history)."""
    if civ_count <= 0:
        ws.civ_cores = []
        return
    H = HIST_SIZE
    e = ws._coarse(ws.elev)
    m = ws._coarse(ws.moist)
    sea0 = 0.42
    yn = (np.arange(H, dtype=np.float32) / H)[:, None]
    lat = np.repeat(1 - np.abs(yn - 0.5) * 2, H, axis=1)
    land = e >= sea0
    temp = np.clip(lat - np.clip(e - sea0, 0, 1) * 0.9, 0, 1)
    warmth = np.clip((temp - 0.26) / 0.55, 0, 1)
    wet = smoothstep((m - 0.1) / 0.8)
    flora0 = warmth * (0.30 + 0.70 * wet) * land
    water = np.clip(ws._coarse(ws.log_accum) * 0.7, 0, 1)
    temperate = np.clip(1 - np.abs(temp - 0.62) / 0.45, 0, 1)
    hab = (0.42 * flora0 + 0.30 * water + 0.28 * temperate) * land
    hab *= np.clip(1 - (e - 0.75) / 0.25, 0, 1)
    cap0 = np.clip(0.15 + 0.95 * hab, 0.0, 1.2) * land       # food capacity
    passable = land * np.clip(1 - np.clip((e - 0.7) / 0.3, 0, 1) * 0.7, 0.1, 1)

    # ---- pick faction cores: habitability maxima, spaced apart --------------
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
        ws.civ_cores = []
        return
    nf = len(cores)

    # ---- generate deterministic shock events over the horizon ---------------
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

    ws.has_history = True
    ws.hist_days = np.array(kf_days, np.float32)
    ws.hist_pop = np.stack(kf_pop)
    ws.hist_own = np.stack(kf_own).astype(np.int16)
    ws.hist_stress = np.stack(kf_stress)
    ws.hist_unrest = np.stack(kf_unrest)
    # normalized founding coords for city markers (yn, xn, faction, founded-day)
    ws.civ_cores = [(cy / H, cx / H, f, float(t0s[f]))
                    for f, (cy, cx) in enumerate(cores)]
