"""
common.py — the time model and the few grid helpers every module shares.

Kept deliberately tiny: anything here is read by at least two of the layer
modules (hydrology, lighting, history, ecosim, world_core).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Time model (M2). t is measured in sim DAYS.
# ---------------------------------------------------------------------------
YEAR_DAYS = 96.0          # one year = 96 days
TIDE_PERIOD = 0.52        # ~semi-diurnal tide
FAUNA_PERIOD = 32.0       # predator-prey limit cycle length (days)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


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


def _neigh4(a):
    """4-neighbor mean on the torus (works for 2D or leading-axis stacks)."""
    ax = a.ndim - 2
    return 0.25 * (np.roll(a, 1, ax) + np.roll(a, -1, ax)
                   + np.roll(a, 1, ax + 1) + np.roll(a, -1, ax + 1))
