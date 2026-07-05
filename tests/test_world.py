"""Invariant tests for the world engine.

The engine's one non-negotiable property is determinism: the world is a pure
function of (seed, x, y, t). These tests pin that down — plus the structural
invariants (zoom coherence, drainage reaching the sea) that a silent bug can
break without anything crashing. The README records exactly such a bug: a D8
sign error that quietly zeroed flow accumulation for a while.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import worldgen as wg
import world_core as wc

SEED = 42


# ---------------------------------------------------------------------------
# Determinism: same seed -> the same world, forever. The pinned values are
# loose (1e-4) so a numpy upgrade's rounding noise passes but a real change
# to the noise stack fails.
# ---------------------------------------------------------------------------

def test_elevation_deterministic_golden():
    e = wg.elevation_window(64, SEED)
    assert e.shape == (64, 64) and e.dtype == np.float32
    assert e.mean() == pytest.approx(0.427693, abs=1e-4)
    assert float(e[0, 0]) == pytest.approx(0.021474, abs=1e-4)
    assert float(e[17, 43]) == pytest.approx(0.600804, abs=1e-4)
    assert float(e[50, 9]) == pytest.approx(0.406921, abs=1e-4)


def test_moisture_deterministic_golden():
    m = wg.moisture_window(64, SEED)
    assert m.mean() == pytest.approx(0.468258, abs=1e-4)


def test_same_seed_same_world():
    a = wg.elevation_window(96, 7, 0.3, 0.7, 0.5)
    b = wg.elevation_window(96, 7, 0.3, 0.7, 0.5)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Zoom coherence: the load-bearing property of noise_window. A window at
# zoom z samples THE SAME function as the whole planet — the central quarter
# of a 512² planet and a 128² window at zoom 4 hit coincident sample points
# with the same octave cutoff, so they must agree exactly.
# ---------------------------------------------------------------------------

def test_zoom_adds_detail_without_moving_the_planet():
    full = wg.elevation_window(512, SEED)
    win = wg.elevation_window(128, SEED, 0.5, 0.5, 0.25)
    crop = full[192:320, 192:320]
    assert np.abs(crop - win).max() < 1e-6


# ---------------------------------------------------------------------------
# Hydrology: after depression filling, EVERY cell's D8 path must terminate in
# the sea (that is what fill_depressions is for). Path-doubling resolves every
# cell to its final sink in log2 steps.
# ---------------------------------------------------------------------------

def test_every_cell_drains_to_the_sea():
    elev = wg.elevation_field(128, SEED)
    hyd = wg.compute_hydrology(elev, SEED)
    p = hyd["parent"]
    for _ in range(20):          # 2^20 >> any path length on a 128² grid
        p = p[p]
    assert bool(hyd["sea"].reshape(-1)[p].all())


def test_accumulation_conserves_cells():
    elev = wg.elevation_field(96, SEED)
    hyd = wg.compute_hydrology(elev, SEED)
    acc = hyd["accum"]
    n = acc.size
    # every cell contributes 1 to itself, so accumulation is at least 1
    assert float(acc.min()) >= 1.0
    # and the sinks together must have collected every cell exactly once
    p = hyd["parent"]
    for _ in range(20):
        p = p[p]
    sink_total = float(acc.reshape(-1)[np.unique(p)].sum())
    assert sink_total == pytest.approx(n, rel=1e-6)


# ---------------------------------------------------------------------------
# Render smoke: every layer renders at a couple of times with sane output,
# through the same state() -> colorize() path every frontend uses.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    return wc.build_world(seed=7, size=96, civ_count=2)


def test_every_layer_renders(world):
    thr = wc.default_river_threshold(world.size)
    for t in (0.0, 10.0, 500.0):
        st = wc.state(world, t, sea_level=0.42, river_thr=thr,
                      season_amp=0.18, tide_amp=0.012, day_night=0.65)
        for key, layer in wc.LAYERS:
            img = wc.colorize(st, key)
            assert img.shape == (96, 96, 3), (key, t)
            assert img.dtype == np.uint8, (key, t)


def test_state_has_no_nans(world):
    st = wc.state(world, 42.0, 0.42, wc.default_river_threshold(96),
                  0.18, 0.012, 0.65)
    for k, v in st.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
            assert np.isfinite(v).all(), k


def test_zoomed_view_renders(world):
    v = world.view(0.31, 0.62, 12.0)
    img = wc.render(v, "composite", 5.0, 0.42,
                    wc.default_river_threshold(96), 0.18, 0.012, 0.65)
    assert img.shape == (96, 96, 3) and img.dtype == np.uint8


def test_ecosim_steps_and_stays_bounded(world):
    eco = world.eco
    eco.reset()
    eco.step(400.0, sea_level=0.55, season_off=0.3)   # flood + hot summers
    for k in ("veg", "fauna", "civ", "fert", "scorch"):
        v = getattr(eco, k)
        assert np.isfinite(v).all(), k
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0, k
