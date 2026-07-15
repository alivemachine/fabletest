"""Tests for scale_core — the multi-scale tile core.

Run: python3 test_scale_core.py   (or pytest test_scale_core.py)

What must hold for Google-Earth-style navigation over a tile world:
 1. determinism — same seed & window => bit-identical tiles, every visit
 2. pan coherence — overlapping windows agree tile-for-tile (snapping)
 3. zoom coherence — the coarse fields at depth match the planet above
 4. entity stability — an individual keeps its world position across zoom,
    and its footprint grows 1 tile -> 2x2 -> 3x3 as tiles shrink
 5. aggregation — below ~half-tile footprint a kind leaves the tile grid
    and shows up in the tile's `contains` statistics instead
 6. the whole ladder — every level from planet to grain renders, with
    sane per-level content and constant tile count
"""

import json
import math

import numpy as np

import scale_core as sc

SEED = 42


def setup_world():
    sc.reset(SEED)
    return sc.find_player(SEED)


def test_determinism():
    px, py = setup_world()
    sc.view(SEED, px, py, 9.54, 64, 40)
    a = sc.colorize("composite").copy()
    kid_a = sc._state["kid"].copy()
    sc.view(SEED, px + 1e6, py, 9.54, 64, 40)      # go away
    sc.view(SEED, px, py, 9.54, 64, 40)            # come back
    b = sc.colorize("composite")
    assert np.array_equal(a, b), "same window must be bit-identical"
    assert np.array_equal(kid_a, sc._state["kid"])
    print("ok determinism — same window, same world, every visit")


def test_pan_coherence():
    px, py = setup_world()
    tm = 76.3
    cx, cy, tm = sc.view(SEED, px, py, tm, 64, 40)
    a = sc._state["bio"].copy()
    ka = sc._state["kid"].copy()
    # pan exactly 10 tiles right: columns 10.. of A == columns ..-10 of B
    sc.view(SEED, cx + 10 * tm, cy, tm, 64, 40)
    b = sc._state["bio"]
    kb = sc._state["kid"]
    assert np.array_equal(a[:, 10:], b[:, :-10]), "pan must slide, not resample"
    assert np.array_equal(ka[:, 10:], kb[:, :-10]), "entities must slide too"
    print("ok pan coherence — panning translates the same tiles")


def test_zoom_coherence():
    """Diving in adds detail; it must not move the planet. The elevation
    seen at building scale stays within the fine-octave budget of the
    elevation seen from orbit at the same world point."""
    px, py = setup_world()
    x = np.float64(px)
    y = np.float64(py)
    coarse = float(sc.elevation01(SEED, x, y, 5000.0))
    fine = float(sc.elevation01(SEED, x, y, 0.01))
    # remaining octaves below 5000 m can add at most sum p^k of the tail
    k0 = math.ceil(math.log2(sc._OCT0_WL / 5000.0))
    tail = (0.58 ** k0 / (1 - 0.58)) / sc._FIXED_TOTAL
    assert abs(fine - coarse) <= tail + 1e-9, (fine, coarse, tail)
    print(f"ok zoom coherence — orbit {coarse:.4f} vs ant-view {fine:.4f} "
          f"(tail budget {tail:.4f})")


def test_constant_tile_count():
    px, py = setup_world()
    for tm in (sc.TILE0_M, 4882.8, 9.54, 0.0186):
        sc.view(SEED, px, py, tm, 64, 40)
        nx, ny, buf = sc.render_rgba("composite")
        assert (nx, ny) == (64, 40)
        assert len(buf) == 64 * 40 * 4
    print("ok constant tile count — 64x40 tiles at every scale")


def test_player_footprint_grows():
    px, py = setup_world()
    sizes = {}
    for tm in (1.19, 0.6, 0.3, 0.2):
        sc.view(SEED, px, py, tm, 64, 40)
        n = int((sc._state["kid"] == sc.PLAYER_KID).sum())
        sizes[tm] = n
    assert sizes[1.19] <= 1, "far: at most a speck"
    assert 1 <= sizes[0.6] <= 4, f"~1 tile at 0.6 m tiles, got {sizes[0.6]}"
    assert sizes[0.3] > sizes[0.6], "closer: more tiles"
    assert sizes[0.2] >= 7, f"~3x3 at 0.2 m tiles, got {sizes[0.2]}"
    # zoom way out: the player must vanish from the grid entirely
    sc.view(SEED, px, py, 4882.8, 64, 40)
    assert int((sc._state["kid"] == sc.PLAYER_KID).sum()) == 0
    print(f"ok player footprint — {sizes} tiles, invisible from orbit")


def test_entity_stable_across_zoom():
    """Find a tree at building scale, zoom onto it 8x: same id, same spot."""
    px, py = setup_world()
    cx, cy, tm = sc.view(SEED, px, py, 9.54, 64, 40)
    tree_kid = next(k["kid"] for k in sc.KINDS if k["name"] == "tree")
    js, is_ = np.nonzero(sc._state["kid"] == tree_kid)
    assert len(is_), "expected trees near spawn at building scale"
    j, i = int(js[0]), int(is_[0])
    eid = int(sc._state["eid"][j, i])
    tx = float(sc._state["X"][j, i])
    ty = float(sc._state["Y"][j, i])
    sc.view(SEED, tx, ty, tm / 8.0, 64, 40)
    kid2 = sc._state["kid"]
    eid2 = sc._state["eid"]
    hit = (kid2 == tree_kid) & (eid2 == eid)
    assert hit.any(), "the SAME tree (same id) must be there when zoomed in"
    assert hit.sum() > 4, "and it must occupy more tiles than before"
    print(f"ok entity stability — tree #{eid} kept its place, "
          f"{int(hit.sum())} tiles at 8x zoom")


def test_aggregation_vs_expansion():
    """Buildings: statistics at district scale, individuals at building scale."""
    px, py = setup_world()
    # find a town center so there is something urban to look at
    urban, bk, bh, bd = sc.urban_at(SEED, np.float64(px), np.float64(py))
    b_kid = next(k["kid"] for k in sc.KINDS if k["name"] == "building")
    sc.view(SEED, px, py, 610.35, 64, 40)
    assert int((sc._state["kid"] == b_kid).sum()) == 0, \
        "13 m buildings must NOT own 610 m tiles"
    recs = [sc.describe(i, j) for j in range(0, 40, 8) for i in range(0, 64, 8)]
    agg = [r for r in recs if any(c["kind"] == "building" for c in r["contains"])]
    assert agg, "some district tile should aggregate buildings"
    print(f"ok aggregation — buildings are statistics at L3 "
          f"({len(agg)}/{len(recs)} sampled tiles), entities at L5")


def test_describe_schema():
    px, py = setup_world()
    sc.view(SEED, px, py, 1.19, 64, 40)
    rec = json.loads(sc.describe_json(32, 20))
    for key in ("tile", "world_m", "tile_size", "level", "biome",
                "elevation", "temperature", "moisture", "contains"):
        assert key in rec, key
    assert rec["entity"]["name"] == "player"
    assert rec["entity"]["interactable"] is True
    print("ok describe schema —", rec["level"], "/", rec["biome"],
          "/ entity:", rec["entity"]["name"])


def test_full_ladder():
    px, py = setup_world()
    for k, (name, _what) in enumerate(sc.LEVELS):
        tm = sc.level_tile_m(k)
        if tm < sc.MIN_TILE_M:
            tm = sc.MIN_TILE_M
        sc.view(SEED, px, py, tm, 64, 40)
        nx, ny, buf = sc.render_rgba("composite")
        assert len(buf) == nx * ny * 4
        lname, lidx = sc.level_name(sc._state["tile_m"])
        assert lidx == min(k, len(sc.LEVELS) - 1), (name, lname)
    print("ok full ladder — all", len(sc.LEVELS), "levels render")


def test_layers():
    px, py = setup_world()
    sc.view(SEED, px, py, 4882.8, 64, 40)
    for layer in sc.LAYER_NAMES:
        nx, ny, buf = sc.render_rgba(layer)
        assert len(buf) == nx * ny * 4, layer
    print("ok layers —", ", ".join(sc.LAYER_NAMES))


if __name__ == "__main__":
    test_determinism()
    test_pan_coherence()
    test_zoom_coherence()
    test_constant_tile_count()
    test_player_footprint_grows()
    test_entity_stable_across_zoom()
    test_aggregation_vs_expansion()
    test_describe_schema()
    test_full_ladder()
    test_layers()
    print("\nall scale_core tests passed")
