"""Tests for the tag-driven texture pipeline (texgen.py).

Runnable two ways:  pytest test_texgen.py   or   python3 test_texgen.py
Uses a real (small) world so derive() is exercised against genuine state.
"""

import os
import shutil
import tempfile

import numpy as np

import texgen as tg
import world_core as wc

SEED, SIZE = 42, 96
_world = None


def world():
    global _world
    if _world is None:
        _world = wc.build_world(SEED, SIZE, 3)
    return _world


def chunk_state(zoom=200.0, t=20.5, cx=0.62, cy=0.44, tiles=32):
    ws = world().stream_view(cx, cy, zoom, tiles)
    st = wc.state(ws, t, 0.42, wc.default_river_threshold(SIZE), 0.18, 0.012, 0.65)
    return ws, st


def fresh_service(**kw):
    root = tempfile.mkdtemp(prefix="texstore_")
    return tg.TextureService(root, **kw), root


# --- schema & keys -----------------------------------------------------------

def test_canonical_key_is_deterministic_and_ordered():
    d1 = tg.descriptor("tree.oak", season="winter", tod="night", lod="group3",
                       temp="cold", growth="mature", cond="pristine",
                       density="dense", wet="wet")
    d2 = tg.descriptor("tree.oak", wet="wet", density="dense", cond="pristine",
                       growth="mature", temp="cold", lod="group3",
                       tod="night", season="winter")
    assert d1.key == d2.key
    assert d1.key_hash == d2.key_hash
    # irrelevant axis (wet, for a tree) must be dropped from the key
    assert "wet=" not in d1.key
    assert d1.key.startswith("tree.oak|lod=group3|")


def test_descriptor_rejects_unknown_value():
    try:
        tg.descriptor("tree.oak", season="monsoon")
    except ValueError:
        return
    raise AssertionError("bad tag value must raise")


def test_density_dropped_at_and_below_single():
    d = tg.descriptor("tree.oak", lod="single", density="dense")
    assert "density=" not in d.key
    d = tg.descriptor("tree.oak", lod="obj4x4", density="dense")
    assert "density=" not in d.key
    assert d.footprint == 4
    d = tg.descriptor("tree.oak", lod="group3", density="dense")
    assert "density=dense" in d.key
    assert d.footprint == 1


def test_lod_ladder():
    assert tg.lod_for_tile_world(tg.TILE0_WORLD) == 0
    assert tg.lod_for_tile_world(tg.TILE0_WORLD * 3) == 1
    assert tg.lod_for_tile_world(tg.TILE0_WORLD * 9) == 2
    assert tg.lod_for_tile_world(tg.TILE0_WORLD / 2) == -1
    assert tg.lod_for_tile_world(tg.TILE0_WORLD / 8) == -3
    assert tg.lod_for_tile_world(1.0) == tg.LOD_MAX          # clamped
    assert tg.lod_for_tile_world(1e-9) == tg.LOD_MIN


def test_pack_decode_roundtrip():
    ws, st = chunk_state()
    df = tg.derive(ws, st)
    for code, desc in df.legend.items():
        again = tg._decode(code)
        assert again.key == desc.key
        # key must parse back to the same descriptor
        assert tg._desc_from_key(desc.key) == desc


# --- derive ----------------------------------------------------------------

def test_derive_collapses_chunk_to_few_keys():
    ws, st = chunk_state()
    df = tg.derive(ws, st)
    assert df.ground.shape == (ws.size, ws.size)
    n_tiles = ws.size * ws.size
    assert 0 < len(df.legend) < n_tiles / 4     # massive dedup is the point
    # every ground code decodes to a ground subject
    for code in np.unique(df.ground):
        assert df.legend[int(code)].subject.startswith("ground.")


def test_derive_is_deterministic():
    ws, st = chunk_state()
    a, b = tg.derive(ws, st), tg.derive(ws, st)
    assert np.array_equal(a.ground, b.ground)
    assert [(p.i, p.j, p.code, p.variation) for p in a.props] == \
           [(p.i, p.j, p.code, p.variation) for p in b.props]


def test_season_and_tod_change_keys():
    ws, st_summer_noon = chunk_state(t=20.5)      # sun_x=.5 noon
    _, st_winter_night = chunk_state(t=68.05)     # sun_x=.05 night
    k1 = {d.key for d in tg.derive(ws, st_summer_noon).legend.values()}
    k2 = {d.key for d in tg.derive(ws, st_winter_night).legend.values()}
    assert k1 != k2
    assert any("tod=day" in k for k in k1)
    assert any("tod=night" in k for k in k2)


def test_negative_lod_props_are_multi_tile_instances():
    # zoom until a tile is far smaller than TILE0 -> lod < 0
    ws, st = chunk_state(zoom=20000.0, tiles=32)
    df = tg.derive(ws, st)
    assert df.lod < 0
    fp = 2 ** (-df.lod)
    for inst in df.props:
        assert inst.footprint == fp
        assert df.legend[inst.code].footprint == fp
    # anchors must land on the lod-0 lattice: no two instances overlap
    seen = set()
    for inst in df.props:
        cell = (inst.i // fp, inst.j // fp)
        assert cell not in seen
        seen.add(cell)


def test_variation_grid_is_stable_across_pans():
    ws1 = world().stream_view(0.62, 0.44, 300.0, 32)
    v1 = tg.variation_grid(ws1, 3)
    # pan by exactly a few tiles: overlapping world cells keep their variation
    tw = ws1.span / ws1.size
    ws2 = world().stream_view(0.62 + 4 * tw, 0.44, 300.0, 32)
    v2 = tg.variation_grid(ws2, 3)
    assert np.array_equal(v1[:, 4:], v2[:, :-4])


# --- prompts ------------------------------------------------------------------

def test_prompt_reflects_tags():
    d = tg.descriptor("tree.oak", lod="group3", season="winter", tod="dusk",
                      temp="cold", growth="mature", cond="pristine",
                      density="dense")
    prompt, negative = tg.build_prompt(d)
    assert "cluster of three" in prompt and "oak" in prompt
    assert "snow" in prompt and "golden hour" in prompt
    assert "transparent background" in prompt
    assert negative == tg.NEGATIVE
    g = tg.descriptor("ground.desert", lod="single", season="summer",
                      tod="day", temp="hot", wet="arid", cond="pristine")
    gp, _ = tg.build_prompt(g)
    assert "tileable" in gp and "desert" in gp


# --- runpod workflow/backend --------------------------------------------------

def test_runpod_payload_shape_and_required_nodes():
    wf = tg.build_runpod_comfyui_workflow(
        prompt="p", negative_prompt="n", seed=7, width=512, height=512
    )
    payload = tg.build_runpod_runsync_payload(wf)
    assert "input" in payload and "workflow" in payload["input"]
    classes = {node["class_type"] for node in wf.values()}
    for need in ("CheckpointLoaderSimple", "CLIPTextEncode", "LoraLoader",
                 "KSampler", "VAEDecode", "SaveImage"):
        assert need in classes


def test_runpod_backend_build_payload_includes_prompt_prefix_and_lora_path():
    b = tg.RunPodComfyUIBackend(
        endpoint_id="ep",
        api_key="key",
        prompt_prefix="isometric stylized setting",
        checkpoint_name="sd_xl_base_1.0.safetensors",
        lora_name="foo.safetensors",
        lora_path="/workspace/ComfyUI/models/loras/",
    )
    job = tg.GenJob("k", "tree sprite", "bad", 11, 640, 2)
    payload = b.build_payload(job)
    wf = payload["input"]["workflow"]
    assert wf["1"]["inputs"]["ckpt_name"] == "sd_xl_base_1.0.safetensors"
    assert wf["2"]["inputs"]["lora_name"] == "/workspace/ComfyUI/models/loras/foo.safetensors"
    assert "isometric stylized setting" in wf["3"]["inputs"]["text"]


def test_runpod_backend_dry_run_works_without_credentials():
    b = tg.RunPodComfyUIBackend(dry_run=True, endpoint_id=None, api_key=None)
    d = tg.descriptor("tree.oak", lod="single", season="summer", tod="day",
                      temp="mild", growth="mature", cond="pristine")
    job = tg.GenJob(d.key, tg.build_prompt(d)[0], tg.NEGATIVE, 123, 64, 2)
    out = b.generate(job)
    assert len(out) == 2 and all(isinstance(x, bytes) for x in out)


def test_civitai_lora_download_support_dry_run_path():
    assert tg.civitai_lora_download_url(
        "https://civitai.com/models/118775/stylized-setting-isometric-sdxl-and-sd15"
    ) == "https://civitai.com/api/download/models/118775"
    root = tempfile.mkdtemp(prefix="lora_")
    try:
        p = tg.download_civitai_lora(dest_dir=root, dry_run=True)
        assert p.endswith("stylized-setting-isometric-sdxl-and-sd15.safetensors")
        assert p.startswith(root)
    finally:
        shutil.rmtree(root)


# --- store & service lifecycle --------------------------------------------------

def test_resolve_serves_placeholder_then_ready():
    svc, root = fresh_service()
    try:
        d = tg.descriptor("tree.pine", lod="single", season="autumn",
                          tod="day", temp="cold", growth="young",
                          cond="pristine")
        r1 = svc.resolve(d)
        assert r1.status == "pending" and r1.served == "placeholder"
        assert len(r1.paths) == svc.variations
        assert all(os.path.exists(p) for p in r1.paths)
        assert svc.pending_count() == 1
        assert svc.pump() == 1
        r2 = svc.resolve(d)
        assert r2.status == "ready" and r2.served == "exact"
        assert all("assets" in p for p in r2.paths)
    finally:
        shutil.rmtree(root)


def test_placeholder_is_deterministic():
    d = tg.descriptor("tree.oak", lod="single", season="summer", tod="day",
                      temp="mild", growth="mature", cond="pristine")
    svc1, r1 = fresh_service()
    svc2, r2 = fresh_service()
    try:
        b1 = open(svc1.resolve(d, enqueue=False).paths[0], "rb").read()
        b2 = open(svc2.resolve(d, enqueue=False).paths[0], "rb").read()
        assert b1 == b2
    finally:
        shutil.rmtree(r1); shutil.rmtree(r2)


def test_fallback_prefers_nearest_ready_neighbor():
    svc, root = fresh_service()
    try:
        base = dict(lod="single", tod="day", temp="mild", growth="mature",
                    cond="pristine")
        summer = tg.descriptor("tree.oak", season="summer", **base)
        winter = tg.descriptor("tree.oak", season="winter", **base)
        far = tg.descriptor("tree.oak", season="winter",
                            lod="single", tod="night", temp="freezing",
                            growth="bare", cond="scorched")
        svc.resolve(summer); svc.pump()             # summer art is now ready
        r = svc.resolve(winter)
        assert r.served == f"fallback:{summer.key}"  # near miss -> substitute
        r = svc.resolve(far)
        assert r.served == "placeholder"             # too different -> placeholder
        svc.pump()
        assert svc.resolve(winter).served == "exact"
    finally:
        shutil.rmtree(root)


def test_failed_backend_records_error_and_retries():
    class Boom(tg.Backend):
        name = "boom"
        def __init__(self): self.calls = 0
        def generate(self, job):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("gpu on fire")
            return tg.PlaceholderBackend().generate(job)

    svc, root = fresh_service(backend=Boom())
    try:
        d = tg.descriptor("rock", lod="single", season="summer", tod="day",
                          temp="mild", density="sparse")
        svc.resolve(d); svc.pump()
        row = svc.store.get(d.key)
        assert row["status"] == "failed" and "gpu on fire" in row["error"]
        assert svc.resolve(d).served == "placeholder"   # still degrades fine
        assert svc.retry_failed() >= 1
        svc.pump()
        assert svc.store.get(d.key)["status"] == "ready"
    finally:
        shutil.rmtree(root)


def test_resolve_field_and_prewarm():
    svc, root = fresh_service()
    try:
        ws, st = chunk_state()
        df = tg.derive(ws, st)
        res = svc.resolve_field(df)
        assert set(res) == set(df.legend)
        n_exact = svc.pending_count()
        assert n_exact == len(df.legend)             # every key queued once
        queued = svc.prewarm_neighbors(df)
        assert queued > 0                            # tomorrow's art is queued
        svc.pump()
        res2 = svc.resolve_field(df)
        assert all(r.served == "exact" for r in res2.values())
        st_stats = svc.stats()
        assert st_stats["by_status"].get("ready", 0) >= len(df.legend)
    finally:
        shutil.rmtree(root)


def test_eviction_frees_bytes_and_regenerates():
    svc, root = fresh_service()
    try:
        d = tg.descriptor("cactus", lod="single", season="summer", tod="day",
                          temp="hot", density="sparse")
        svc.resolve(d); svc.pump()
        assert svc.store.stats()["ready_bytes"] > 0
        assert svc.store.evict_lru(0) == 1
        assert svc.store.stats()["ready_bytes"] == 0
        r = svc.resolve(d)                            # miss again -> requeued
        assert r.status == "pending"
        svc.pump()
        assert svc.resolve(d).served == "exact"       # regenerated identically
    finally:
        shutil.rmtree(root)


def test_variation_files_use_distinct_seeds():
    svc, root = fresh_service()
    try:
        d = tg.descriptor("house", lod="group3", season="summer", tod="day",
                          density="dense", cond="pristine")
        svc.resolve(d); svc.pump()
        blobs = [open(p, "rb").read() for p in svc.resolve(d).paths]
        assert len(set(blobs)) == len(blobs)          # variations differ
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    import sys, traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
