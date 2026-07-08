#!/usr/bin/env python3
"""Rebuild index.json from Supabase files with subject metadata."""
import json, re, os, shutil, datetime as dt
import supabase_store, texgen, world_core as wc
import export_texture_gallery as etg

SEED, SIZE = 42, 128

def main() -> int:
    storage = supabase_store.SupabaseStorage()
    base_url = storage.public_url("").rstrip("/")
    print(f"Listing Supabase '{storage.bucket}' img/ ...")
    items = storage.list("img/")
    pattern = re.compile(r"^([0-9a-f]{16})_v(\d+)(?:_\d+)?\.png$", re.I)
    by_hash = {}
    for item in items:
        fname = item.get("name", "").removeprefix("img/").lstrip("/")
        m = pattern.match(fname)
        if m:
            by_hash.setdefault(m.group(1), []).append(fname)
    for h in by_hash:
        by_hash[h].sort()
    print(f"Found {len(by_hash)} distinct hashes")

    print(f"Running world sim to get subject metadata (no image generation)...")
    store_path = "_meta_store_tmp"
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    svc = texgen.TextureService(store_path, backend=texgen.PlaceholderBackend())
    world = wc.build_world(SEED, SIZE, 3)
    sea, thr = 0.42, wc.default_river_threshold(SIZE)
    for label, zoom, t in etg.VIEWS:
        chunk = world.stream_view(0.62, 0.44, zoom, 48)
        st = wc.state(chunk, t, sea, thr, 0.18, 0.012, 0.65)
        df = texgen.derive(chunk, st)
        for desc in df.legend.values():
            p, n = texgen.build_prompt(desc)
            svc.store.insert_pending(desc, p, n, svc.variations, svc.px_for(desc))
        print(f"  [{label}] {len(df.legend)} keys", flush=True)
    with svc.store.lock:
        rows = svc.store.db.execute("SELECT key_hash, subject, lod, tags FROM assets").fetchall()
    svc.store.db.close()
    shutil.rmtree(store_path)
    meta = {h: {"subject": s, "lod": l, "tags": json.loads(t or "{}")} for h, s, l, t in rows}
    print(f"Matched {sum(1 for h in by_hash if h in meta)}/{len(by_hash)} hashes to subjects")

    assets = []
    for h, fnames in sorted(by_hash.items()):
        entry = {"hash": h, "files": fnames}
        entry.update(meta.get(h, {}))
        assets.append(entry)

    index = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z"),
        "count": len(assets),
        "base_url": base_url,
        "assets": assets,
    }
    url = storage.upload_json("index.json", index)
    print(f"Done: {sum(1 for a in assets if 'subject' in a)}/{len(assets)} assets have subject -> {url}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
