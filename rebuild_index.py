#!/usr/bin/env python3
"""Rebuild index.json directly from files already in Supabase.

Lists img/ in the bucket, groups files by key_hash, uploads a fresh
index.json. No world simulation, no generation, runs in seconds.

Usage:
    python rebuild_index.py
Requires: SUPABASE_URL and SUPABASE_SERVICE_KEY env vars.
"""
import json, os, re, datetime as dt
import supabase_store

def main() -> int:
    storage = supabase_store.SupabaseStorage()
    base_url = storage.public_url("").rstrip("/")

    print(f"Listing Supabase bucket '{storage.bucket}' img/ …")
    items = storage.list("img/")
    print(f"Found {len(items)} raw items")
    if items:
        print(f"  first item keys: {list(items[0].keys())}")
        print(f"  first item name: {items[0].get('name','?')}")

    # Group by hash: files are named <hash>_v<N>.png (or <hash>_v<N>)
    by_hash: dict[str, list[str]] = {}
    pattern = re.compile(r'^([0-9a-f]{16})_v(\d+)(?:_\d+)?(\.png)?$', re.I)
    for item in items:
        fname = item.get("name", "")
        # Supabase may return name with or without the prefix
        fname = fname.removeprefix("img/").lstrip("/")
        m = pattern.match(fname)
        if m:
            h = m.group(1)
            by_hash.setdefault(h, []).append(fname)

    # Sort variations within each hash
    for h in by_hash:
        by_hash[h].sort()

    assets = [
        {"hash": h, "files": fnames}
        for h, fnames in sorted(by_hash.items())
    ]

    print(f"Grouped into {len(assets)} assets")
    if not assets:
        print("\nNo files matched pattern <16hexchars>_v<N>.png")
        print("Raw names sample:", [i.get("name","") for i in items[:10]])
        return 1

    index = {
        "generated_at": dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "count": len(assets),
        "base_url": base_url,
        "assets": assets,
    }

    url = storage.upload_json("index.json", index)
    print(f"Uploaded index.json with {len(assets)} assets → {url}")

    # Also write web/textures/config.json so the local gallery knows the base_url
    config_path = os.path.join(os.path.dirname(__file__), "web", "textures", "config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({"base_url": base_url}, f, indent=1)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

