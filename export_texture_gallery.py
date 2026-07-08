#!/usr/bin/env python3
"""Export the texture store to a static gallery the web page can browse.

Populates a texture store with a representative spread of world views
(4 seasons x day/night, far and near zoom), pumps the generation queue, then
copies every ready asset into web/textures/ with an index.json manifest:

    web/textures/index.json
    web/textures/img/<key_hash>_v<i>.png

The web page's "Generated textures" section reads that manifest. With no
backend configured this publishes the deterministic procedural placeholders;
run with a real backend (e.g. RUNPOD_ENDPOINT_ID/RUNPOD_API_KEY +
--backend runpod-comfyui) to publish real art — same keys, same filenames.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import shutil
from pathlib import Path

import world_core as wc
import texgen

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "web" / "textures"

# (label, zoom, sim day) — noon is N.5; a year is 96 days, so seasons sit a
# quarter-year apart. Two night views exercise the tod axis.
VIEWS = [
    ("far spring noon", 4.0, 8.5),
    ("far summer noon", 4.0, 32.5),
    ("far autumn noon", 4.0, 56.5),
    ("far winter noon", 4.0, 80.5),
    ("far summer night", 4.0, 32.05),
    ("far winter night", 4.0, 80.05),
    ("mid summer noon", 24.0, 32.5),
    ("mid winter dusk", 24.0, 80.8),
    ("close summer noon", 100.0, 32.5),
    ("near summer noon", 400.0, 32.5),
    ("near winter night", 400.0, 80.05),
]


def make_backend(name: str) -> texgen.Backend | None:
    if name == "placeholder":
        return None  # service default
    if name == "runpod-comfyui":
        return texgen.RunPodComfyUIBackend()
    raise SystemExit(f"unknown backend {name!r}")


def reset_stranded(svc: texgen.TextureService) -> None:
    """A killed run leaves keys stuck in 'generating'; nothing is running
    when this script starts, so put them back to 'pending' for re-queueing."""
    with svc.store.lock:
        n = svc.store.db.execute(
            "UPDATE assets SET status='pending' WHERE status='generating'"
        ).rowcount
        svc.store.db.commit()
    if n:
        print(f"reset {n} stranded 'generating' key(s) to pending")


def populate(svc: texgen.TextureService, seed: int, size: int,
             concurrency: int = 1) -> None:
    world = wc.build_world(seed, size, 3)
    sea, thr = 0.42, wc.default_river_threshold(size)
    for label, zoom, t in VIEWS:
        chunk = world.stream_view(0.62, 0.44, zoom, 48)
        st = wc.state(chunk, t, sea, thr, 0.18, 0.012, 0.65)
        df = texgen.derive(chunk, st)
        svc.resolve_field(df)
        print(f"[{label}] -> {len(df.legend)} distinct keys")

    total = svc._queue.qsize()
    done = itertools.count(1)

    def progress(key: str) -> None:
        print(f"[{next(done)}/{total}] {key}", flush=True)

    ran = svc.pump(concurrency=concurrency, progress=progress)
    print(f"generated {ran} assets via '{svc.backend.name}' backend "
          f"(concurrency={concurrency})")


def collect_entries(svc: texgen.TextureService):
    """All ready assets with their variation files: [(entry_dict, [paths])]."""
    out = []
    with svc.store.lock:
        rows = svc.store.db.execute(
            "SELECT key, key_hash, subject, lod, tags, status, backend, px"
            " FROM assets WHERE status='ready' ORDER BY subject, key"
        ).fetchall()
        for key, key_hash, subject, lod, tags, status, backend, px in rows:
            paths = [
                p
                for (p,) in svc.store.db.execute(
                    "SELECT path FROM variations WHERE key=? ORDER BY idx", (key,)
                )
            ]
            srcs, files = [], []
            for i, p in enumerate(paths):
                if isinstance(p, str) and p.startswith("http"):
                    # RunPod already uploaded to cloud storage — use URL directly
                    fname = p.rsplit("/", 1)[-1]  # last path segment
                    srcs.append(p)
                    files.append(fname)
                else:
                    src = Path(p)
                    if not src.is_absolute():
                        src = ROOT / src
                    if not src.exists():
                        continue
                    srcs.append(src)
                    files.append(f"{key_hash}_v{i}.png")
            if files:
                out.append((
                    {
                        "key": key,
                        "hash": key_hash,
                        "subject": subject,
                        "lod": lod,
                        "tags": json.loads(tags) if tags else {},
                        "backend": backend,
                        "px": px,
                        "files": files,
                    },
                    srcs,
                ))
    return out


def build_index(entries: list, base_url: str | None = None) -> dict:
    index = {
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "count": len(entries),
        "assets": entries,
    }
    if base_url:
        index["base_url"] = base_url
    return index


def export(svc: texgen.TextureService, out_dir: Path) -> int:
    """Local export: copy ready art into web/textures/ for file: / Pages use."""
    img_dir = out_dir / "img"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for entry, srcs in collect_entries(svc):
        for name, src in zip(entry["files"], srcs):
            if not (isinstance(src, str) and src.startswith("http")):
                shutil.copyfile(src, img_dir / name)
        entries.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(
        json.dumps(build_index(entries), indent=1) + "\n")
    return len(entries)


def publish_supabase(svc: texgen.TextureService, out_dir: Path) -> int:
    """Publish ready art to the Supabase bucket: img/<hash>_v<i>.png +
    index.json, and point the web page at it via web/textures/config.json."""
    import supabase_store

    store = supabase_store.SupabaseStorage()
    store.ensure_bucket()
    base_url = store.public_url("")[:-1]  # bucket root, no trailing slash

    entries, uploaded = [], 0
    todo = collect_entries(svc)
    for n, (entry, srcs) in enumerate(todo, 1):
        for name, src in zip(entry["files"], srcs):
            if isinstance(src, str) and src.startswith("http"):
                pass  # RunPod already uploaded to the correct path
            else:
                store.upload_png(f"img/{name}", Path(src).read_bytes())
            uploaded += 1
        entries.append(entry)
        print(f"[upload {n}/{len(todo)}] {entry['key']}", flush=True)

    store.upload_json("index.json", build_index(entries, base_url))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"base_url": base_url}, indent=1) + "\n")
    print(f"published {uploaded} files + index.json -> {base_url}/")
    return len(entries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--store", default="texture_store")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--backend", default="placeholder", choices=["placeholder", "runpod-comfyui"]
    )
    ap.add_argument(
        "--concurrency", type=int, default=1,
        help="generation jobs to keep in flight at once (lets several "
             "RunPod workers run in parallel)",
    )
    ap.add_argument(
        "--publish", choices=["local", "supabase"], default="local",
        help="local: copy files into web/textures/; supabase: upload the "
             "database to the Supabase bucket and write web/textures/config.json",
    )
    args = ap.parse_args()

    svc = texgen.TextureService(args.store, backend=make_backend(args.backend))
    reset_stranded(svc)
    populate(svc, args.seed, args.size, concurrency=args.concurrency)
    if args.publish == "supabase":
        n = publish_supabase(svc, args.out.resolve())
        print(f"published {n} assets to Supabase")
    else:
        n = export(svc, args.out.resolve())
        print(f"exported {n} assets -> {args.out}/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
