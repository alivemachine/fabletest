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


def populate(svc: texgen.TextureService, seed: int, size: int) -> None:
    world = wc.build_world(seed, size, 3)
    sea, thr = 0.42, wc.default_river_threshold(size)
    for label, zoom, t in VIEWS:
        chunk = world.stream_view(0.62, 0.44, zoom, 48)
        st = wc.state(chunk, t, sea, thr, 0.18, 0.012, 0.65)
        df = texgen.derive(chunk, st)
        svc.resolve_field(df)
        print(f"[{label}] -> {len(df.legend)} distinct keys")
    ran = svc.pump()
    print(f"generated {ran} assets via '{svc.backend.name}' backend")


def export(svc: texgen.TextureService, out_dir: Path) -> int:
    img_dir = out_dir / "img"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    entries = []
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
            files = []
            for i, p in enumerate(paths):
                src = Path(p)
                if not src.is_absolute():
                    src = ROOT / src
                if not src.exists():
                    continue
                name = f"{key_hash}_v{i}.png"
                shutil.copyfile(src, img_dir / name)
                files.append(name)
            if files:
                entries.append(
                    {
                        "key": key,
                        "hash": key_hash,
                        "subject": subject,
                        "lod": lod,
                        "tags": json.loads(tags) if tags else {},
                        "backend": backend,
                        "px": px,
                        "files": files,
                    }
                )

    index = {
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "count": len(entries),
        "assets": entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
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
    args = ap.parse_args()

    svc = texgen.TextureService(args.store, backend=make_backend(args.backend))
    populate(svc, args.seed, args.size)
    n = export(svc, args.out.resolve())
    print(f"exported {n} assets -> {args.out}/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
