#!/usr/bin/env python3
"""One-shot smoke test for the RunPod texture backend.

Checks endpoint health, submits a single small generation job, and saves the
returned PNG(s) so you can eyeball them. Run this before a full gallery export
to confirm credentials, input format, and cold-start behaviour:

    export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
    python runpod_smoke.py                     # tries comfyui, then prompt format
    python runpod_smoke.py --format prompt     # force one format
    python runpod_smoke.py --px 512            # smaller/faster

Exit code 0 = at least one format produced images (the winner is printed —
set RUNPOD_INPUT_FORMAT to it for export_texture_gallery.py runs).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import texgen

OUT_DIR = Path(__file__).resolve().parent / "runpod_smoke_out"

_DESC = texgen.descriptor("tree.oak", lod="single", season="summer", tod="day",
                          temp="mild", growth="mature", cond="pristine")
SMOKE_JOB = texgen.GenJob(
    key=_DESC.key,
    prompt=texgen.build_prompt(_DESC)[0],
    negative=texgen.NEGATIVE,
    seed=42,
    px=1024,
    n=1,
)


def try_format(fmt: str, px: int) -> list | None:
    print(f"\n=== input format: {fmt} ===")
    backend = texgen.RunPodComfyUIBackend(input_format=fmt)
    job = texgen.GenJob(SMOKE_JOB.key, SMOKE_JOB.prompt, SMOKE_JOB.negative,
                        SMOKE_JOB.seed, px, SMOKE_JOB.n)
    print("payload:", json.dumps(backend.build_payload(job))[:200], "...")
    t0 = time.monotonic()
    try:
        images = backend.generate(job)
    except Exception as e:
        print(f"FAILED after {time.monotonic() - t0:.1f}s: {e}")
        return None
    print(f"OK: {len(images)} image(s) in {time.monotonic() - t0:.1f}s")
    return images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", choices=["comfyui", "prompt"],
                    help="force one input format instead of trying both")
    ap.add_argument("--px", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    probe = texgen.RunPodComfyUIBackend()
    print(f"endpoint: {probe.base_url}  (mode={probe.mode})")
    try:
        health = probe.health()
        print("health:", json.dumps(health))
        workers = health.get("workers", {})
        if not workers.get("ready") and not workers.get("running"):
            print("note: no warm workers — expect a cold start of several minutes")
    except Exception as e:
        print(f"health check failed ({e}) — trying generation anyway")

    formats = [args.format] if args.format else ["comfyui", "prompt"]
    for fmt in formats:
        images = try_format(fmt, args.px)
        if images:
            args.out.mkdir(parents=True, exist_ok=True)
            for i, png in enumerate(images):
                p = args.out / f"smoke_{fmt}_{i}.png"
                if isinstance(png, str):  # URL returned — download for inspection
                    import urllib.request
                    with urllib.request.urlopen(png) as r:
                        png = r.read()
                p.write_bytes(png)
                print("saved:", p)
            print(f"\nSUCCESS — endpoint speaks {fmt!r}. "
                  f"Set RUNPOD_INPUT_FORMAT={fmt} for gallery exports.")
            return 0
    print("\nAll formats failed — see errors above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
