#!/usr/bin/env python3
"""Generate Full HD Godot client screenshots and publish a web manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import world_core as wc


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "web" / "godot_frames"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.json"


@dataclass
class ShotPoint:
    cx: float
    cy: float
    kind: str


def _wrap01(v: float) -> float:
    return v % 1.0


def build_points(seed: int, size: int, civ_count: int, shots: int) -> list[ShotPoint]:
    world = wc.build_world(seed, size, civ_count)
    cores = list(getattr(world, "civ_cores", []))
    rng = np.random.default_rng(seed ^ 0x9E3779B9)

    points: list[ShotPoint] = []
    if cores:
        near_target = min(shots, max(3, shots // 2))
        order = rng.permutation(len(cores))
        for i in range(near_target):
            core = cores[int(order[i % len(order)])]
            cy, cx = float(core[0]), float(core[1])
            points.append(
                ShotPoint(
                    cx=_wrap01(cx + float(rng.uniform(-0.03, 0.03))),
                    cy=_wrap01(cy + float(rng.uniform(-0.03, 0.03))),
                    kind="near_settlement",
                )
            )

    while len(points) < shots:
        points.append(
            ShotPoint(
                cx=float(rng.uniform(0.0, 1.0)),
                cy=float(rng.uniform(0.0, 1.0)),
                kind="random",
            )
        )

    rng.shuffle(points)
    return points[:shots]


def wait_for_bridge(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.8) as resp:
                if resp.status == 200:
                    return
        except urllib.error.URLError:
            pass
        time.sleep(0.12)
    raise RuntimeError(f"bridge did not become healthy at {url}")


def run_capture(
    godot_bin: str,
    output_dir: Path,
    points_file: Path,
    width: int,
    height: int,
) -> None:
    cmd = [
        godot_bin,
        "--headless",
        "--path",
        str(ROOT / "godot_client"),
        "--script",
        "res://scripts/capture_runner.gd",
        "--",
        "--output-dir",
        str(output_dir),
        "--points-json",
        str(points_file),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Godot capture failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)


def write_manifest(
    manifest_path: Path,
    output_dir: Path,
    points: list[ShotPoint],
    seed: int,
    size: int,
    civ_count: int,
    width: int,
    height: int,
) -> None:
    missing: list[str] = []
    entries = []
    for i, p in enumerate(points):
        filename = f"frame_{i:02d}.png"
        if not (output_dir / filename).exists():
            missing.append(filename)
        entries.append(
            {
                "file": filename,
                "kind": p.kind,
                "cx": round(p.cx, 6),
                "cy": round(p.cy, 6),
            }
        )
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "seed": seed,
        "size": size,
        "civ_count": civ_count,
        "width": width,
        "height": height,
        "screenshots": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Generated screenshots:")
    for item in entries:
        exists = (output_dir / item["file"]).exists()
        marker = "✓" if exists else "!"
        print(f" {marker} {item['file']} ({item['kind']}, cx={item['cx']}, cy={item['cy']})")
    if missing:
        raise RuntimeError(f"Missing expected screenshots: {', '.join(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 10 Full HD Godot client screenshots and update web manifest."
    )
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN", "godot4"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--civ-count", type=int, default=3)
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--health-timeout", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shots <= 0:
        raise SystemExit("--shots must be > 0")
    if shutil.which(args.godot_bin) is None:
        raise SystemExit(
            f"Could not find Godot executable '{args.godot_bin}'. "
            "Set --godot-bin or GODOT_BIN."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("frame_*.png"):
        old.unlink()

    points = build_points(args.seed, args.size, args.civ_count, args.shots)
    tmpdir = Path(tempfile.mkdtemp(prefix="godot-screenshot-gen-"))
    points_file = tmpdir / "points.json"
    points_file.write_text(json.dumps([asdict(p) for p in points]), encoding="utf-8")

    bridge_cmd = [
        sys.executable,
        str(ROOT / "godot_bridge.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--seed",
        str(args.seed),
        "--size",
        str(args.size),
        "--civ-count",
        str(args.civ_count),
    ]
    bridge = subprocess.Popen(
        bridge_cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_bridge(args.host, args.port, args.health_timeout)
        run_capture(args.godot_bin, output_dir, points_file, args.width, args.height)
        write_manifest(
            args.manifest.resolve(),
            output_dir,
            points,
            args.seed,
            args.size,
            args.civ_count,
            args.width,
            args.height,
        )
    finally:
        if bridge.poll() is None:
            try:
                bridge.send_signal(signal.SIGINT)
            except Exception:
                bridge.terminate()
            try:
                bridge.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                bridge.kill()
        if bridge.stdout:
            log = bridge.stdout.read().strip()
            if log:
                print(log)
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
