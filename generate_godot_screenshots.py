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


def build_points(
    seed: int, size: int, civ_count: int, shots: int, sea_level: float = 0.42
) -> list[ShotPoint]:
    world = wc.build_world(seed, size, civ_count)
    cores = list(getattr(world, "civ_cores", []))
    rng = np.random.default_rng(seed ^ 0x9E3779B9)

    def on_land(cx: float, cy: float) -> bool:
        # A zoom-12 view spans ~16 planet cells, so one land cell isn't
        # enough — require a 5x5 block of land around the point or the
        # rendered frame can still be all ocean.
        j = int(cx * world.size) % world.size
        i = int(cy * world.size) % world.size
        block = world.elev[np.ix_([(i + di) % world.size for di in range(-2, 3)],
                                  [(j + dj) % world.size for dj in range(-2, 3)])]
        return float(block.min()) > sea_level + 0.01

    def nudge_to_land(cx: float, cy: float) -> tuple[float, float]:
        if on_land(cx, cy):
            return cx, cy
        for _ in range(24):
            nx = _wrap01(cx + float(rng.uniform(-0.05, 0.05)))
            ny = _wrap01(cy + float(rng.uniform(-0.05, 0.05)))
            if on_land(nx, ny):
                return nx, ny
        return cx, cy

    points: list[ShotPoint] = []
    if cores:
        near_target = min(shots, max(3, shots // 2))
        order = rng.permutation(len(cores))
        for i in range(near_target):
            core = cores[int(order[i % len(order)])]
            cy, cx = float(core[0]), float(core[1])
            cx, cy = nudge_to_land(
                _wrap01(cx + float(rng.uniform(-0.02, 0.02))),
                _wrap01(cy + float(rng.uniform(-0.02, 0.02))),
            )
            points.append(ShotPoint(cx=cx, cy=cy, kind="near_settlement"))

    # Random shots stick to land — an all-ocean frame shows nothing worth
    # publishing to the gallery.
    attempts = 0
    while len(points) < shots and attempts < shots * 400:
        attempts += 1
        cx = float(rng.uniform(0.0, 1.0))
        cy = float(rng.uniform(0.0, 1.0))
        if on_land(cx, cy):
            points.append(ShotPoint(cx=cx, cy=cy, kind="random"))
    while len(points) < shots:  # pathological all-ocean world
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


def import_project(godot_bin: str) -> None:
    """Import project resources once so the capture run finds them ready.

    A fresh checkout has no .godot/ import cache; running the capture script
    directly would fail to load scenes. Importing is a pure asset step, so the
    dummy headless renderer is fine here."""
    cmd = [godot_bin, "--headless", "--path", str(ROOT / "godot_client"), "--import"]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        print(
            f"warning: project import exited {proc.returncode} (continuing):\n"
            f"{proc.stderr.strip()[-2000:]}",
            file=sys.stderr,
        )


def run_capture(
    godot_bin: str,
    output_dir: Path,
    points_file: Path,
    width: int,
    height: int,
    renderer: str = "opengl3",
    sim_day: float = -1.0,
    seed: int = -1,
    size: int = -1,
    civ_count: int = -1,
) -> None:
    # Godot 4's --headless mode uses the dummy rendering server, so the root
    # viewport has no texture to read back — captures need a real renderer.
    # Default to the GL compatibility driver, which renders under xvfb with
    # Mesa's software rasterizer (no GPU required).
    if renderer == "headless":
        render_args = ["--headless"]
    else:
        render_args = ["--rendering-driver", renderer]
    cmd = [
        godot_bin,
        *render_args,
        "--resolution",
        f"{width}x{height}",
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
        "--sim-day",
        str(sim_day),
        "--seed",
        str(seed),
        "--size",
        str(size),
        "--civ-count",
        str(civ_count),
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
    sim_day: float = -1.0,
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
        "sim_day": sim_day,
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
    parser.add_argument(
        "--sim-day",
        type=float,
        default=1152.5,
        help="Absolute sim day to render at (N.5 = local noon, year = 96 days)."
        " Default is noon of year 12, deep enough into the history timeline"
        " for settlements to exist. Pass -1 for the live clock (day 0).",
    )
    parser.add_argument("--sea-level", type=float, default=0.42)
    parser.add_argument(
        "--renderer",
        default=os.environ.get("GODOT_RENDERER", "opengl3"),
        choices=["opengl3", "opengl3_es", "vulkan", "headless"],
        help="Rendering driver for the capture run. 'headless' uses Godot's "
        "dummy renderer and cannot capture pixels; the default 'opengl3' "
        "works under xvfb with Mesa software GL.",
    )
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

    points = build_points(
        args.seed, args.size, args.civ_count, args.shots, sea_level=args.sea_level
    )
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
        import_project(args.godot_bin)
        run_capture(
            args.godot_bin,
            output_dir,
            points_file,
            args.width,
            args.height,
            renderer=args.renderer,
            sim_day=args.sim_day,
            seed=args.seed,
            size=args.size,
            civ_count=args.civ_count,
        )
        write_manifest(
            args.manifest.resolve(),
            output_dir,
            points,
            args.seed,
            args.size,
            args.civ_count,
            args.width,
            args.height,
            sim_day=args.sim_day,
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
