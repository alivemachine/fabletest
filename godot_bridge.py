"""Local HTTP bridge between the Python world core and a Godot client.

The bridge owns one evolving world instance. A Godot scene polls /frame with
its current world-space center and receives a quantized, player-centered chunk
of biome, height, vegetation, river, cloud, and lighting data.
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

import world_core as wc


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _q_u8(field: np.ndarray) -> list[int]:
    return np.rint(np.clip(field, 0.0, 1.0) * 255.0).astype(np.uint8).ravel().tolist()


def _q_i16(field: np.ndarray) -> list[int]:
    return field.astype(np.int16).ravel().tolist()


def _q_u8_array(field: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.rint(np.clip(field, 0.0, 1.0) * 255.0).astype(np.uint8).ravel()
    )


def _q_i16_array(field: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(field.astype("<i2", copy=False).ravel())


def _mean_u8(field: np.ndarray) -> int:
    return int(np.rint(np.clip(float(np.mean(field)), 0.0, 1.0) * 255.0))


class WorldBridge:
    def __init__(self, seed: int, size: int, civ_count: int):
        self.seed = int(seed)
        self.size = int(size)
        self.civ_count = int(civ_count)
        self.t = 0.0
        self.lock = threading.Lock()
        self.world = wc.build_world(self.seed, self.size, self.civ_count)

    def _rebuild(self, seed: int, size: int, civ_count: int) -> None:
        seed = int(seed)
        size = int(size)
        civ_count = int(civ_count)
        if (seed, size, civ_count) == (self.seed, self.size, self.civ_count):
            return
        self.seed = seed
        self.size = size
        self.civ_count = civ_count
        self.t = 0.0
        self.world = wc.build_world(self.seed, self.size, self.civ_count)

    def _advance(self, dt_seconds: float, speed_days_per_second: float,
                 playing: bool, sea_level: float, season_amp: float,
                 tide_amp: float) -> None:
        if not playing:
            return
        dt_days = max(0.0, dt_seconds) * max(0.0, speed_days_per_second)
        if dt_days <= 0.0:
            return
        mid_t = self.t + dt_days * 0.5
        _sea_eff, season_off, _sun_x = wc.frame_params(mid_t, sea_level, tide_amp, season_amp)
        eco = getattr(self.world, "eco", None)
        if eco is not None:
            eco.step(dt_days, sea_level, season_off)
        self.t += dt_days

    def _payload(self, ws: wc.WorldSlice, st: dict[str, Any], target_tiles: int) -> dict[str, Any]:
        fields = self._payload_fields(ws, st)
        return {
            "seed": self.seed,
            "planet_size": self.size,
            "view_tiles_target": int(target_tiles),
            "size": int(ws.size),
            "time_days": float(st["t"]),
            "span": float(ws.span),
            "center": {"cx": float(ws.cx), "cy": float(ws.cy)},
            "tile_world": float(ws.span / max(ws.size, 1)),
            "sea_level": float(st["sea_level"]),
            "sea_effective": float(st["sea_eff"]),
            "sunlight_mean": _mean_u8(st["sunlight"]),
            "cloud_mean": _mean_u8(st["clouds"]),
            "fields": {name: arr.tolist() for name, arr in fields.items()},
        }

    def _payload_fields(self, ws: wc.WorldSlice, st: dict[str, Any]) -> dict[str, np.ndarray]:
        river_alpha = getattr(ws, "river_alpha", np.zeros_like(ws.elev))
        brook_alpha = getattr(ws, "brook_alpha", np.zeros_like(ws.elev))
        river_gate = np.clip(ws.river_disc / max(float(st["river_thr"]), 1.0), 0.0, 1.0) ** 0.6
        river = np.maximum(river_alpha * river_gate, brook_alpha).astype(np.float32)
        veg_live = np.clip(st["veg"] * st["veg_health"], 0.0, 1.0)
        return {
            "biome_id": _q_i16_array(st["biome_id"]),
            "height": _q_u8_array(ws.elev),
            "sunlight": _q_u8_array(st["sunlight"]),
            "river": _q_u8_array(river),
            "vegetation": _q_u8_array(veg_live),
            "scorch": _q_u8_array(st["scorch"]),
        }

    def _payload_binary(self, ws: wc.WorldSlice, st: dict[str, Any], target_tiles: int) -> bytes:
        fields = self._payload_fields(ws, st)
        header = struct.pack(
            "<4sHHIHHfffffffBBH",
            b"FTB1",
            1,
            int(ws.size),
            int(self.seed),
            int(self.size),
            int(target_tiles),
            float(st["t"]),
            float(ws.span),
            float(ws.cx),
            float(ws.cy),
            float(ws.span / max(ws.size, 1)),
            float(st["sea_level"]),
            float(st["sea_eff"]),
            _mean_u8(st["sunlight"]),
            _mean_u8(st["clouds"]),
            0,
        )
        parts = [header]
        for name in ("biome_id", "height", "sunlight", "river", "vegetation", "scorch"):
            parts.append(fields[name].tobytes())
        return b"".join(parts)

    def snapshot(self, *, seed: int | None = None, size: int | None = None,
                 civ_count: int | None = None, cx: float = 0.5, cy: float = 0.5,
                 zoom: float = 12.0, view_tiles: int = 24, dt_seconds: float = 0.0,
                 speed: float = 0.35, playing: bool = True, sea_level: float = 0.42,
                 river_thr: float | None = None, season_amp: float = 0.18,
                 tide_amp: float = 0.012, day_night: float = 0.65,
                 reset: bool = False) -> dict[str, Any]:
        with self.lock:
            self._rebuild(seed if seed is not None else self.seed,
                          size if size is not None else self.size,
                          civ_count if civ_count is not None else self.civ_count)
            if reset:
                eco = getattr(self.world, "eco", None)
                if eco is not None:
                    eco.reset()
                self.t = 0.0
            river_thr = (wc.default_river_threshold(self.size)
                         if river_thr is None else float(river_thr))
            self._advance(dt_seconds, speed, playing, sea_level, season_amp, tide_amp)
            zoom = max(1.0, float(zoom))
            view_tiles = max(8, int(view_tiles))
            chunk = self.world.stream_view(cx % 1.0, cy % 1.0, zoom, view_tiles)
            st = wc.state(chunk, self.t, sea_level, river_thr, season_amp, tide_amp, day_night)
            return self._payload(chunk, st, view_tiles)

    def snapshot_binary(self, *, seed: int | None = None, size: int | None = None,
                        civ_count: int | None = None, cx: float = 0.5, cy: float = 0.5,
                        zoom: float = 12.0, view_tiles: int = 24, dt_seconds: float = 0.0,
                        speed: float = 0.35, playing: bool = True, sea_level: float = 0.42,
                        river_thr: float | None = None, season_amp: float = 0.18,
                        tide_amp: float = 0.012, day_night: float = 0.65,
                        reset: bool = False) -> bytes:
        with self.lock:
            self._rebuild(seed if seed is not None else self.seed,
                          size if size is not None else self.size,
                          civ_count if civ_count is not None else self.civ_count)
            if reset:
                eco = getattr(self.world, "eco", None)
                if eco is not None:
                    eco.reset()
                self.t = 0.0
            river_thr = (wc.default_river_threshold(self.size)
                         if river_thr is None else float(river_thr))
            self._advance(dt_seconds, speed, playing, sea_level, season_amp, tide_amp)
            zoom = max(1.0, float(zoom))
            view_tiles = max(8, int(view_tiles))
            chunk = self.world.stream_view(cx % 1.0, cy % 1.0, zoom, view_tiles)
            st = wc.state(chunk, self.t, sea_level, river_thr, season_amp, tide_amp, day_night)
            return self._payload_binary(chunk, st, view_tiles)


def _parse_bool(values: dict[str, list[str]], name: str, default: bool) -> bool:
    raw = values.get(name, ["1" if default else "0"])[0].strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parse_float(values: dict[str, list[str]], name: str, default: float) -> float:
    try:
        return float(values.get(name, [default])[0])
    except (TypeError, ValueError):
        return float(default)


def _parse_int(values: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(values.get(name, [default])[0])
    except (TypeError, ValueError):
        return int(default)


class BridgeHandler(BaseHTTPRequestHandler):
    bridge: WorldBridge | None = None
    protocol_version = "HTTP/1.1"

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"ok": True})
            return
        if parsed.path not in {"/frame", "/frame.bin"}:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        bridge = self.bridge
        if bridge is None:
            self._send_json({"error": "bridge unavailable"}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        qs = parse_qs(parsed.query)
        args = {
            "seed": _parse_int(qs, "seed", bridge.seed),
            "size": _parse_int(qs, "size", bridge.size),
            "civ_count": _parse_int(qs, "civ_count", bridge.civ_count),
            "cx": _clamp(_parse_float(qs, "cx", 0.5), 0.0, 1.0),
            "cy": _clamp(_parse_float(qs, "cy", 0.5), 0.0, 1.0),
            "zoom": max(1.0, _parse_float(qs, "zoom", 12.0)),
            "view_tiles": max(8, _parse_int(qs, "view_tiles", 24)),
            "dt_seconds": max(0.0, _parse_float(qs, "dt", 0.0)),
            "speed": max(0.0, _parse_float(qs, "speed", 0.35)),
            "playing": _parse_bool(qs, "playing", True),
            "sea_level": _clamp(_parse_float(qs, "sea_level", 0.42), 0.20, 0.70),
            "river_thr": _parse_float(qs, "river_thr", wc.default_river_threshold(bridge.size)),
            "season_amp": _clamp(_parse_float(qs, "season_amp", 0.18), 0.0, 0.5),
            "tide_amp": _clamp(_parse_float(qs, "tide_amp", 0.012), 0.0, 0.05),
            "day_night": _clamp(_parse_float(qs, "day_night", 0.65), 0.0, 1.0),
            "reset": _parse_bool(qs, "reset", False),
        }
        if parsed.path == "/frame.bin":
            self._send_binary(bridge.snapshot_binary(**args))
            return
        payload = bridge.snapshot(**args)
        self._send_json(payload)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "keep-alive")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, body: bytes, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve world_core chunks to a Godot client.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--civ-count", type=int, default=3)
    args = parser.parse_args()

    BridgeHandler.bridge = WorldBridge(args.seed, args.size, args.civ_count)
    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    print(f"serving Godot bridge at http://{args.host}:{args.port}/frame")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()