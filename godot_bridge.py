"""Local HTTP bridge between the Python world core and a Godot client.

The bridge owns one evolving world instance. A Godot scene polls /frame with
its current world-space center and receives a quantized, player-centered chunk
of biome, height, vegetation, river, cloud, and lighting data.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
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

        # --- directional flow: water runs downhill of the CARVED terrain (the
        # river valleys are cut into ws.elev), so the channel direction is the
        # negative gradient. Angle is quantized to a byte (0..255 == 0..TAU,
        # +x east, +y south); speed is nonzero only on actual watercourses and
        # scales with discharge, so a trunk river pushes harder than a brook.
        gy, gx = np.gradient(ws.elev, ws.pixel_world, ws.pixel_world)
        ang = np.arctan2(-gy, -gx)
        flow_dir = ((ang / (2.0 * np.pi)) % 1.0).astype(np.float32)
        disc_n = np.clip(np.log1p(np.maximum(ws.river_disc, 0.0))
                         / max(float(getattr(ws, "log_norm", 1.0)), 1e-6), 0.0, 1.0)
        on_stream = river > 0.02
        still_water = ws.elev < float(st["sea_eff"])   # sea/lakes: no current
        flow_speed = np.where(on_stream & ~still_water,
                              np.clip(0.25 + 0.75 * disc_n, 0.25, 1.0),
                              0.0).astype(np.float32)

        # --- living fauna (same math as the fauna layer): herd + predator
        # densities the game can spawn/behave from
        fh = st["fauna_health"]
        herb, pred = wc.fauna_field(st["veg"], st["t"])
        civ_p, fid = wc.civ_population(ws, st["t"])
        herb = np.clip(herb * fh * (1.0 - 0.7 * np.clip(civ_p, 0.0, 1.0)), 0.0, 1.0)
        pred = np.clip(pred * fh, 0.0, 1.0)

        # --- structures from the M4 settlement expand(): buildings paint
        # alpha 1.0, roads ~0.75 -> a 3-state byte the game can collide with
        settle_a, _settle_rgb = wc._settlements(ws, st["t"], float(st["sea_level"]))
        structure = np.zeros(settle_a.shape, np.uint8)
        structure[settle_a >= 0.4] = 1                 # road
        structure[settle_a >= 0.95] = 2                # building
        faction = np.clip(fid.astype(np.int16) + 1, 0, 255).astype(np.uint8)

        return {
            "biome_id": _q_i16_array(st["biome_id"]),
            "height": _q_u8_array(ws.elev),
            "sunlight": _q_u8_array(st["sunlight"]),
            "river": _q_u8_array(river),
            "vegetation": _q_u8_array(veg_live),
            "scorch": _q_u8_array(st["scorch"]),
            "flow_dir": _q_u8_array(flow_dir),
            "flow_speed": _q_u8_array(flow_speed),
            "fauna_herb": _q_u8_array(herb),
            "fauna_pred": _q_u8_array(pred),
            "structure": np.ascontiguousarray(structure.ravel()),
            "faction": np.ascontiguousarray(faction.ravel()),
            "moisture": _q_u8_array(ws.moist),
            "temperature": _q_u8_array(st["tf"]),
        }

    def _bitmap_payload(self, ws: wc.WorldSlice, st: dict[str, Any]) -> dict[str, np.ndarray]:
        """Build the three bitmaps Godot reads directly — no per-field reconstruction.

        color_map   (N×N×3 uint8)  — BIOME layer only (no lighting). Pure stable
                                     colors from worldgen.BIOME_COLORS. Godot keys
                                     each pixel into its TileDefinition (texture,
                                     physics, walkability, resource …). Must not
                                     include lighting — the composite render shifts
                                     every frame with day/night and breaks lookups.
        property_map (N×N×4 uint8) — per-tile game data that Godot CAN'T infer from
                                     color alone:
                                       R = height  (0-255, normalized above sea_level)
                                       G = surface flags: 0=land 1=water 2=river-on-land
                                       B = structure: 0=none 1=road 2=building
                                       A = faction id+1 (0=unclaimed)
        data_map    (N×N×4 uint8)  — per-tile world-state channels:
                                       R = sunlight  (Godot applies this as tile lighting)
                                       G = vegetation (live)
                                       B = flow direction (0-255 → 0-TAU)
                                       A = flow speed (0=still)
        """
        fields = self._payload_fields(ws, st)

        # --- color_map: pure stable biome color (no lighting, no compositing).
        # The "biome" layer returns BIOME_LUT[biome_id] — exact matches to the
        # TILE_PALETTE keys in terrain_chunk.gd. Day/night is applied separately
        # via data_map.R (sunlight) so Godot controls illumination itself.
        biome_rgb = wc.colorize(st, "biome")
        color_map = np.clip(biome_rgb, 0, 255).astype(np.uint8)          # N×N×3

        # --- property_map
        n = ws.size
        prop = np.zeros((n, n, 4), np.uint8)
        elev = ws.elev
        sea_eff  = float(st["sea_eff"])   # tide-adjusted: determines water boundary
        sea_level = float(st["sea_level"]) # stable geographic baseline: used for height
        range_norm = max(1.0 - sea_level, 1e-6)
        # R: height — land tiles use their own elevation; water tiles use the
        # current tide-adjusted waterline (sea_eff) so the sea surface rises
        # and falls with the tide instead of sitting at a fixed 8 px.  Land
        # tiles near the waterline transition smoothly because the tile at
        # exactly sea_eff encodes the same height whether it's land or water.
        land_mask = elev >= sea_eff
        sea_eff_norm = float(np.clip((sea_eff - sea_level) / range_norm, 0.0, 1.0))
        water_height_byte = int(sea_eff_norm * 255 + 0.5)
        prop[..., 0] = np.where(land_mask,
                                np.clip((elev - sea_level) / range_norm * 255, 0, 255).astype(np.uint8),
                                water_height_byte)
        # G: surface type
        river_alpha = getattr(ws, "river_alpha", np.zeros_like(elev))
        brook_alpha = getattr(ws, "brook_alpha", np.zeros_like(elev))
        river_gate = np.clip(ws.river_disc / max(float(st["river_thr"]), 1.0), 0.0, 1.0) ** 0.6
        river = np.maximum(river_alpha * river_gate, brook_alpha)
        surf = np.zeros((n, n), np.uint8)
        surf[~land_mask] = 1                              # water
        surf[land_mask & (river > 0.05)] = 2             # river on land
        prop[..., 1] = surf
        # B: structure (already computed in _payload_fields)
        prop[..., 2] = fields["structure"].reshape(n, n)
        # A: faction
        prop[..., 3] = fields["faction"].reshape(n, n)

        # --- data_map: R=sunlight G=vegetation B=flow_dir A=flow_speed
        data = np.zeros((n, n, 4), np.uint8)
        data[..., 0] = fields["sunlight"].reshape(n, n)   # per-tile lighting
        data[..., 1] = fields["vegetation"].reshape(n, n)
        data[..., 2] = fields["flow_dir"].reshape(n, n)
        data[..., 3] = fields["flow_speed"].reshape(n, n)

        return {
            "color_map": np.ascontiguousarray(color_map),
            "property_map": np.ascontiguousarray(prop),
            "data_map": np.ascontiguousarray(data),
        }

    def _payload_binary(self, ws: wc.WorldSlice, st: dict[str, Any], target_tiles: int) -> bytes:
        bitmaps = self._bitmap_payload(ws, st)
        # Header: magic "FTB1", version=3, then same geometry fields as before.
        # sea_level (not sea_eff) is the stable geographic baseline Godot uses
        # for column-height calculation — tide only affects the waterline flag
        # already baked into property_map.G.
        header = struct.pack(
            "<4sHHIHHfffffffBBH",
            b"FTB1",
            3,
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
        return b"".join([
            header,
            bitmaps["color_map"].tobytes(),     # N*N*3 bytes
            bitmaps["property_map"].tobytes(),  # N*N*4 bytes
            bitmaps["data_map"].tobytes(),      # N*N*4 bytes
        ])

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


def _evict_port(port: int) -> None:
    """Kill any process already listening on *port* before we bind to it.
    Uses netstat on Windows, lsof on POSIX. Best-effort — never blocks startup."""
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL
            )
            pids: set[int] = set()
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                    try:
                        pid = int(parts[4])
                        if pid != os.getpid():
                            pids.add(pid)
                    except ValueError:
                        pass
            for pid in pids:
                subprocess.call(
                    ["taskkill", "/F", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            if pids:
                import time as _t; _t.sleep(0.5)
        else:
            out = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"], text=True, stderr=subprocess.DEVNULL
            )
            for pid_s in out.split():
                try:
                    pid = int(pid_s)
                    if pid != os.getpid():
                        os.kill(pid, signal.SIGTERM)
                except (ValueError, OSError):
                    pass
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve world_core chunks to a Godot client.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--civ-count", type=int, default=3)
    args = parser.parse_args()

    BridgeHandler.bridge = WorldBridge(args.seed, args.size, args.civ_count)
    _evict_port(args.port)
    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    print(f"serving Godot bridge at http://{args.host}:{args.port}/frame")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        print(f"\nBridge crashed: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()