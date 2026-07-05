"""
lighting.py — sun-driven lighting from the current heightfield: per-pixel sun
direction, terrain normals, projected terrain shadows, and cloud shadows.
Everything here is a pure function of the window's grids + the sun position,
so lighting stays seekable like the rest of the render core.
"""

import numpy as np

from common import smoothstep, _shift_max8

NORMAL_RELIEF_WORLD = 0.018
SHADOW_RELIEF_WORLD = 0.090
SOLAR_TILT = np.deg2rad(47.0)   # season_off in [-0.5,0.5] -> +/-23.5 deg
TERRAIN_SHADOW_STEPS = 56
CLOUD_WORLD_HEIGHT = 0.010
CLOUD_SHADOW_STRENGTH = 0.50


def _sample_offset(a, ox, oy, fill):
    """Sample `a` with a non-wrapping integer offset: out[y,x] = a[y+oy,x+ox]."""
    out = np.full_like(a, fill)
    h, w = a.shape
    if abs(ox) >= w or abs(oy) >= h:
        return out
    sx0, sx1 = max(0, ox), min(w, w + ox)
    sy0, sy1 = max(0, oy), min(h, h + oy)
    dx0, dx1 = max(0, -ox), min(w, w - ox)
    dy0, dy1 = max(0, -oy), min(h, h - oy)
    out[dy0:dy1, dx0:dx1] = a[sy0:sy1, sx0:sx1]
    return out


def _sun_field(ws, sun_x, season_off):
    """Per-pixel sun direction in local east/south/up coordinates."""
    lat = ws.lat_signed * (0.5 * np.pi)
    hour = (ws.xn[None, :] - sun_x) * (2 * np.pi)
    decl = SOLAR_TILT * season_off
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_hour, cos_hour = np.sin(hour), np.cos(hour)
    sin_decl, cos_decl = np.sin(decl), np.cos(decl)
    sx = -cos_decl * sin_hour
    sy = sin_lat * cos_decl * cos_hour - cos_lat * sin_decl
    sz = sin_lat * sin_decl + cos_lat * cos_decl * cos_hour
    inv = 1.0 / np.maximum(np.sqrt(sx * sx + sy * sy + sz * sz), 1e-6)
    return ((sx * inv).astype(np.float32),
            (sy * inv).astype(np.float32),
            (sz * inv).astype(np.float32))


def _terrain_shadow(height, sx, sy, sz, pixel_world):
    """Approximate terrain cast-shadow visibility in [0,1] for one sun ray."""
    if sz <= 1e-4:
        return np.zeros_like(height, np.float32)
    horiz = float(np.hypot(sx, sy))
    if horiz <= 1e-6:
        return np.ones_like(height, np.float32)
    dx, dy = sx / horiz, sy / horiz
    rise = pixel_world * sz / horiz
    soft = max(pixel_world * SHADOW_RELIEF_WORLD * 1.8, 1e-5)
    horizon = np.full_like(height, -1e9, np.float32)
    steps = min(max(height.shape) - 1,
                max(12, min(TERRAIN_SHADOW_STEPS,
                            int(12 + 34 * horiz / max(sz, 0.12)))))
    for step in range(1, steps + 1):
        ox = int(round(dx * step))
        oy = int(round(dy * step))
        if ox == 0 and oy == 0:
            continue
        sample = _sample_offset(height, ox, oy, -1e9)
        np.maximum(horizon, sample - step * rise, out=horizon)
    return smoothstep((height - horizon + soft) / (2 * soft)).astype(np.float32)


def _cloud_shadow(clouds, sx, sy, sz, pixel_world):
    """Project cloud cover onto the ground along the sun ray."""
    if sz <= 1e-4:
        return np.ones_like(clouds, np.float32)
    scale = CLOUD_WORLD_HEIGHT / max(sz * pixel_world, 1e-6)
    ox = int(round(sx * scale))
    oy = int(round(sy * scale))
    cover = _sample_offset(clouds, ox, oy, 0.0).astype(np.float32)
    cover = 0.65 * cover + 0.35 * _shift_max8(cover)
    return np.clip(1.0 - CLOUD_SHADOW_STRENGTH * cover,
                   1.0 - CLOUD_SHADOW_STRENGTH, 1.0).astype(np.float32)


def _lighting_fields(ws, sun_x, season_off, day_night, clouds):
    """Derived lighting payload: normals, sun visibility, and shadow masks."""
    sx, sy, sz = _sun_field(ws, sun_x, season_off)
    sun = np.stack((sx, sy, sz), axis=-1)
    ndotl = np.clip(np.sum(ws.normal * sun, axis=2), 0, 1).astype(np.float32)
    day = smoothstep((sz + 0.10) / 0.20).astype(np.float32)
    cy = ws.size // 2
    cx = ws.size // 2
    sx0 = float(sx[cy, cx])
    sy0 = float(sy[cy, cx])
    sz0 = float(max(sz[cy, cx], 0.0))
    terrain_vis = _terrain_shadow(ws.height, sx0, sy0, sz0, ws.pixel_world)
    cloud_vis = _cloud_shadow(clouds, sx0, sy0, max(sz0, 0.08), ws.pixel_world)
    direct = ndotl * terrain_vis * cloud_vis
    lit = day * (0.28 + 0.72 * direct)
    floor = 1.0 - 0.72 * day_night
    sunlight = floor + (1.0 - floor) * lit
    return {
        "normal": ws.normal,
        "sun_dir": np.array([sx0, sy0, sz0], np.float32),
        "sun_up": np.clip(sz, 0, 1).astype(np.float32),
        "terrain_shadow": terrain_vis,
        "cloud_shadow": cloud_vis,
        "sunlight": np.clip(sunlight, 0, 1).astype(np.float32),
    }
