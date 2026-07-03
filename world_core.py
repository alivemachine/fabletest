"""
world_core.py — the pure render core, shared by every interface.

world(seed, x, y, t) -> layers. This module owns everything between
worldgen.py's static fields and pixels on a screen:

    - vectorized biome classification (same thresholds as worldgen)
    - the M2 time layer: day/night sweep, hemisphere-opposed seasons, tides
    - per-layer RGB renderers (composite, elevation, temperature, moisture,
      flow, biomes, daylight)

Consumers: world_viewer.py (desktop matplotlib console) and web/index.html
(the same code running in the browser through Pyodide). No matplotlib, no
PIL, no I/O here — numpy in, uint8 RGB out.
"""

import numpy as np

from worldgen import elevation_field, moisture_field, compute_rivers, BIOME_COLORS

# ---------------------------------------------------------------------------
# Time model (M2). t is measured in sim DAYS.
# ---------------------------------------------------------------------------
YEAR_DAYS = 96.0          # one year = 96 days
TIDE_PERIOD = 0.52        # ~semi-diurnal tide

LAYERS = [
    ("composite", "World"),
    ("elevation", "Elevation"),
    ("temperature", "Temperature"),
    ("moisture", "Moisture"),
    ("flow", "Flow"),
    ("biome", "Biomes"),
    ("light", "Daylight"),
]

BIOME_NAMES = list(BIOME_COLORS.keys())
BIOME_LUT = np.array([BIOME_COLORS[n] for n in BIOME_NAMES], dtype=np.float32)
BID = {n: i for i, n in enumerate(BIOME_NAMES)}


def biome_ids(e, t, m, sea):
    """Vectorized version of worldgen.classify_biomes -> int16 ids."""
    water = e < sea
    conds = [
        water & (e < sea - 0.12),
        water & (e >= sea - 0.05),
        water,
        e > 0.92,
        e > 0.82,
        e < sea + 0.015,
        t < 0.2,
        t < 0.35,
        t < 0.5,
        m < 0.25,
        m < 0.45,
        m < 0.65,
        m < 0.82,
    ]
    choices = [BID[n] for n in (
        "deep_ocean", "shallow", "ocean", "high_peak", "mountain", "beach",
        "snow", "tundra", "taiga", "desert", "savanna", "grassland", "forest")]
    return np.select(conds, choices, default=BID["jungle"]).astype(np.int16)


def temperature_t(elev, lat, lat_signed, sea_eff, season_off):
    t = lat + season_off * lat_signed - np.clip(elev - sea_eff, 0, None) * 0.9
    return np.clip(t, 0, 1)


def daylight_row(xn, sun_x, depth):
    """Per-longitude light factor: 1 at noon, floor at midnight."""
    c = np.cos(2 * np.pi * (xn - sun_x))
    s = np.clip((c + 0.15) / 0.45, 0, 1)
    s = s * s * (3 - 2 * s)
    floor = 1 - 0.72 * depth
    return floor + (1 - floor) * s


def color_ramp(v, stops, colors):
    """v in [0,1] -> RGB via piecewise-linear ramp. colors: (k,3)."""
    colors = np.asarray(colors, dtype=np.float32)
    return np.stack([np.interp(v, stops, colors[:, c]) for c in range(3)], axis=-1)


TMP_RAMP = ([0.0, 0.5, 1.0], [(58, 103, 196), (232, 228, 216), (200, 80, 46)])
MOI_RAMP = ([0.0, 0.5, 1.0], [(201, 163, 90), (214, 214, 196), (62, 143, 122)])
FLOW_RAMP = ([0.0, 0.5, 1.0], [(12, 18, 30), (32, 110, 150), (180, 235, 250)])


class WorldSlice:
    """Static per-resolution data + grids (full res, or strided for thumbs)."""

    def __init__(self, elev, moist, accum):
        self.elev, self.moist, self.accum = elev, moist, accum
        size = elev.shape[0]
        yn = (np.arange(size, dtype=np.float32) / size)[:, None]
        self.lat = 1 - np.abs(yn - 0.5) * 2
        self.lat_signed = (0.5 - yn) * 2
        self.xn = np.arange(size, dtype=np.float32) / size
        gy, gx = np.gradient(elev)
        self.shade = np.clip(1 - (gx + gy) * 2.2, 0.75, 1.25).astype(np.float32)
        self.log_accum = np.log1p(accum) / np.log1p(accum.max())

    def strided(self, st):
        s = WorldSlice.__new__(WorldSlice)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(s, k, v[::st, ::st] if v.ndim == 2 else v[::st])
        return s


def build_world(seed, size):
    """The heavy, seed-only part: fields + D8 flow. Returns a WorldSlice."""
    elev = elevation_field(size, seed).astype(np.float32)
    moist = moisture_field(size, seed).astype(np.float32)
    _, accum = compute_rivers(elev, 0.5)     # accumulation is sea-level-free
    return WorldSlice(elev, moist, accum.astype(np.float32))


def default_river_threshold(size):
    """350 is tuned for 512²; scale down for smaller worlds."""
    return round(350 * (size / 512) ** 1.5)


def frame_params(t, sea_level, tide_amp, season_amp):
    sea_eff = sea_level + tide_amp * np.sin(2 * np.pi * t / TIDE_PERIOD)
    season_off = season_amp * np.sin(2 * np.pi * t / YEAR_DAYS)
    sun_x = t % 1.0
    return sea_eff, season_off, sun_x


def render(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night):
    """Pure: (world slice, layer name, sim time, knobs) -> uint8 RGB image."""
    sea_eff, season_off, sun_x = frame_params(t, sea_level, tide_amp, season_amp)
    e = ws.elev

    if layer == "elevation":
        img = np.empty(e.shape + (3,), np.float32)
        sea_m = e < sea_eff
        f = np.clip(e / max(sea_eff, 1e-6), 0, 1)
        img[..., 0] = 16 + 70 * f
        img[..., 1] = 34 + 106 * f
        img[..., 2] = 78 + 108 * f
        g = np.clip((e - sea_eff) / max(1 - sea_eff, 1e-6), 0, 1)
        land_rgb = color_ramp(g, [0.0, 0.55, 1.0],
                              [(88, 140, 80), (168, 150, 96), (245, 245, 248)])
        img[~sea_m] = land_rgb[~sea_m]
    elif layer == "temperature":
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        img = color_ramp(tf, *TMP_RAMP)
    elif layer == "moisture":
        img = color_ramp(ws.moist, *MOI_RAMP)
    elif layer == "flow":
        img = color_ramp(ws.log_accum, *FLOW_RAMP)
    elif layer == "light":
        l = daylight_row(ws.xn, sun_x, day_night)[None, :]
        img = np.empty(e.shape + (3,), np.float32)
        img[..., 0] = 255 * l
        img[..., 1] = 248 * l
        img[..., 2] = 80 + 145 * l
    else:  # biome / composite
        tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
        ids = biome_ids(e, tf, ws.moist, sea_eff)
        img = BIOME_LUT[ids]
        if layer == "composite":
            land = e >= sea_eff
            img = img.copy()
            img[land] *= ws.shade[land, None]
            rivers = (ws.accum > river_thr) & land
            img[rivers] = (70, 130, 200)
            l = daylight_row(ws.xn, sun_x, day_night)[None, :]
            img[..., 0] *= l
            img[..., 1] *= l * 0.96 + 0.04
            img[..., 2] *= l * 0.82 + 0.18
    return np.clip(img, 0, 255).astype(np.uint8)


def render_rgba_bytes(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night):
    """Same as render(), packed as (width, RGBA bytes) for canvas blitting."""
    img = render(ws, layer, t, sea_level, river_thr, season_amp, tide_amp, day_night)
    h, w, _ = img.shape
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = img
    rgba[..., 3] = 255
    return w, rgba.tobytes()
