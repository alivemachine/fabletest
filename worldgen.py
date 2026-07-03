"""
worldgen.py  —  Milestone 0 + part of Milestone 1

The whole philosophy in one file: the world is a PURE FUNCTION of (seed, x, y).
Nothing here is stored. Everything is recomputed from the seed. Each function
below is one LAYER of the stack, and each layer only reads the layers above it:

    noise ->  elevation
    elevation + latitude ->  temperature
    noise ->  moisture
    (elevation, temperature, moisture) ->  biome        [pure lookup]
    elevation ->  flow direction (D8) ->  rivers & lakes  [deterministic water]

The noise is TILEABLE, so the map wraps seamlessly in both directions —
that is your "explore forever" requirement, for free.

Run:  python3 worldgen.py
Out:  world_<seed>.png
"""

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# LAYER 0 — NOISE.  Tileable fractal value-noise. Static, pure f(seed, x, y).
# ---------------------------------------------------------------------------

def _tileable_octave(size, period, rng):
    """One octave of value noise that wraps seamlessly (period divides size)."""
    # random values on a coarse lattice of shape (period, period)
    lattice = rng.random((period, period)).astype(np.float32)
    # bilinear upsample to (size, size) with WRAPAROUND indexing -> tiles
    coords = np.linspace(0, period, size, endpoint=False)
    i0 = np.floor(coords).astype(int) % period
    i1 = (i0 + 1) % period
    frac = (coords - np.floor(coords)).astype(np.float32)
    # smoothstep for softer interpolation
    frac = frac * frac * (3 - 2 * frac)
    # interpolate rows then cols
    top = lattice[i0][:, i0] * (1 - frac)[None, :] + lattice[i0][:, i1] * frac[None, :]
    bot = lattice[i1][:, i0] * (1 - frac)[None, :] + lattice[i1][:, i1] * frac[None, :]
    return top * (1 - frac)[:, None] + bot * frac[:, None]


def fractal_noise(size, seed, base_period=4, octaves=6):
    """Sum of tileable octaves -> fractal, seamless, deterministic in [0,1]."""
    rng = np.random.default_rng(seed)
    field = np.zeros((size, size), np.float32)
    amp, total, period = 1.0, 0.0, base_period
    for _ in range(octaves):
        if period > size:
            break
        field += amp * _tileable_octave(size, period, rng)
        total += amp
        amp *= 0.5
        period *= 2
    field /= total
    return (field - field.min()) / (field.max() - field.min())


# ---------------------------------------------------------------------------
# LAYER 1 — ELEVATION.  Domain-warped for less "blobby" coastlines.
# ---------------------------------------------------------------------------

def elevation_field(size, seed):
    warp = fractal_noise(size, seed + 99, base_period=3, octaves=4)
    base = fractal_noise(size, seed, base_period=3, octaves=7)
    # blend warp in to break up symmetry; keep it simple & tileable
    e = 0.75 * base + 0.25 * warp
    return (e - e.min()) / (e.max() - e.min())


# ---------------------------------------------------------------------------
# LAYER 2 — TEMPERATURE.  Vertical gradient (poles cold) + altitude cooling.
#           Seasons would be a slow horizontal offset added to this. (M2)
# ---------------------------------------------------------------------------

def temperature_field(size, elevation, sea_level):
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    latitude = 1 - np.abs(y - 0.5) * 2          # 1 at equator, 0 at poles
    latitude = np.repeat(latitude, size, axis=1)
    altitude_cooling = np.clip((elevation - sea_level), 0, 1) * 0.9
    t = latitude - altitude_cooling
    return np.clip(t, 0, 1)


# ---------------------------------------------------------------------------
# LAYER 2 — MOISTURE.  Its own noise field (rainfall advection is M1-next).
# ---------------------------------------------------------------------------

def moisture_field(size, seed):
    return fractal_noise(size, seed + 555, base_period=4, octaves=5)


# ---------------------------------------------------------------------------
# LAYER 3 — BIOME.  Pure lookup on (elevation, temperature, moisture).
# ---------------------------------------------------------------------------

# biome -> RGB
BIOME_COLORS = {
    "deep_ocean":  (30,  60, 120),
    "ocean":       (45,  85, 155),
    "shallow":     (70, 130, 180),
    "beach":       (210, 200, 150),
    "desert":      (222, 200, 120),
    "savanna":     (180, 190,  90),
    "grassland":   (120, 180,  90),
    "forest":      ( 60, 140,  70),
    "jungle":      ( 30, 110,  55),
    "taiga":       ( 90, 140, 110),
    "tundra":      (170, 180, 170),
    "snow":        (235, 240, 245),
    "mountain":    (130, 125, 120),
    "high_peak":   (200, 200, 205),
}

def classify_biomes(elevation, temperature, moisture, sea_level):
    size = elevation.shape[0]
    biome = np.empty((size, size), dtype=object)

    water = elevation < sea_level
    deep = elevation < sea_level - 0.12
    shallow = water & (elevation >= sea_level - 0.05)
    biome[water] = "ocean"
    biome[deep] = "deep_ocean"
    biome[shallow] = "shallow"

    land = ~water
    beach = land & (elevation < sea_level + 0.015)
    high = land & (elevation > 0.82)
    peak = land & (elevation > 0.92)
    mid = land & ~beach & ~high

    t, m = temperature, moisture
    # mid-elevation land: temperature x moisture table
    biome[mid & (t < 0.2)] = "snow"
    biome[mid & (t >= 0.2) & (t < 0.35)] = "tundra"
    biome[mid & (t >= 0.35) & (t < 0.5) & (m < 0.5)] = "taiga"
    biome[mid & (t >= 0.35) & (t < 0.5) & (m >= 0.5)] = "taiga"
    biome[mid & (t >= 0.5) & (m < 0.25)] = "desert"
    biome[mid & (t >= 0.5) & (m >= 0.25) & (m < 0.45)] = "savanna"
    biome[mid & (t >= 0.5) & (m >= 0.45) & (m < 0.65)] = "grassland"
    biome[mid & (t >= 0.5) & (m >= 0.65) & (m < 0.82)] = "forest"
    biome[mid & (t >= 0.5) & (m >= 0.82)] = "jungle"

    biome[beach] = "beach"
    biome[high] = "mountain"
    biome[peak] = "high_peak"
    # any land cell still unset -> grassland fallback
    unset = land & (biome == None)  # noqa: E711
    biome[unset] = "grassland"
    return biome


# ---------------------------------------------------------------------------
# LAYER 1 (water) — D8 FLOW.  Deterministic hydrology: flow direction from
#           each cell to its lowest neighbor, then flow ACCUMULATION.
#           High accumulation on land = a river. Wraps on the torus.
# ---------------------------------------------------------------------------

def compute_rivers(elevation, sea_level, river_threshold=350):
    size = elevation.shape[0]
    e = elevation
    # 8 neighbor offsets
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # For each cell, find lowest neighbor (wraparound = torus)
    flat = e.reshape(-1)
    n = size * size
    lowest = np.arange(n)                       # default: drains to self (sink)
    lowest_val = flat.copy()
    yy, xx = np.divmod(np.arange(n), size)
    for dy, dx in offsets:
        ny = (yy + dy) % size
        nx = (xx + dx) % size
        nidx = ny * size + nx
        nval = flat[nidx]                       # elevation AT neighbor (y+dy, x+dx)
        better = nval < lowest_val
        lowest = np.where(better, nidx, lowest)
        lowest_val = np.where(better, nval, lowest_val)

    # Flow accumulation: process cells from highest to lowest, push water down.
    order = np.argsort(-flat)                    # descending elevation
    accum = np.ones(n, dtype=np.int32)
    for idx in order:
        tgt = lowest[idx]
        if tgt != idx:
            accum[tgt] += accum[idx]

    accum = accum.reshape(size, size)
    rivers = (accum > river_threshold) & (e >= sea_level)
    return rivers, accum


# ---------------------------------------------------------------------------
# RENDER — turn the layers into a picture.
# ---------------------------------------------------------------------------

def render(seed=42, size=512, sea_level=0.42):
    elevation = elevation_field(size, seed)
    temperature = temperature_field(size, elevation, sea_level)
    moisture = moisture_field(size, seed)
    biome = classify_biomes(elevation, temperature, moisture, sea_level)
    rivers, _ = compute_rivers(elevation, sea_level)

    img = np.zeros((size, size, 3), dtype=np.uint8)
    for name, color in BIOME_COLORS.items():
        img[biome == name] = color

    # subtle hillshade so mountains read as 3D
    gy, gx = np.gradient(elevation)
    shade = np.clip(1 - (gx + gy) * 2.2, 0.75, 1.25)
    land = elevation >= sea_level
    img[land] = np.clip(img[land] * shade[land, None], 0, 255).astype(np.uint8)

    # paint rivers over the top
    img[rivers] = (70, 130, 200)

    Image.fromarray(img).save(f"world_{seed}.png")
    return f"world_{seed}.png"


if __name__ == "__main__":
    for s in (42, 7, 2024):
        path = render(seed=s)
        print("wrote", path)
