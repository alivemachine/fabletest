r"""Tag-driven texture pipeline — organization & management of generated sprites.

The problem: every tile's appearance depends on continuous world state
(temperature, season phase, time of day, vegetation, ecosystem health, zoom …),
and textures come from an image-gen API that is slow and costs money per image.
You cannot generate "a texture per tile" — you must generate "a texture per
DISTINCT APPEARANCE" and let every tile that looks the same share it.

So the pipeline is three collapses, in order:

  1. QUANTIZE   continuous state -> a small discrete TAG per axis
                (temperature 0.371 -> "mild"; season phase 0.61 -> "autumn").
                The infinite state space becomes finite and enumerable.
  2. CANONICAL  subject + relevant tags -> one deterministic KEY string.
                Axes irrelevant to a subject are dropped from its key (ocean
                doesn't care about growth), which is the main combinatorial
                control: the naive tag product is ~10^5-10^6 combos, but each
                subject only keys on its own axes, and only combos the camera
                actually VISITS are ever generated.
  3. DEDUP      a whole on-screen chunk bit-packs each tile's tags into one
                integer code; np.unique collapses ~9k tiles to typically a few
                dozen distinct keys per frame. Those keys — not tiles — are
                what the store caches and the backend generates.

Identity is deterministic all the way down (the world is a pure function, and
so is its skin): the same tile state always produces the same key, the same
key always produces the same prompt and generation seeds, and a tile picks
WHICH of a key's N variations it shows by hashing its world-lattice coords —
so revisiting a place shows the exact same trees, without storing anything
per tile.

The LOD ladder (signed, anchored at lod 0 = one object per screen tile):

    lod +4  group81   one sprite = a whole forest / town district
    lod +3  group27   dozens of trees / a hamlet
    lod +2  group9    ~9 trees / a cluster of houses
    lod +1  group3    a cluster of 3 trees / 3 houses     (aggregate upward x3)
    lod  0  single    one tree / one house per tile
    lod -1  obj2x2    one object spans 2x2 tiles          (magnify downward x2)
    lod -2  obj4x4    one object spans 4x4 tiles
    lod -3  obj8x8    one object spans 8x8 tiles, full detail

Above 0 a sprite REPRESENTS more objects; below 0 a single object COVERS more
tiles (buildings, trees, the player, river reaches — everything keeps a
representation at every zoom). Ground textures stay one-per-tile at every lod
(they just get close-up variants); props switch from "one instance per tile"
to an instance list anchored on the fixed lod-0 object lattice, so the same
oak stays the same oak while it grows from 1 to 64 tiles across.

Lifecycle of a key:  (miss) -> pending -> generating -> ready
                                      \-> failed (kept; retryable)
While a key is not ready the resolver degrades gracefully:
  exact ready -> nearest ready neighbor (weighted tag distance, same subject)
             -> deterministic procedural placeholder (instant, always works).
Ready assets can be EVICTED under a byte budget (LRU) with zero loss — the
key is deterministic, so anything evicted regenerates identically on demand.

Nothing in here knows what is on the other side of the image-gen API: the
`Backend` protocol takes (prompt, seed, size, n) and returns PNG bytes. A
ComfyUI/SDXL RunPod worker, a hosted API, or the built-in placeholder painter
are interchangeable. See TEXTURES.md for the full design.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import numpy as np

import world_core as wc
from worldgen import BIOME_COLORS

# ---------------------------------------------------------------------------
# The tag schema. Adding an axis = adding a row here (and a quantizer in
# derive()); keys stay canonical because order and vocabulary live in ONE place.
# `ordinal` axes measure fallback distance in steps ("cold" is nearer "freezing"
# than "hot"); categorical axes are hit-or-miss. `weight` is how bad a mismatch
# is when the resolver substitutes a neighbor (see _tag_distance).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Axis:
    name: str
    values: tuple
    weight: float
    ordinal: bool


LOD_MIN, LOD_MAX = -3, 4
LOD_NAMES = ("obj8x8", "obj4x4", "obj2x2", "single",
             "group3", "group9", "group27", "group81")   # index = lod - LOD_MIN
TILE0_WORLD = 1.0 / 2048.0    # world span of a lod-0 tile (one object per tile)

AXES = (
    Axis("lod",     LOD_NAMES,                                          4.0, True),
    Axis("season",  ("spring", "summer", "autumn", "winter"),           1.0, False),
    Axis("tod",     ("night", "dawn", "day", "dusk"),                   0.5, False),
    Axis("temp",    ("freezing", "cold", "mild", "warm", "hot"),        1.0, True),
    Axis("wet",     ("arid", "dry", "damp", "wet"),                     0.5, True),
    Axis("growth",  ("bare", "sprout", "young", "mature", "lush"),      1.5, True),
    Axis("cond",    ("pristine", "stressed", "withered", "scorched"),   2.0, True),
    Axis("density", ("sparse", "patchy", "dense"),                      1.0, True),
)
AXIS_INDEX = {a.name: i for i, a in enumerate(AXES)}
# bits per axis: values + 1 (encoded level 0 = "axis not relevant to subject")
AXIS_BITS = tuple(int(np.ceil(np.log2(len(a.values) + 1))) for a in AXES)

# ---------------------------------------------------------------------------
# Subjects — the closed vocabulary of things a texture can depict. Grounds are
# generated 1:1 from the biome table so new sub-biomes appear here for free.
# `axes` is the subject's RELEVANCE MASK: only these axes enter its key.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Subject:
    name: str
    layer: str            # "ground" | "prop"
    axes: frozenset
    color: tuple          # placeholder-painter base RGB
    phrase: str           # prompt noun phrase


_WATER_BIOMES = {"deep_ocean", "ocean", "shallow"}
_GROUND_AXES = frozenset({"lod", "season", "tod", "temp", "wet", "cond"})
_WATER_AXES = frozenset({"lod", "tod", "temp"})
_VEG_AXES = frozenset({"lod", "density", "season", "tod", "temp", "growth", "cond"})
_ROCK_AXES = frozenset({"lod", "density", "season", "tod", "temp"})
_HOUSE_AXES = frozenset({"lod", "density", "season", "tod", "cond"})

_PROP_DEFS = [
    # name          axes         color            phrase
    ("none",      frozenset(),  (0, 0, 0),       ""),
    ("tree.oak",  _VEG_AXES,    (52, 128, 58),   "oak tree with a round leafy canopy"),
    ("tree.dark_oak", _VEG_AXES, (30, 84, 40),   "dense old-growth dark oak tree"),
    ("tree.pine", _VEG_AXES,    (44, 100, 82),   "tall conifer pine tree"),
    ("tree.jungle", _VEG_AXES,  (28, 118, 52),   "lush tropical jungle tree with vines"),
    ("tree.palm", _VEG_AXES,    (88, 158, 74),   "palm tree with arched fronds"),
    ("tree.acacia", _VEG_AXES,  (128, 138, 62),  "flat-topped acacia tree"),
    ("tree.dead", _VEG_AXES,    (86, 70, 58),    "dead burned tree, bare charred branches"),
    ("shrub",     _VEG_AXES,    (96, 128, 76),   "low hardy shrub"),
    ("cactus",    _ROCK_AXES,   (96, 150, 88),   "desert cactus"),
    ("rock",      _ROCK_AXES,   (128, 124, 118), "weathered boulder"),
    ("house",     _HOUSE_AXES,  (176, 128, 88),  "small rustic house with a pitched roof"),
    ("road",      frozenset({"lod", "season", "tod", "wet"}),
                                (150, 134, 108), "dirt road, wheel ruts"),
    # not placed by derive() — the same key/prompt/store path serves any
    # consumer-side subject (the player, NPCs, boats …) that asks for it
    ("player",    frozenset({"lod", "season", "tod"}),
                                (222, 196, 160), "the player character, a lone traveler"),
]

SUBJECTS: list[Subject] = []
for _b in wc.BIOME_NAMES:
    SUBJECTS.append(Subject(
        f"ground.{_b}", "ground",
        _WATER_AXES if _b in _WATER_BIOMES else _GROUND_AXES,
        BIOME_COLORS[_b],
        f"seamless {_b.replace('_', ' ')} ground terrain tile"))
SUBJECTS.append(Subject("ground.river", "ground", _GROUND_AXES,
                        (70, 130, 180), "seamless river water ground tile"))
for _n, _a, _c, _p in _PROP_DEFS:
    SUBJECTS.append(Subject(_n, "prop", _a, _c, _p))

SUBJ_INDEX = {s.name: i for i, s in enumerate(SUBJECTS)}
_SUBJ_BITS = int(np.ceil(np.log2(len(SUBJECTS))))
# relevance matrix [n_subjects, n_axes]
_REL = np.zeros((len(SUBJECTS), len(AXES)), bool)
for _i, _s in enumerate(SUBJECTS):
    for _ax in _s.axes:
        _REL[_i, AXIS_INDEX[_ax]] = True


# ---------------------------------------------------------------------------
# Descriptor & canonical key
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Descriptor:
    """One distinct appearance: a subject plus its relevant, quantized tags."""
    subject: str
    tags: tuple                # ((axis, value), ...) in schema order

    @property
    def key(self) -> str:
        return "|".join([self.subject] + [f"{a}={v}" for a, v in self.tags])

    @property
    def key_hash(self) -> str:
        return hashlib.sha1(self.key.encode()).hexdigest()[:16]

    @property
    def lod(self) -> int:
        for a, v in self.tags:
            if a == "lod":
                return LOD_NAMES.index(v) + LOD_MIN
        return 0

    @property
    def footprint(self) -> int:
        """Tiles per side this sprite covers on screen (negative lods only)."""
        return 2 ** max(0, -self.lod)

    def tag_dict(self) -> dict:
        return dict(self.tags)


def descriptor(subject: str, **tags) -> Descriptor:
    """Build a Descriptor by hand (consumer-side subjects like `player`).
    Irrelevant/unknown axes are dropped; missing relevant axes default to the
    axis's first value, so a partial spec still yields a canonical key."""
    s = SUBJECTS[SUBJ_INDEX[subject]]
    lod_name = tags.get("lod", LOD_NAMES[-LOD_MIN])
    single_or_below = LOD_NAMES.index(lod_name) + LOD_MIN <= 0
    out = []
    for ax in AXES:
        if ax.name not in s.axes:
            continue
        if ax.name == "density" and single_or_below:
            continue                     # same rule as _pack: no group density
        v = tags.get(ax.name, ax.values[0])
        if v not in ax.values:
            raise ValueError(f"{ax.name}={v!r} not in {ax.values}")
        out.append((ax.name, v))
    return Descriptor(subject, tuple(out))


def _decode(code: int) -> Descriptor:
    sid = code & ((1 << _SUBJ_BITS) - 1)
    code >>= _SUBJ_BITS
    tags = []
    for ax, bits in zip(AXES, AXIS_BITS):
        lv = code & ((1 << bits) - 1)
        code >>= bits
        if lv:                          # 0 = axis not relevant to this subject
            tags.append((ax.name, ax.values[lv - 1]))
    return Descriptor(SUBJECTS[sid].name, tuple(tags))


def _pack(subj_id, levels, lod):
    """Bit-pack subject id + per-axis levels into one int64 code per tile.
    Levels are stored +1 so 0 always means "not relevant"; relevance comes
    from the subject's mask (with `density` masked out at lod <= 0, where a
    single object has no group density)."""
    rel = _REL.copy()
    if lod <= 0:
        rel[:, AXIS_INDEX["density"]] = False
    code = subj_id.astype(np.int64)
    shift = _SUBJ_BITS
    for i, (ax, bits) in enumerate(zip(AXES, AXIS_BITS)):
        enc = np.where(rel[subj_id, i], levels[i].astype(np.int64) + 1, 0)
        code |= enc << shift
        shift += bits
    return code


# ---------------------------------------------------------------------------
# Quantizers — continuous state -> axis level indices
# ---------------------------------------------------------------------------

def lod_for_tile_world(tile_world: float) -> int:
    """Signed LOD from the on-screen tile's world size. Groups aggregate by
    x3 per step above lod 0; objects magnify by x2 per step below it."""
    r = tile_world / TILE0_WORLD
    if r >= 1.0:
        lod = int(round(np.log(r) / np.log(3.0)))
    else:
        lod = -int(round(np.log(1.0 / r) / np.log(2.0)))
    return int(np.clip(lod, LOD_MIN, LOD_MAX))


def _tod_level(sun_x: float) -> int:
    f = sun_x % 1.0                              # 0 midnight, .5 noon
    if f < 0.18 or f >= 0.82:
        return 0                                 # night
    if f < 0.32:
        return 1                                 # dawn
    if f < 0.68:
        return 2                                 # day
    return 3                                     # dusk


def _season_levels(t: float, lat_signed: np.ndarray) -> np.ndarray:
    """Per-tile season index; the southern hemisphere is half a year out of
    phase (same convention as temperature_t's `season_off * lat_signed`)."""
    ph = (t % wc.YEAR_DAYS) / wc.YEAR_DAYS       # sin peak (N summer) at .25
    phase = np.where(lat_signed >= 0, ph, (ph + 0.5) % 1.0)
    return np.floor(((phase + 0.125) % 1.0) * 4).astype(np.uint8)  # peak mid-summer


def _lattice_hash(ws, cell_world: float, salt: int = 0):
    """Deterministic uniform [0,1) per world-lattice cell of size `cell_world`.
    Keyed on WORLD coords (not screen coords) so panning/zooming never
    reshuffles which cells hold a rock or which variation a tree shows."""
    tw = ws.span / max(ws.size, 1)
    x0 = ws.cx - ws.span / 2.0
    y0 = ws.cy - ws.span / 2.0
    xs = np.floor((x0 + (np.arange(ws.size) + 0.5) * tw) / cell_world).astype(np.int64)
    ys = np.floor((y0 + (np.arange(ws.size) + 0.5) * tw) / cell_world).astype(np.int64)
    ix, iy = np.meshgrid(xs, ys)
    h = (ix * 73856093) ^ (iy * 19349663) ^ np.int64((ws.seed + salt) * 83492791)
    return ((h & 0x7FFFFFFF).astype(np.float64) / 0x7FFFFFFF), ix, iy


def variation_grid(ws, n_variations: int, lod: int = 0) -> np.ndarray:
    """Which of a key's N variations each tile shows — a pure hash of the
    tile's lod-lattice cell, so the same tree at the same place is always the
    same sprite, at every visit, with nothing stored."""
    cell = max(TILE0_WORLD * (3 ** max(lod, 0)), ws.span / max(ws.size, 1))
    h, _, _ = _lattice_hash(ws, cell, salt=7)
    return (h * n_variations).astype(np.int32) % max(n_variations, 1)


# ---------------------------------------------------------------------------
# derive() — a chunk's state -> descriptor field (the link tile <-> texture)
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """One placed prop sprite: window-tile anchor + footprint in tiles."""
    i: int                 # column of the anchor (top-left) tile
    j: int                 # row
    code: int
    variation: int
    footprint: int = 1


@dataclass
class DescriptorField:
    """Everything the renderer needs to skin one chunk: a per-tile ground
    code grid, a prop instance list, and the legend decoding codes -> keys."""
    lod: int
    ground: np.ndarray                    # (n,n) int64 codes
    props: list                           # [Instance]
    legend: dict = field(default_factory=dict)   # code -> Descriptor

    def descriptors(self):
        return list(self.legend.values())


_TREE_OF_BIOME = {
    "forest": "tree.oak", "glade": "tree.oak", "dark_forest": "tree.dark_oak",
    "taiga": "tree.pine", "jungle": "tree.jungle", "jungle_clear": "tree.jungle",
    "savanna": "tree.acacia", "acacia_scrub": "tree.acacia",
    "oasis": "tree.palm", "beach": "tree.palm",
    "grassland": "tree.oak", "meadow": "tree.oak", "tall_grass": "tree.oak",
    "wheat_soil": "tree.oak",
}
_TREE_VEG_MIN = {
    "forest": 0.22, "glade": 0.22, "dark_forest": 0.20, "taiga": 0.22,
    "jungle": 0.20, "jungle_clear": 0.22, "savanna": 0.38, "acacia_scrub": 0.34,
    "oasis": 0.24, "beach": 0.30, "grassland": 0.55, "meadow": 0.50,
    "tall_grass": 0.55, "wheat_soil": 0.60,
}
_SHRUB_BIOMES = ("tundra", "shrub_steppe", "rocky_tundra")
_CACTUS_BIOMES = ("desert", "reg_rock", "dunes")
_ROCK_BIOMES = ("mountain", "scree", "high_peak", "rocky_tundra")


def _river_mask(ws, st):
    river_alpha = getattr(ws, "river_alpha", None)
    if river_alpha is None:
        return np.zeros_like(ws.elev, bool)
    brook_alpha = getattr(ws, "brook_alpha", np.zeros_like(ws.elev))
    gate = np.clip(ws.river_disc / max(float(st["river_thr"]), 1.0), 0, 1) ** 0.6
    return np.maximum(river_alpha * gate, brook_alpha) > 0.35


def derive(ws, st, n_variations: int = 3) -> DescriptorField:
    """Quantize a chunk's per-tile state into its descriptor field.

    Every input is already in `st` / `ws` (the same payload the Godot bridge
    streams), so this is pure array math: ~10 vectorized quantizers, one
    bit-pack, one np.unique. A 96x96 chunk collapses to a few dozen keys."""
    n = ws.size
    tw = ws.span / max(n, 1)
    lod = lod_for_tile_world(tw)
    bid = st["biome_id"]
    veg_live = np.clip(st["veg"] * st["veg_health"], 0, 1)
    land = st["e"] >= st["sea_eff"]
    river = _river_mask(ws, st) & land

    # --- per-axis levels (uint8 grids; scalar axes broadcast) ------------
    lv = [None] * len(AXES)
    lv[AXIS_INDEX["lod"]] = np.full((n, n), lod - LOD_MIN, np.uint8)
    lv[AXIS_INDEX["season"]] = _season_levels(st["t"], ws.lat_signed)
    lv[AXIS_INDEX["tod"]] = np.full((n, n), _tod_level(st["sun_x"]), np.uint8)
    lv[AXIS_INDEX["temp"]] = np.digitize(st["tf"], (0.16, 0.32, 0.55, 0.75)).astype(np.uint8)
    lv[AXIS_INDEX["wet"]] = np.digitize(st["moist"], (0.25, 0.45, 0.65)).astype(np.uint8)
    lv[AXIS_INDEX["growth"]] = np.digitize(veg_live, (0.15, 0.35, 0.60, 0.85)).astype(np.uint8)
    lv[AXIS_INDEX["cond"]] = np.select(
        [st["scorch"] > 0.45, st["veg_health"] < 0.35, st["veg_health"] < 0.80],
        [3, 2, 1], 0).astype(np.uint8)
    lv[AXIS_INDEX["density"]] = np.digitize(veg_live, (0.35, 0.65)).astype(np.uint8)

    # --- ground subject: biome 1:1, river overrides on land --------------
    biome_to_subj = np.array([SUBJ_INDEX[f"ground.{b}"] for b in wc.BIOME_NAMES],
                             np.int32)
    g_subj = biome_to_subj[np.clip(bid, 0, len(wc.BIOME_NAMES) - 1)]
    g_subj = np.where(river, SUBJ_INDEX["ground.river"], g_subj)
    ground_codes = _pack(g_subj, lv, lod)

    # --- prop subject per tile -------------------------------------------
    def _isin(names):
        return np.isin(bid, [wc.BID[b] for b in names])

    p_subj = np.zeros((n, n), np.int32)          # 0 = SUBJ_INDEX["none"]... map later
    none_id = SUBJ_INDEX["none"]
    p_subj[:] = none_id
    for b, tree in _TREE_OF_BIOME.items():
        m = (bid == wc.BID[b]) & (veg_live >= _TREE_VEG_MIN[b])
        p_subj[m] = SUBJ_INDEX[tree]
    p_subj[_isin(_SHRUB_BIOMES) & (veg_live >= 0.15)] = SUBJ_INDEX["shrub"]
    coin, _, _ = _lattice_hash(ws, max(tw, TILE0_WORLD), salt=3)
    p_subj[_isin(_CACTUS_BIOMES) & (coin < 0.10)] = SUBJ_INDEX["cactus"]
    p_subj[_isin(_ROCK_BIOMES) & (coin < 0.18)] = SUBJ_INDEX["rock"]
    is_tree = np.isin(p_subj, [SUBJ_INDEX[s] for s in
                               ("tree.oak", "tree.dark_oak", "tree.pine",
                                "tree.jungle", "tree.palm", "tree.acacia", "shrub")])
    p_subj[is_tree & (st["scorch"] > 0.45)] = SUBJ_INDEX["tree.dead"]
    # settlements (the M4 expand) override vegetation
    settle_a, _rgb = wc._settlements(ws, st["t"], float(st["sea_level"]))
    p_subj[settle_a >= 0.4] = SUBJ_INDEX["road"]
    p_subj[settle_a >= 0.95] = SUBJ_INDEX["house"]
    p_subj[~land | river] = none_id
    prop_codes = _pack(p_subj, lv, lod)
    prop_codes[p_subj == none_id] = 0            # 0 = no prop, never a legend key

    # --- prop instances ----------------------------------------------------
    var = variation_grid(ws, n_variations, lod)
    props: list[Instance] = []
    if lod >= 0:
        jj, ii = np.nonzero(prop_codes)
        for j, i in zip(jj.tolist(), ii.tolist()):
            props.append(Instance(i, j, int(prop_codes[j, i]), int(var[j, i]), 1))
    else:
        # below lod 0 an object outgrows its tile: props live on the FIXED
        # lod-0 object lattice, each instance anchored at the cell's top-left
        # window tile and spanning footprint x footprint tiles. The state is
        # sampled once per cell (at its min-index tile) so the whole building
        # is one sprite, not footprint^2 disagreeing tiles.
        fp = 2 ** (-lod)
        _, ix, iy = _lattice_hash(ws, TILE0_WORLD)
        cell = (iy - iy.min()) * np.int64(1 << 32) + (ix - ix.min())
        uniq, inv = np.unique(cell, return_inverse=True)
        flat = inv.reshape(n, n)
        k = len(uniq)
        min_i = np.full(k, n, np.int64); min_j = np.full(k, n, np.int64)
        cols = np.broadcast_to(np.arange(n), (n, n))
        rows = np.broadcast_to(np.arange(n)[:, None], (n, n))
        np.minimum.at(min_i, flat.ravel(), cols.ravel())
        np.minimum.at(min_j, flat.ravel(), rows.ravel())
        for c in range(k):
            j0, i0 = int(min_j[c]), int(min_i[c])
            # representative tile: the cell center inside the window
            jc = min(j0 + fp // 2, n - 1); ic = min(i0 + fp // 2, n - 1)
            if flat[jc, ic] != c:
                jc, ic = j0, i0
            code = int(prop_codes[jc, ic])
            if code:
                props.append(Instance(i0, j0, code, int(var[jc, ic]), fp))

    # --- legend -------------------------------------------------------------
    legend = {}
    for code in np.unique(ground_codes).tolist():
        legend[code] = _decode(code)
    for inst in props:
        if inst.code not in legend:
            legend[inst.code] = _decode(inst.code)
    return DescriptorField(lod, ground_codes, props, legend)


# ---------------------------------------------------------------------------
# Prompt builder — a Descriptor -> the text the image model sees. This is the
# ONLY place that knows about prompts; swapping art direction = editing tables.
# ---------------------------------------------------------------------------

STYLE = ("isometric pixel art, 16-bit retro game sprite, SNES style, "
         "clean silhouette, limited palette")
NEGATIVE = ("photo, realistic, 3d render, blurry, text, watermark, frame, "
            "gradient background")

_TAG_PHRASES = {
    ("season", "spring"): "early spring, fresh green buds",
    ("season", "summer"): "midsummer, full foliage",
    ("season", "autumn"): "autumn, orange and brown leaves",
    ("season", "winter"): "winter, snow-dusted",
    ("tod", "night"): "night, cool moonlight",
    ("tod", "dawn"): "dawn, warm low light",
    ("tod", "day"): "bright daylight",
    ("tod", "dusk"): "dusk, golden hour light",
    ("temp", "freezing"): "frozen, icy",
    ("temp", "cold"): "cold climate",
    ("temp", "mild"): "",
    ("temp", "warm"): "warm climate",
    ("temp", "hot"): "hot, sun-baked",
    ("wet", "arid"): "parched dry ground",
    ("wet", "dry"): "dry",
    ("wet", "damp"): "",
    ("wet", "wet"): "lush and damp",
    ("growth", "bare"): "barren",
    ("growth", "sprout"): "young sapling stage",
    ("growth", "young"): "young, half grown",
    ("growth", "mature"): "fully grown",
    ("growth", "lush"): "old and thriving, overgrown",
    ("cond", "pristine"): "",
    ("cond", "stressed"): "slightly wilted",
    ("cond", "withered"): "withered and dying",
    ("cond", "scorched"): "burned and charred",
    ("density", "sparse"): "sparsely scattered",
    ("density", "patchy"): "loosely grouped",
    ("density", "dense"): "densely packed",
}

_LOD_PHRASES = {
    "obj8x8": "extreme close-up, large highly detailed sprite of one",
    "obj4x4": "close-up, large detailed sprite of one",
    "obj2x2": "detailed sprite of one",
    "single": "one",
    "group3": "a small cluster of three",
    "group9": "a group of about nine",
    "group27": "a dense patch of dozens of",
    "group81": "a vast expanse of countless",
}


def build_prompt(desc: Descriptor) -> tuple:
    s = SUBJECTS[SUBJ_INDEX[desc.subject]]
    tags = desc.tag_dict()
    bits = [STYLE]
    if s.layer == "ground":
        bits.append(f"{s.phrase}, top-down tileable texture")
    else:
        noun = s.phrase + ("s" if tags.get("lod", "single").startswith("group")
                           else "")
        bits.append(f"{_LOD_PHRASES.get(tags.get('lod', 'single'), 'one')} {noun}")
    for a, v in desc.tags:
        if a == "lod":
            continue
        p = _TAG_PHRASES.get((a, v), "")
        if p:
            bits.append(p)
    if s.layer == "prop":
        bits.append("isolated on transparent background")
    return ", ".join(b for b in bits if b), NEGATIVE


# ---------------------------------------------------------------------------
# Backends — the pluggable far side of the pipeline.
# ---------------------------------------------------------------------------

@dataclass
class GenJob:
    key: str
    prompt: str
    negative: str
    seed: int              # deterministic: derived from the key
    px: int                # output size (scales with |negative lod|)
    n: int                 # how many variations to produce in one call


class Backend:
    """Protocol: generate(job) -> list of PNG bytes, len == job.n.
    Variation i MUST use seed job.seed + i so regeneration is reproducible."""
    name = "abstract"

    def generate(self, job: GenJob) -> list:
        raise NotImplementedError


def _paint_placeholder(desc: Descriptor, px: int, seed: int):
    """Deterministic procedural stand-in, tinted by the actual tags — you can
    SEE season/night/scorch working before any real model is connected."""
    from PIL import Image, ImageDraw
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    s = SUBJECTS[SUBJ_INDEX[desc.subject]]
    tags = desc.tag_dict()
    r, g, b = [float(c) for c in s.color]
    season = tags.get("season")
    if season == "winter":
        r, g, b = r * 0.5 + 128, g * 0.5 + 128, b * 0.5 + 132
    elif season == "autumn":
        r, g, b = min(255, r * 1.25 + 30), g * 0.85, b * 0.6
    elif season == "spring":
        g = min(255, g * 1.15 + 12)
    cond = tags.get("cond")
    if cond == "scorched":
        r, g, b = 60 + r * 0.15, 48 + g * 0.1, 40 + b * 0.1
    elif cond == "withered":
        r, g, b = r * 0.8 + 30, g * 0.7 + 20, b * 0.6
    tod = tags.get("tod")
    dim = {"night": 0.35, "dawn": 0.75, "dusk": 0.7}.get(tod, 1.0)
    tint_b = 1.25 if tod == "night" else 1.0
    col = (int(np.clip(r * dim, 0, 255)), int(np.clip(g * dim, 0, 255)),
           int(np.clip(b * dim * tint_b, 0, 255)), 255)

    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if s.layer == "ground":
        # iso diamond + hash speckle
        d.polygon([(px // 2, 0), (px - 1, px // 2), (px // 2, px - 1),
                   (0, px // 2)], fill=col, outline=(0, 0, 0, 90))
        for _ in range(px):
            x, y = rng.randint(px // 4, 3 * px // 4, 2)
            f = 0.8 + 0.4 * rng.rand()
            d.point((x, y), fill=(int(col[0] * f) & 255, int(col[1] * f) & 255,
                                  int(col[2] * f) & 255, 255))
    else:
        lod = desc.lod
        count = {0: 1, 1: 3, 2: 7, 3: 14, 4: 24}.get(max(lod, 0), 1)
        grow = {"bare": 0.4, "sprout": 0.5, "young": 0.7,
                "mature": 0.9, "lush": 1.0}.get(tags.get("growth"), 0.9)
        base_r = px * 0.30 * grow / (1.0 + 0.45 * max(lod, 0))
        for c in range(count):
            cx = px / 2 if count == 1 else rng.uniform(px * 0.2, px * 0.8)
            cy = px * 0.55 if count == 1 else rng.uniform(px * 0.35, px * 0.8)
            rad = base_r * (0.75 + 0.5 * rng.rand())
            if desc.subject == "house":
                d.rectangle([cx - rad, cy - rad * 0.7, cx + rad, cy + rad * 0.7],
                            fill=col, outline=(0, 0, 0, 120))
                d.polygon([(cx - rad * 1.1, cy - rad * 0.7),
                           (cx + rad * 1.1, cy - rad * 0.7),
                           (cx, cy - rad * 1.6)],
                          fill=(int(col[0] * 0.7), int(col[1] * 0.5),
                                int(col[2] * 0.45), 255))
            elif desc.subject == "rock":
                d.polygon([(cx - rad, cy + rad * 0.6), (cx, cy - rad),
                           (cx + rad, cy + rad * 0.6)], fill=col,
                          outline=(0, 0, 0, 120))
            else:
                trunk = (int(90 * dim), int(64 * dim), int(40 * dim), 255)
                d.rectangle([cx - rad * 0.12, cy, cx + rad * 0.12,
                             cy + rad * 0.9], fill=trunk)
                d.ellipse([cx - rad, cy - rad * 1.4, cx + rad, cy + rad * 0.2],
                          fill=col, outline=(0, 0, 0, 90))
    return img


class PlaceholderBackend(Backend):
    """Instant, free, deterministic — the fallback painter promoted to a
    backend, so the whole pipeline runs end-to-end with no API attached."""
    name = "placeholder"

    def generate(self, job: GenJob) -> list:
        import io
        desc = _desc_from_key(job.key)
        out = []
        for i in range(job.n):
            buf = io.BytesIO()
            _paint_placeholder(desc, job.px, job.seed + i).save(buf, "PNG")
            out.append(buf.getvalue())
        return out


class RestBackend(Backend):
    """Generic adapter for "any image gen api": POSTs one JSON job, expects
    base64 PNGs back. Point it at a thin shim in front of ComfyUI/SDXL on a
    RunPod (or any hosted API). Contract documented in TEXTURES.md.

        request:  {"prompt", "negative_prompt", "seed", "width", "height",
                   "num_images", "key"}
        response: {"images": ["<base64 png>", ...]}
    """
    name = "rest"

    def __init__(self, url: str, timeout: float = 120.0, headers: dict | None = None):
        self.url, self.timeout = url, timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def generate(self, job: GenJob) -> list:
        body = json.dumps({
            "prompt": job.prompt, "negative_prompt": job.negative,
            "seed": job.seed, "width": job.px, "height": job.px,
            "num_images": job.n, "key": job.key,
        }).encode()
        req = urllib.request.Request(self.url, data=body, headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        images = [base64.b64decode(im) for im in data["images"]]
        if len(images) != job.n:
            raise RuntimeError(f"backend returned {len(images)} images, wanted {job.n}")
        return images


RUNPOD_DEFAULT_PROMPT = (
    "isometric stylized setting, tiny fantasy village on a cliff, tile-game "
    "environment, soft sunlight, clean shapes, SDXL, high detail"
)
RUNPOD_DEFAULT_LORA_PAGE_URL = (
    "https://civitai.com/models/118775/stylized-setting-isometric-sdxl-and-sd15"
)
RUNPOD_DEFAULT_LORA_DIR = "/workspace/ComfyUI/models/loras/"
RUNPOD_DEFAULT_SDXL_CKPT = "sd_xl_base_1.0.safetensors"


def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    return int(v) if v is not None else int(default)


def _env_float(name: str, default: float) -> float:
    v = _env_str(name)
    return float(v) if v is not None else float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env_str(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def civitai_lora_download_url(url: str = RUNPOD_DEFAULT_LORA_PAGE_URL) -> str:
    """Normalize a Civitai model page URL to a downloadable endpoint."""
    m = re.match(r"^https://civitai\.com/models/(\d+)(?:[/?].*)?$", url.strip())
    if m:
        return f"https://civitai.com/api/download/models/{m.group(1)}"
    return url


def download_civitai_lora(
    *,
    source_url: str = RUNPOD_DEFAULT_LORA_PAGE_URL,
    dest_dir: str = RUNPOD_DEFAULT_LORA_DIR,
    filename: str = "stylized-setting-isometric-sdxl-and-sd15.safetensors",
    timeout: float = 120.0,
    token: str | None = None,
    dry_run: bool = False,
) -> str:
    """Download the LoRA safetensors into `dest_dir` and return its full path."""
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, filename)
    if dry_run:
        return out_path
    url = civitai_lora_download_url(source_url)
    headers = {"User-Agent": "fabletest-texgen/1.0"}
    if token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={urllib.parse.quote(token)}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if b"<html" in data[:512].lower():
        raise RuntimeError(
            "LoRA download returned HTML instead of safetensors; check Civitai URL/token."
        )
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def build_runpod_comfyui_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    checkpoint_name: str = RUNPOD_DEFAULT_SDXL_CKPT,
    lora_name: str = "stylized-setting-isometric-sdxl-and-sd15.safetensors",
    lora_strength_model: float = 0.8,
    lora_strength_clip: float = 0.8,
    steps: int = 30,
    cfg: float = 7.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    denoise: float = 1.0,
    filename_prefix: str = "fabletest",
    batch_size: int = 1,
) -> dict:
    """Build a ComfyUI SDXL workflow JSON compatible with RunPod worker-comfyui."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint_name}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "model": ["1", 0], "clip": ["1", 1],
            "lora_name": lora_name,
            "strength_model": float(lora_strength_model),
            "strength_clip": float(lora_strength_clip),
        }},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative_prompt, "clip": ["2", 1]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": int(width), "height": int(height),
                         "batch_size": int(batch_size)}},
        "6": {"class_type": "KSampler", "inputs": {
            "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
            "sampler_name": sampler_name, "scheduler": scheduler,
            "denoise": float(denoise), "model": ["2", 0],
            "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["5", 0],
        }},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
    }


def build_runpod_runsync_payload(workflow: dict) -> dict:
    return {"input": {"workflow": workflow}}


class RunPodComfyUIBackend(Backend):
    """RunPod runsync backend for worker-comfyui SDXL custom workflows."""
    name = "runpod-comfyui"

    def __init__(
        self,
        endpoint_id: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        retry_backoff: float | None = None,
        checkpoint_name: str | None = None,
        lora_name: str | None = None,
        lora_path: str | None = None,
        lora_strength_model: float = 0.8,
        lora_strength_clip: float = 0.8,
        prompt_prefix: str | None = None,
        dry_run: bool | None = None,
    ):
        self.endpoint_id = endpoint_id or _env_str("RUNPOD_ENDPOINT_ID")
        self.api_key = api_key or _env_str("RUNPOD_API_KEY")
        self.timeout = float(timeout if timeout is not None else _env_float("RUNPOD_TIMEOUT_SEC", 120.0))
        self.retries = int(retries if retries is not None else _env_int("RUNPOD_RETRIES", 2))
        self.retry_backoff = float(
            retry_backoff if retry_backoff is not None else _env_float("RUNPOD_RETRY_BACKOFF_SEC", 1.5)
        )
        self.checkpoint_name = checkpoint_name or _env_str("RUNPOD_SDXL_CHECKPOINT", RUNPOD_DEFAULT_SDXL_CKPT)
        self.lora_name = lora_name or _env_str(
            "RUNPOD_LORA_NAME", "stylized-setting-isometric-sdxl-and-sd15.safetensors"
        )
        self.lora_path = lora_path or _env_str("RUNPOD_LORA_PATH", RUNPOD_DEFAULT_LORA_DIR)
        self.lora_strength_model = float(_env_float("RUNPOD_LORA_STRENGTH_MODEL", lora_strength_model))
        self.lora_strength_clip = float(_env_float("RUNPOD_LORA_STRENGTH_CLIP", lora_strength_clip))
        self.prompt_prefix = prompt_prefix or _env_str("RUNPOD_PROMPT_PREFIX", RUNPOD_DEFAULT_PROMPT)
        self.dry_run = bool(_env_bool("RUNPOD_DRY_RUN", False) if dry_run is None else dry_run)
        self.filename_prefix = _env_str("RUNPOD_FILENAME_PREFIX", "fabletest")
        self.url = (f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
                    if self.endpoint_id else None)
        if not self.checkpoint_name:
            raise ValueError("RUNPOD_SDXL_CHECKPOINT must be configured for SDXL.")
        if self.lora_path and not self.lora_path.endswith("/"):
            self.lora_path += "/"
        self.full_lora_name = (
            f"{self.lora_path}{self.lora_name}" if self.lora_path and "/" not in self.lora_name else self.lora_name
        )

    def build_payload(self, job: GenJob) -> dict:
        prompt = f"{self.prompt_prefix}, {job.prompt}" if self.prompt_prefix else job.prompt
        workflow = build_runpod_comfyui_workflow(
            prompt=prompt,
            negative_prompt=job.negative,
            seed=job.seed,
            width=job.px,
            height=job.px,
            checkpoint_name=self.checkpoint_name,
            lora_name=self.full_lora_name,
            lora_strength_model=self.lora_strength_model,
            lora_strength_clip=self.lora_strength_clip,
            filename_prefix=self.filename_prefix,
            batch_size=job.n,
        )
        return build_runpod_runsync_payload(workflow)

    def _decode_images(self, data: dict, want: int) -> list:
        output = data.get("output", {})
        imgs = output.get("images", output if isinstance(output, list) else [])
        out = []
        for item in imgs:
            if isinstance(item, str):
                b64 = item.split(",", 1)[-1]
                out.append(base64.b64decode(b64))
            elif isinstance(item, dict):
                b64 = item.get("image") or item.get("base64")
                if b64:
                    b64 = b64.split(",", 1)[-1]
                    out.append(base64.b64decode(b64))
        if len(out) != want:
            raise RuntimeError(f"runpod returned {len(out)} images, wanted {want}: {data.get('status')}")
        return out

    def generate(self, job: GenJob) -> list:
        if self.dry_run:
            return PlaceholderBackend().generate(job)
        if not self.url or not self.api_key:
            raise ValueError("RunPod backend requires RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY (or dry-run mode).")
        body = json.dumps(self.build_payload(job)).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                return self._decode_images(data, job.n)
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                    KeyError, RuntimeError, ValueError) as e:
                last_err = e
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_backoff * (2 ** attempt))
        raise RuntimeError(f"runpod runsync request failed after {self.retries + 1} attempts: {last_err}") from last_err


def _desc_from_key(key: str) -> Descriptor:
    parts = key.split("|")
    tags = tuple(tuple(p.split("=", 1)) for p in parts[1:])
    return Descriptor(parts[0], tags)


# ---------------------------------------------------------------------------
# TextureStore — SQLite manifest + content-addressed files. The manifest is
# the system of record for "which appearances exist, in what state, used when";
# the PNGs on disk are disposable (evict freely; keys regenerate identically).
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets(
    key         TEXT PRIMARY KEY,
    key_hash    TEXT UNIQUE NOT NULL,
    subject     TEXT NOT NULL,
    lod         INTEGER NOT NULL,
    tags        TEXT NOT NULL,          -- json {axis: value}
    prompt      TEXT,
    negative    TEXT,
    status      TEXT NOT NULL,          -- pending|generating|ready|failed|evicted
    variations  INTEGER NOT NULL,
    px          INTEGER NOT NULL,
    backend     TEXT,
    error       TEXT,
    created_at  REAL, generated_at REAL, last_used_at REAL,
    use_count   INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS variations(
    key TEXT NOT NULL, idx INTEGER NOT NULL,
    path TEXT NOT NULL, bytes INTEGER, sha1 TEXT,
    provisional INTEGER DEFAULT 0,      -- 1 = placeholder art, replace on ready
    PRIMARY KEY(key, idx)
);
CREATE INDEX IF NOT EXISTS idx_subject_status ON assets(subject, status);
CREATE INDEX IF NOT EXISTS idx_lru ON assets(status, last_used_at);
"""


class TextureStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "assets"), exist_ok=True)
        os.makedirs(os.path.join(root, "placeholders"), exist_ok=True)
        self.db = sqlite3.connect(os.path.join(root, "store.db"),
                                  check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self.lock = threading.RLock()

    # -- paths ---------------------------------------------------------------
    def asset_dir(self, desc: Descriptor) -> str:
        return os.path.join(self.root, "assets", desc.subject, desc.key_hash)

    def placeholder_dir(self, desc: Descriptor) -> str:
        return os.path.join(self.root, "placeholders", desc.key_hash)

    # -- rows ----------------------------------------------------------------
    def get(self, key: str):
        with self.lock:
            r = self.db.execute("SELECT * FROM assets WHERE key=?", (key,)).fetchone()
        if r is None:
            return None
        cols = [c[1] for c in self.db.execute("PRAGMA table_info(assets)")]
        return dict(zip(cols, r))

    def key_for_hash(self, key_hash: str):
        with self.lock:
            r = self.db.execute("SELECT key FROM assets WHERE key_hash=?",
                                (key_hash,)).fetchone()
        return r[0] if r else None

    def insert_pending(self, desc: Descriptor, prompt: str, negative: str,
                       variations: int, px: int):
        now = time.time()
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO assets(key,key_hash,subject,lod,tags,"
                "prompt,negative,status,variations,px,created_at,last_used_at)"
                " VALUES(?,?,?,?,?,?,?,'pending',?,?,?,?)",
                (desc.key, desc.key_hash, desc.subject, desc.lod,
                 json.dumps(desc.tag_dict()), prompt, negative,
                 variations, px, now, now))
            self.db.commit()

    def set_status(self, key: str, status: str, error: str | None = None,
                   backend: str | None = None):
        with self.lock:
            self.db.execute(
                "UPDATE assets SET status=?, error=?, backend=COALESCE(?,backend),"
                " generated_at=CASE WHEN ?='ready' THEN ? ELSE generated_at END"
                " WHERE key=?",
                (status, error, backend, status, time.time(), key))
            self.db.commit()

    def attach_files(self, key: str, paths: list, provisional: bool):
        with self.lock:
            for i, p in enumerate(paths):
                blob = open(p, "rb").read() if os.path.exists(p) else b""
                self.db.execute(
                    "INSERT OR REPLACE INTO variations(key,idx,path,bytes,sha1,"
                    "provisional) VALUES(?,?,?,?,?,?)",
                    (key, i, p, len(blob),
                     hashlib.sha1(blob).hexdigest()[:16], int(provisional)))
            self.db.commit()

    def paths(self, key: str, include_provisional: bool = True) -> list:
        q = "SELECT path FROM variations WHERE key=?"
        if not include_provisional:
            q += " AND provisional=0"
        with self.lock:
            rows = self.db.execute(q + " ORDER BY idx", (key,)).fetchall()
        return [r[0] for r in rows]

    def touch(self, keys, count=None):
        now = time.time()
        with self.lock:
            for k in keys:
                self.db.execute(
                    "UPDATE assets SET last_used_at=?, use_count=use_count+? WHERE key=?",
                    (now, (count or {}).get(k, 1), k))
            self.db.commit()

    def ready_same_subject(self, subject: str, limit: int = 256) -> list:
        with self.lock:
            rows = self.db.execute(
                "SELECT key, tags FROM assets WHERE subject=? AND status='ready'"
                " ORDER BY last_used_at DESC LIMIT ?", (subject, limit)).fetchall()
        return [(k, json.loads(t)) for k, t in rows]

    def stats(self) -> dict:
        with self.lock:
            by = dict(self.db.execute(
                "SELECT status, COUNT(*) FROM assets GROUP BY status").fetchall())
            total_bytes = self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM variations WHERE provisional=0"
            ).fetchone()[0]
            n = self.db.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        return {"assets": n, "by_status": by, "ready_bytes": int(total_bytes)}

    def evict_lru(self, budget_bytes: int) -> int:
        """Delete least-recently-used READY art until under budget. Rows stay
        (status='evicted') so history/prompts survive; files regenerate on
        demand because the key is deterministic. Returns assets evicted."""
        evicted = 0
        with self.lock:
            while True:
                total = self.db.execute(
                    "SELECT COALESCE(SUM(bytes),0) FROM variations v JOIN assets a"
                    " ON a.key=v.key WHERE a.status='ready' AND v.provisional=0"
                ).fetchone()[0]
                if total <= budget_bytes:
                    break
                row = self.db.execute(
                    "SELECT key FROM assets WHERE status='ready'"
                    " ORDER BY last_used_at ASC LIMIT 1").fetchone()
                if row is None:
                    break
                key = row[0]
                for (p,) in self.db.execute(
                        "SELECT path FROM variations WHERE key=?", (key,)):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                self.db.execute("DELETE FROM variations WHERE key=?", (key,))
                self.db.execute("UPDATE assets SET status='evicted' WHERE key=?",
                                (key,))
                evicted += 1
            self.db.commit()
        return evicted


# ---------------------------------------------------------------------------
# Fallback distance — how acceptable is serving key B while key A generates?
# ---------------------------------------------------------------------------

def _tag_distance(want: dict, have: dict) -> float:
    d = 0.0
    for ax in AXES:
        w, h = want.get(ax.name), have.get(ax.name)
        if w is None and h is None:
            continue
        if w is None or h is None:
            d += 0.5 * ax.weight
        elif w != h:
            if ax.ordinal:
                d += ax.weight * abs(ax.values.index(w) - ax.values.index(h))
            else:
                d += ax.weight
    return d


# ---------------------------------------------------------------------------
# TextureService — the front door. resolve() never blocks on generation:
# it answers with the best art that EXISTS and quietly queues what SHOULD.
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    key: str
    key_hash: str
    subject: str
    status: str            # pending | generating | ready | failed
    served: str            # exact | fallback:<key> | placeholder
    paths: list
    tags: dict
    footprint: int = 1


class TextureService:
    def __init__(self, root: str = "texture_store", backend: Backend | None = None,
                 variations: int = 3, tile_px: int = 64,
                 max_fallback_distance: float = 6.0):
        self.store = TextureStore(root)
        self.backend = backend or PlaceholderBackend()
        self.variations = variations
        self.tile_px = tile_px
        self.max_fallback_distance = max_fallback_distance
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._queued: set = set()
        self._seq = 0
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # -- sizing: close-up objects render at higher resolution -----------------
    def px_for(self, desc: Descriptor) -> int:
        return self.tile_px * desc.footprint

    # -- the front door --------------------------------------------------------
    def resolve(self, desc: Descriptor, enqueue: bool = True,
                priority: int = 1) -> Resolution:
        """Best available art for one descriptor, NOW. Exact ready art wins;
        otherwise the nearest ready neighbor of the same subject; otherwise a
        deterministic placeholder — and (unless enqueue=False) the exact key
        is queued for real generation so next visit upgrades."""
        row = self.store.get(desc.key)
        if row is None:
            prompt, negative = build_prompt(desc)
            self.store.insert_pending(desc, prompt, negative,
                                      self.variations, self.px_for(desc))
            row = self.store.get(desc.key)
        status = row["status"]
        if status == "ready":
            paths = self.store.paths(desc.key, include_provisional=False)
            if paths:
                return Resolution(desc.key, desc.key_hash, desc.subject, "ready",
                                  "exact", paths, desc.tag_dict(), desc.footprint)
            status = "pending"                     # files gone: regenerate
            self.store.set_status(desc.key, "pending")
        if status == "evicted":                    # LRU-evicted: same story
            status = "pending"
            self.store.set_status(desc.key, "pending")
        if enqueue and status in ("pending", "failed"):
            self._enqueue(desc.key, priority)

        # nearest ready neighbor of the same subject
        want = desc.tag_dict()
        best, best_d = None, self.max_fallback_distance
        for key, tags in self.store.ready_same_subject(desc.subject):
            d = _tag_distance(want, tags)
            if d < best_d:
                best, best_d = key, d
        if best:
            paths = self.store.paths(best, include_provisional=False)
            if paths:
                return Resolution(desc.key, desc.key_hash, desc.subject, status,
                                  f"fallback:{best}", paths, want, desc.footprint)

        # deterministic placeholder, cached on disk once per key
        paths = self._ensure_placeholder(desc)
        return Resolution(desc.key, desc.key_hash, desc.subject, status,
                          "placeholder", paths, want, desc.footprint)

    def resolve_field(self, dfield: DescriptorField, enqueue: bool = True) -> dict:
        """Resolve every distinct appearance in a chunk. Priority = how many
        tiles want the key, so the most visible art generates first. Returns
        {code: Resolution} matching the field's legend."""
        counts: dict = {}
        uniq, cnt = np.unique(dfield.ground, return_counts=True)
        for c, k in zip(uniq.tolist(), cnt.tolist()):
            counts[c] = counts.get(c, 0) + k
        for inst in dfield.props:
            counts[inst.code] = counts.get(inst.code, 0) + inst.footprint ** 2
        out = {}
        touch_keys, touch_counts = [], {}
        for code, desc in dfield.legend.items():
            res = self.resolve(desc, enqueue=enqueue,
                               priority=counts.get(code, 1))
            out[code] = res
            touch_keys.append(desc.key)
            touch_counts[desc.key] = counts.get(code, 1)
        self.store.touch(touch_keys, touch_counts)
        return out

    def prewarm_neighbors(self, dfield: DescriptorField) -> int:
        """Speculatively queue the keys the world is ABOUT to need, at low
        priority so visible work always wins. The clock and camera are
        predictable: season and time-of-day advance cyclically, and zoom moves
        one lod at a time — so for every appearance on screen now, queue its
        next-season, next-tod, and lod+-1 twins. By the time dusk actually
        falls the dusk art is already on disk; that is what makes 'any time,
        any distance, any season' load instantly. Returns keys queued."""
        queued = 0
        for desc in dfield.legend.values():
            tags = desc.tag_dict()
            twins = []
            for ax_name in ("season", "tod"):
                ax = AXES[AXIS_INDEX[ax_name]]
                if ax_name in tags:
                    nxt = ax.values[(ax.values.index(tags[ax_name]) + 1)
                                    % len(ax.values)]
                    twins.append({**tags, ax_name: nxt})
            for dl in (-1, 1):
                lod2 = desc.lod + dl
                if LOD_MIN <= lod2 <= LOD_MAX:
                    twins.append({**tags, "lod": LOD_NAMES[lod2 - LOD_MIN]})
            for t2 in twins:
                d2 = descriptor(desc.subject, **t2)
                if self.store.get(d2.key) is None:
                    prompt, negative = build_prompt(d2)
                    self.store.insert_pending(d2, prompt, negative,
                                              self.variations, self.px_for(d2))
                row = self.store.get(d2.key)
                if row["status"] in ("pending", "failed"):
                    self._enqueue(d2.key, priority=0)
                    queued += 1
        return queued

    # -- generation queue -------------------------------------------------------
    def _enqueue(self, key: str, priority: int):
        with self._lock:
            if key in self._queued:
                return
            self._queued.add(key)
            self._seq += 1
            self._queue.put((-priority, self._seq, key))

    def pending_count(self) -> int:
        return self._queue.qsize()

    def pump(self, max_jobs: int | None = None) -> int:
        """Run queued generation jobs synchronously (tests, demos, batch
        pre-warming). Returns how many jobs ran."""
        ran = 0
        while max_jobs is None or ran < max_jobs:
            try:
                _, _, key = self._queue.get_nowait()
            except queue.Empty:
                break
            self._run_job(key)
            ran += 1
        return ran

    def start_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def stop_worker(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2.0)

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                _, _, key = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._run_job(key)
            except Exception:
                pass                                  # job errors land in the row

    def _run_job(self, key: str):
        desc = _desc_from_key(key)
        row = self.store.get(key)
        if row is None or row["status"] == "ready":
            with self._lock:
                self._queued.discard(key)
            return
        self.store.set_status(key, "generating", backend=self.backend.name)
        prompt, negative = row["prompt"], row["negative"]
        seed = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
        job = GenJob(key, prompt, negative, seed, self.px_for(desc),
                     self.variations)
        try:
            images = self.backend.generate(job)
            adir = self.store.asset_dir(desc)
            os.makedirs(adir, exist_ok=True)
            paths = []
            for i, png in enumerate(images):
                p = os.path.join(adir, f"v{i}.png")
                with open(p, "wb") as f:
                    f.write(png)
                paths.append(p)
            self.store.attach_files(key, paths, provisional=False)
            self.store.set_status(key, "ready", backend=self.backend.name)
        except Exception as e:                        # noqa: BLE001
            self.store.set_status(key, "failed", error=str(e)[:500],
                                  backend=self.backend.name)
        finally:
            with self._lock:
                self._queued.discard(key)

    # -- placeholder cache -------------------------------------------------------
    def _ensure_placeholder(self, desc: Descriptor) -> list:
        pdir = self.store.placeholder_dir(desc)
        paths = [os.path.join(pdir, f"v{i}.png") for i in range(self.variations)]
        if not all(os.path.exists(p) for p in paths):
            os.makedirs(pdir, exist_ok=True)
            seed = int(hashlib.sha1(desc.key.encode()).hexdigest()[:8], 16)
            for i, p in enumerate(paths):
                _paint_placeholder(desc, self.px_for(desc), seed + i).save(p, "PNG")
            self.store.attach_files(desc.key, paths, provisional=True)
        return paths

    # -- maintenance ---------------------------------------------------------------
    def retry_failed(self) -> int:
        with self.store.lock:
            rows = self.store.db.execute(
                "SELECT key FROM assets WHERE status='failed'").fetchall()
        for (k,) in rows:
            self.store.set_status(k, "pending")
            self._enqueue(k, 1)
        return len(rows)

    def stats(self) -> dict:
        s = self.store.stats()
        s["queued"] = self.pending_count()
        s["backend"] = self.backend.name
        return s


# ---------------------------------------------------------------------------
# Demo: derive + resolve a real world at three zoom/time points, generate with
# the placeholder backend, and write a contact sheet of every distinct key.
#   python3 texgen.py --seed 42 --size 128
# ---------------------------------------------------------------------------

def _demo():
    import argparse
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--store", default="texture_store")
    ap.add_argument("--out", default="exports/texture_demo_sheet.png")
    args = ap.parse_args()

    world = wc.build_world(args.seed, args.size, 3)
    svc = TextureService(args.store)
    sea, thr = 0.42, wc.default_river_threshold(args.size)

    views = [
        ("far, summer noon",   4.0,   20.5),
        ("near, summer noon",  400.0, 20.5),
        ("near, winter night", 400.0, 68.05),
    ]
    all_res: dict = {}
    for label, zoom, t in views:
        chunk = world.stream_view(0.62, 0.44, zoom, 48)
        st = wc.state(chunk, t, sea, thr, 0.18, 0.012, 0.65)
        df = derive(chunk, st)
        res = svc.resolve_field(df)
        print(f"[{label}] lod={df.lod} ({LOD_NAMES[df.lod - LOD_MIN]})  "
              f"tiles={chunk.size}^2 -> {len(df.legend)} distinct keys, "
              f"{len(df.props)} prop instances")
        all_res.update({r.key: r for r in res.values()})

    ran = svc.pump()
    print(f"generated {ran} assets via '{svc.backend.name}' backend")
    print("store:", svc.stats())

    # contact sheet: one row per key, its variations side by side
    rows = sorted(all_res.values(), key=lambda r: r.key)[:80]
    cell, pad, label_w = 64, 4, 560
    sheet = Image.new("RGB", (label_w + (cell + pad) * svc.variations + pad,
                              (cell + pad) * len(rows) + pad), (24, 24, 30))
    d = ImageDraw.Draw(sheet)
    for r_i, res in enumerate(rows):
        res2 = svc.resolve(_desc_from_key(res.key), enqueue=False)
        y = pad + r_i * (cell + pad)
        d.text((pad, y + cell // 2 - 5), res2.key[:86], fill=(210, 210, 215))
        for v_i, p in enumerate(res2.paths[:svc.variations]):
            im = Image.open(p).convert("RGBA").resize((cell, cell))
            sheet.paste(im, (label_w + pad + v_i * (cell + pad), y), im)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.save(args.out)
    print(f"contact sheet -> {args.out}  ({len(rows)} keys)")


if __name__ == "__main__":
    _demo()
