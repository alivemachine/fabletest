extends Node2D

const PixelArt = preload("res://scripts/pixel_art.gd")
const FRAME_PACKET_HEADER_BYTES := 48
const FRAME_PACKET_VERSION_MIN := 1
const FRAME_PACKET_VERSION_MAX := 2

# faction tints, matching world_core.CIV_COLORS (index = faction id)
const CIV_COLORS := [
    Color8(214, 69, 65), Color8(232, 184, 58), Color8(58, 176, 168),
    Color8(150, 96, 210), Color8(232, 128, 52), Color8(96, 178, 84),
]

const BIOME_INFO := [
    {"name": "deep_ocean", "base": Color8(30, 60, 120), "accent": Color8(68, 100, 156), "water": true, "foliage": false},
    {"name": "ocean", "base": Color8(45, 85, 155), "accent": Color8(82, 125, 188), "water": true, "foliage": false},
    {"name": "shallow", "base": Color8(70, 130, 180), "accent": Color8(110, 170, 210), "water": true, "foliage": false},
    {"name": "beach", "base": Color8(210, 200, 150), "accent": Color8(232, 219, 176), "water": false, "foliage": false},
    {"name": "desert", "base": Color8(222, 200, 120), "accent": Color8(238, 221, 152), "water": false, "foliage": false},
    {"name": "savanna", "base": Color8(180, 190, 90), "accent": Color8(208, 215, 124), "water": false, "foliage": true},
    {"name": "grassland", "base": Color8(120, 180, 90), "accent": Color8(154, 208, 120), "water": false, "foliage": true},
    {"name": "forest", "base": Color8(60, 140, 70), "accent": Color8(96, 174, 106), "water": false, "foliage": true},
    {"name": "jungle", "base": Color8(30, 110, 55), "accent": Color8(76, 150, 96), "water": false, "foliage": true},
    {"name": "taiga", "base": Color8(90, 140, 110), "accent": Color8(126, 172, 144), "water": false, "foliage": true},
    {"name": "tundra", "base": Color8(170, 180, 170), "accent": Color8(196, 202, 196), "water": false, "foliage": false},
    {"name": "snow", "base": Color8(235, 240, 245), "accent": Color8(250, 252, 255), "water": false, "foliage": false},
    {"name": "mountain", "base": Color8(130, 125, 120), "accent": Color8(162, 156, 150), "water": false, "foliage": false},
    {"name": "high_peak", "base": Color8(200, 200, 205), "accent": Color8(230, 232, 238), "water": false, "foliage": false},
    # sub-biome ground types (worldgen.BIOME_COLORS, appended in this order):
    # the color codes Godot keys tiles from — wheat fields vs jungle vs oasis
    {"name": "tall_grass", "base": Color8(100, 168, 62), "accent": Color8(134, 198, 96), "water": false, "foliage": true},
    {"name": "meadow", "base": Color8(146, 198, 108), "accent": Color8(178, 222, 142), "water": false, "foliage": true},
    {"name": "wheat_soil", "base": Color8(196, 176, 98), "accent": Color8(220, 202, 128), "water": false, "foliage": false},
    {"name": "acacia_scrub", "base": Color8(162, 176, 74), "accent": Color8(192, 204, 108), "water": false, "foliage": true},
    {"name": "dunes", "base": Color8(236, 216, 142), "accent": Color8(248, 234, 172), "water": false, "foliage": false},
    {"name": "reg_rock", "base": Color8(198, 170, 112), "accent": Color8(220, 194, 140), "water": false, "foliage": false},
    {"name": "shrub_steppe", "base": Color8(208, 196, 132), "accent": Color8(228, 216, 160), "water": false, "foliage": true},
    {"name": "oasis", "base": Color8(62, 168, 118), "accent": Color8(98, 198, 150), "water": false, "foliage": true},
    {"name": "glade", "base": Color8(98, 168, 88), "accent": Color8(132, 198, 120), "water": false, "foliage": true},
    {"name": "dark_forest", "base": Color8(38, 98, 48), "accent": Color8(66, 128, 76), "water": false, "foliage": true},
    {"name": "jungle_clear", "base": Color8(72, 148, 82), "accent": Color8(104, 178, 112), "water": false, "foliage": true},
    {"name": "rocky_tundra", "base": Color8(148, 152, 142), "accent": Color8(176, 180, 170), "water": false, "foliage": false},
    {"name": "scree", "base": Color8(108, 102, 96), "accent": Color8(136, 130, 122), "water": false, "foliage": false},
]

var tile_width := 56.0
var tile_height := 28.0
var relief_scale := 72.0
var water_column_height := 8.0
var render_radius := 13

var _has_payload := false
var _size := 0
var _seed := 0
var _sea_effective := 0.42
var _tile_world := 0.003472
var _grid_n := 288
var focus_uv := Vector2(0.5, 0.5)
var _last_chunk_center_uv := Vector2(0.5, 0.5)
var _time_days := 0.0
var _sunlight_mean := 180
var _cloud_mean := 110
var _water_texture: Texture2D
var _tree_textures: Dictionary = {}
# Persistent tile cache keyed by wrapped world-tile Vector2i -> draw record.
# Chunks are merged in as they arrive; drawing samples this cache around the
# player. Data can arrive late, partial, or never -- rendering never waits.
var _tile_cache: Dictionary = {}
var _iso_order: Array = []
var _build_thread: Thread
var _build_mutex := Mutex.new()
var _build_ready := false
var _build_result: Dictionary = {}
var _queued_payload = null
var _build_token := 0

var anchor := Vector2.ZERO


func _ready() -> void:
    _water_texture = PixelArt.make_water_texture(Color8(66, 122, 196), Color8(170, 212, 238))
    _rebuild_iso_order()


func _rebuild_iso_order() -> void:
    # Precompute the fixed render-window offsets once, sorted back-to-front so
    # the isometric painter's order is correct without per-frame sorting.
    _iso_order.clear()
    var r := render_radius
    for oy in range(-r, r + 1):
        for ox in range(-r, r + 1):
            _iso_order.append(Vector2i(ox, oy))
    _iso_order.sort_custom(func(a, b): return (a.x + a.y) < (b.x + b.y))


func _process(_delta: float) -> void:
    # Merge a freshly-built chunk into the persistent tile cache when the worker
    # finishes. This only ADDS data; it never repositions anything already drawn.
    # Rendering is driven purely by the player position, so late/partial/no data
    # all look fine -- the world just fills in around the character.
    if _build_ready:
        _build_mutex.lock()
        var result: Dictionary = _build_result
        _build_result = {}
        _build_ready = false
        _build_mutex.unlock()
        if _build_thread != null:
            _build_thread.wait_to_finish()
            _build_thread = null
        if int(result.get("token", -1)) == _build_token:
            _merge_snapshot(result.get("snapshot", {}))
        if _queued_payload != null:
            var next_payload = _queued_payload
            _queued_payload = null
            _start_build(next_payload)


func _exit_tree() -> void:
    if _build_thread != null:
        _build_thread.wait_to_finish()
        _build_thread = null


func set_focus(uv: Vector2, screen_anchor: Vector2) -> void:
    # The terrain layer origin is pinned to the screen anchor. The player never
    # moves off-screen and never reaches an edge: tiles are drawn in a fixed
    # radius AROUND the player every frame, sliding smoothly sub-tile.
    focus_uv = uv
    anchor = screen_anchor
    position = anchor
    queue_redraw()


func has_payload() -> bool:
    return _has_payload


func tile_world() -> float:
    return _tile_world


func render_radius_tiles() -> int:
    return render_radius


func stream_size() -> int:
    return _size


func time_days() -> float:
    return _time_days


func sunlight_mean_byte() -> int:
    return _sunlight_mean


func cloud_mean_byte() -> int:
    return _cloud_mean


func apply_payload(payload: Dictionary) -> void:
    if not payload.has("fields"):
        return
    if _build_thread != null:
        _queued_payload = payload
        return
    _start_build(payload)


func set_offline_grid(enabled: bool, tiles: int) -> void:
    # Build a static grey checker grid centered at uv (0.5, 0.5) using the exact
    # same field layout the bridge sends, so movement/placement behaves identical
    # to the real map -- but with zero networking. Pure smoothness test.
    if not enabled:
        return
    var size: int = max(8, tiles)
    var cells := size * size
    var biome := PackedByteArray()
    var height := PackedByteArray()
    var sunlight := PackedByteArray()
    var river := PackedByteArray()
    var vegetation := PackedByteArray()
    var scorch := PackedByteArray()
    biome.resize(cells * 2)
    height.resize(cells)
    sunlight.resize(cells)
    river.resize(cells)
    vegetation.resize(cells)
    scorch.resize(cells)
    for gy in range(size):
        for gx in range(size):
            var i := gy * size + gx
            # grassland biome id (6) everywhere -> neutral, but tint via checker
            biome.encode_s16(i * 2, 6)
            var checker := (gx + gy) % 2
            height[i] = 150 + checker * 12
            sunlight[i] = 190
            river[i] = 0
            vegetation[i] = 0
            scorch[i] = 0
    var payload := {
        "size": size,
        "seed": 0,
        "sea_effective": 0.42,
        "tile_world": 1.0 / float(size),
        "center": {"cx": 0.5, "cy": 0.5},
        "time_days": 0.0,
        "sunlight_mean": 190,
        "cloud_mean": 90,
        "fields": {
            "biome_id": biome,
            "height": height,
            "sunlight": sunlight,
            "river": river,
            "vegetation": vegetation,
            "scorch": scorch,
        },
    }
    _tile_cache.clear()
    _merge_snapshot(_build_snapshot(payload))


func apply_payload_bytes(body: PackedByteArray) -> void:
    if body.is_empty():
        return
    if _build_thread != null:
        _queued_payload = body
        return
    _start_build(body)


func focus_radius_tiles(world_uv: Vector2) -> float:
    # How far (in tiles) the player has walked from the center of the last chunk
    # we streamed. main.gd uses this to decide when to prefetch the next chunk.
    var dx := _wrapped_delta(world_uv.x, _last_chunk_center_uv.x) / _tile_world
    var dy := _wrapped_delta(world_uv.y, _last_chunk_center_uv.y) / _tile_world
    return max(absf(dx), absf(dy))


func cell_info(world_uv: Vector2) -> Dictionary:
    # Everything the game knows about the cell under `world_uv`: biome name &
    # id (the tile color code), water & directional flow (uv axes, +y south),
    # fauna densities, structure (0 none / 1 road / 2 building), owning
    # faction, moisture/temperature, sunlight, vegetation, scorch, height.
    # Returns {} until that cell has streamed in — callers must tolerate it.
    var tile := _world_to_tile(world_uv)
    var rec = _tile_cache.get(_tile_key(tile.x, tile.y), null)
    if rec == null:
        return {}
    return rec.get("env", {})


func ground_height_for_focus(world_uv: Vector2) -> float:
    var tile := _world_to_tile(world_uv)
    var rec = _tile_cache.get(_tile_key(tile.x, tile.y), null)
    if rec == null:
        return 12.0
    return float(rec["height"])


func _world_tile_f(world_uv: Vector2) -> Vector2:
    # Continuous world-tile coordinate (fractional). One unit == one tile.
    return Vector2(world_uv.x / _tile_world, world_uv.y / _tile_world)


func _world_to_tile(world_uv: Vector2) -> Vector2i:
    var f := _world_tile_f(world_uv)
    return Vector2i(int(floor(f.x + 0.5)), int(floor(f.y + 0.5)))


func _tile_key(tx: int, ty: int) -> Vector2i:
    # Wrap into the global tile lattice so the world is seamless/toroidal.
    var n := _grid_n
    return Vector2i(((tx % n) + n) % n, ((ty % n) + n) % n)


func _draw() -> void:
    # Draw a fixed radius of tiles AROUND the player, sampled from the cache.
    # The player's fractional tile position gives a smooth sub-tile slide; any
    # tile with no data yet is skipped (shows the sky/void placeholder), so the
    # experience stays fluid whether data is present, partial, or absent.
    var f := _world_tile_f(focus_uv)
    var base_x := int(floor(f.x + 0.5))
    var base_y := int(floor(f.y + 0.5))
    var frac := Vector2(f.x - float(base_x), f.y - float(base_y))
    # Screen offset for the player's sub-tile position (iso projection).
    var slide := Vector2((frac.x - frac.y) * tile_width * 0.5,
        (frac.x + frac.y) * tile_height * 0.5)

    for off_v in _iso_order:
        var off: Vector2i = off_v
        var rec = _tile_cache.get(_tile_key(base_x + off.x, base_y + off.y), null)
        if rec == null:
            continue
        var iso := Vector2(float(off.x - off.y) * tile_width * 0.5,
            float(off.x + off.y) * tile_height * 0.5) - slide
        var height_px: float = rec["height_px"]
        _draw_block(iso, height_px, rec["top"], rec["left"], rec["right"])
        var water_scale: float = rec["water_scale"]
        if water_scale > 0.0:
            _draw_water_surface(iso, height_px, water_scale, rec["water_tint"])
        var tree_biome: int = rec["tree_biome"]
        if tree_biome >= 0:
            _draw_tree(iso, height_px, tree_biome, rec["tree_sun"], rec["tree_veg"])
        if int(rec.get("structure", 0)) == 2:
            _draw_building(iso, height_px, rec.get("wall", Color8(200, 190, 170)))


func _start_build(payload) -> void:
    _build_token += 1
    _build_thread = Thread.new()
    var err := _build_thread.start(Callable(self, "_build_snapshot_thread").bind(payload, _build_token))
    if err != OK:
        _build_thread = null
        _merge_snapshot(_build_snapshot_variant(payload))


func _build_snapshot_thread(payload, token: int) -> void:
    var snapshot: Dictionary = _build_snapshot_variant(payload)
    _build_mutex.lock()
    _build_result = {"token": token, "snapshot": snapshot}
    _build_ready = true
    _build_mutex.unlock()


func _merge_snapshot(snapshot: Dictionary) -> void:
    if snapshot.is_empty():
        return
    _size = int(snapshot.get("size", 0))
    _seed = int(snapshot.get("seed", 0))
    _sea_effective = float(snapshot.get("sea_effective", 0.42))
    _tile_world = max(float(snapshot.get("tile_world", 0.001)), 1e-6)
    _grid_n = max(1, int(round(1.0 / _tile_world)))
    _last_chunk_center_uv = snapshot.get("center_uv", Vector2(0.5, 0.5))
    _time_days = float(snapshot.get("time_days", _time_days))
    _sunlight_mean = int(snapshot.get("sunlight_mean", _sunlight_mean))
    _cloud_mean = int(snapshot.get("cloud_mean", _cloud_mean))
    # Merge this chunk's tiles into the persistent global cache. Newer data for a
    # tile overwrites older data (keeps the simulation fresh); tiles outside this
    # chunk are left untouched, so previously-seen terrain persists behind you.
    var tiles: Array = snapshot.get("tiles", [])
    for rec_v in tiles:
        var rec: Dictionary = rec_v
        _tile_cache[rec["key"]] = rec
    _has_payload = not _tile_cache.is_empty()
    _prune_cache()
    queue_redraw()


func _prune_cache() -> void:
    # Bound memory: drop tiles far outside a generous window around the player so
    # the cache can't grow without limit as you roam the whole planet.
    var limit := (render_radius + 24) * 2
    if _tile_cache.size() <= 6000:
        return
    var center := _world_to_tile(focus_uv)
    var stale: Array = []
    for key_v in _tile_cache.keys():
        var key: Vector2i = key_v
        var dx: int = absi(_wrapi(key.x - center.x, _grid_n))
        var dy: int = absi(_wrapi(key.y - center.y, _grid_n))
        if dx > limit or dy > limit:
            stale.append(key)
    for key_v in stale:
        _tile_cache.erase(key_v)


func _wrapi(v: int, n: int) -> int:
    var m := ((v % n) + n) % n
    if m > n / 2:
        m -= n
    return m


func _build_snapshot_variant(payload) -> Dictionary:
    if typeof(payload) == TYPE_PACKED_BYTE_ARRAY:
        var packet := _decode_frame_packet(payload)
        if packet.is_empty():
            var parsed: Variant = JSON.parse_string(payload.get_string_from_utf8())
            if typeof(parsed) == TYPE_DICTIONARY:
                return _build_snapshot(parsed)
            return {}
        return _build_snapshot(packet)
    if typeof(payload) == TYPE_DICTIONARY:
        return _build_snapshot(payload)
    return {}


func _field_u8(field, index: int, default_value: int = 0) -> int:
    if field is PackedByteArray:
        return int(field[index]) if index >= 0 and index < field.size() else default_value
    if field is Array:
        return int(field[index]) if index >= 0 and index < field.size() else default_value
    return default_value


func _field_s16(field, index: int, default_value: int = 0) -> int:
    if field is PackedByteArray:
        var byte_index := index * 2
        if byte_index >= 0 and byte_index + 1 < field.size():
            return int(field.decode_s16(byte_index))
        return default_value
    if field is Array:
        return int(field[index]) if index >= 0 and index < field.size() else default_value
    return default_value


func _has_field_u8(field, index: int) -> bool:
    if field is PackedByteArray or field is Array:
        return index >= 0 and index < field.size()
    return false


func _has_field_s16(field, index: int) -> bool:
    if field is PackedByteArray:
        var byte_index := index * 2
        return byte_index >= 0 and byte_index + 1 < field.size()
    if field is Array:
        return index >= 0 and index < field.size()
    return false


func _build_snapshot(payload: Dictionary) -> Dictionary:
    var fields: Dictionary = payload["fields"]
    var size: int = int(payload.get("size", 0))
    var seed: int = int(payload.get("seed", 0))
    var sea_effective: float = float(payload.get("sea_effective", 0.42))
    var tile_world: float = max(float(payload.get("tile_world", 0.001)), 1e-6)
    var center: Dictionary = payload.get("center", {"cx": 0.5, "cy": 0.5})
    var center_uv: Vector2 = Vector2(float(center.get("cx", 0.5)), float(center.get("cy", 0.5)))
    var heights = fields.get("height", PackedByteArray())
    var biomes = fields.get("biome_id", PackedByteArray())
    var sunlight = fields.get("sunlight", PackedByteArray())
    var rivers = fields.get("river", PackedByteArray())
    var vegetation = fields.get("vegetation", PackedByteArray())
    var scorch = fields.get("scorch", PackedByteArray())
    var flow_dir = fields.get("flow_dir", PackedByteArray())
    var flow_speed = fields.get("flow_speed", PackedByteArray())
    var fauna_herb = fields.get("fauna_herb", PackedByteArray())
    var fauna_pred = fields.get("fauna_pred", PackedByteArray())
    var structure_f = fields.get("structure", PackedByteArray())
    var faction_f = fields.get("faction", PackedByteArray())
    var moisture_f = fields.get("moisture", PackedByteArray())
    var temperature_f = fields.get("temperature", PackedByteArray())
    # Map each chunk cell to its GLOBAL tile coordinate. The chunk's center cell
    # (size/2, size/2) sits at center_uv, i.e. global tile round(center_uv/tile).
    var grid_n: int = max(1, int(round(1.0 / tile_world)))
    var origin_x := int(floor(center_uv.x / tile_world + 0.5)) - size / 2
    var origin_y := int(floor(center_uv.y / tile_world + 0.5)) - size / 2
    var tiles: Array = []
    if size > 0:
        for gy in range(size):
            for gx in range(size):
                var index := gy * size + gx
                if not _has_field_u8(heights, index) or not _has_field_s16(biomes, index):
                    continue
                var biome_id := clampi(_field_s16(biomes, index), 0, BIOME_INFO.size() - 1)
                var info: Dictionary = BIOME_INFO[biome_id]
                var base: Color = info["base"]
                var accent: Color = info["accent"]
                var water: bool = info["water"]
                var foliage: bool = info["foliage"]
                var sun := float(_field_u8(sunlight, index, 199)) / 255.0
                var river := float(_field_u8(rivers, index, 0)) / 255.0
                var veg := float(_field_u8(vegetation, index, 0)) / 255.0
                var burn := float(_field_u8(scorch, index, 0)) / 255.0
                var elev := float(_field_u8(heights, index)) / 255.0
                var structure := _field_u8(structure_f, index, 0)
                var faction := _field_u8(faction_f, index, 0) - 1
                var speed := float(_field_u8(flow_speed, index, 0)) / 255.0
                var flow := Vector2.ZERO
                if speed > 0.01:
                    var ang := float(_field_u8(flow_dir, index, 0)) / 255.0 * TAU
                    flow = Vector2(cos(ang), sin(ang)) * speed
                var height_px := _column_height_px_for(elev, sea_effective)
                var top_color := base.lerp(accent, 0.18).lerp(Color.WHITE, sun * 0.24)
                if burn > 0.0:
                    top_color = top_color.lerp(Color8(82, 56, 42), burn * 0.58)
                if structure == 1:
                    top_color = top_color.lerp(Color8(124, 104, 78), 0.62)  # dirt road
                var left_color := top_color.darkened(0.28 if not water else 0.10)
                var right_color := top_color.darkened(0.16 if not water else 0.18)
                var water_scale := 0.0
                var water_tint := Color(0.0, 0.0, 0.0, 0.0)
                if water:
                    water_scale = 0.94
                    water_tint = Color(0.94, 0.99, 1.0, 0.96)
                elif river > 0.05:
                    water_scale = 0.40 + river * 0.42
                    water_tint = Color(0.76, 0.92, 1.0, clampf(0.50 + river * 0.45, 0.50, 0.92))
                var gtx := origin_x + gx
                var gty := origin_y + gy
                var tree_biome := -1
                var tree_sun := 0.0
                var tree_veg := 0.0
                if structure == 0 and foliage and veg > 0.26 and _hash01(gtx, gty, seed) > 0.55:
                    tree_biome = biome_id
                    tree_sun = sun
                    tree_veg = veg
                var wall_color := Color(0.0, 0.0, 0.0, 0.0)
                if structure == 2:
                    var tint: Color = CIV_COLORS[clampi(faction, 0, CIV_COLORS.size() - 1)] \
                        if faction >= 0 else Color8(160, 150, 132)
                    wall_color = tint.lerp(Color8(228, 214, 186), 0.58)
                var key := Vector2i(((gtx % grid_n) + grid_n) % grid_n,
                    ((gty % grid_n) + grid_n) % grid_n)
                tiles.append({
                    "key": key,
                    "height": elev,
                    "height_px": height_px,
                    "top": top_color,
                    "left": left_color,
                    "right": right_color,
                    "water_scale": water_scale,
                    "water_tint": water_tint,
                    "tree_biome": tree_biome,
                    "tree_sun": tree_sun,
                    "tree_veg": tree_veg,
                    "structure": structure,
                    "wall": wall_color,
                    # everything the PLAYER reads about this cell, in one place
                    "env": {
                        "biome_id": biome_id,
                        "biome": info["name"],
                        "water": water,
                        "river": river,
                        "flow": flow,               # direction * speed, uv axes
                        "herb": float(_field_u8(fauna_herb, index, 0)) / 255.0,
                        "pred": float(_field_u8(fauna_pred, index, 0)) / 255.0,
                        "structure": structure,     # 0 none / 1 road / 2 building
                        "faction": faction,         # -1 = unclaimed
                        "moisture": float(_field_u8(moisture_f, index, 0)) / 255.0,
                        "temperature": float(_field_u8(temperature_f, index, 0)) / 255.0,
                        "sunlight": sun,
                        "vegetation": veg,
                        "scorch": burn,
                        "height": elev,
                    },
                })
    return {
        "size": size,
        "seed": seed,
        "sea_effective": sea_effective,
        "tile_world": tile_world,
        "center_uv": center_uv,
        "time_days": float(payload.get("time_days", 0.0)),
        "sunlight_mean": int(payload.get("sunlight_mean", 180)),
        "cloud_mean": int(payload.get("cloud_mean", 110)),
        "tiles": tiles,
    }


func _copy_bytes(body: PackedByteArray, offset: int, count: int) -> PackedByteArray:
    var out := PackedByteArray()
    if count <= 0 or offset < 0 or offset + count > body.size():
        return out
    out.resize(count)
    for i in range(count):
        out[i] = body[offset + i]
    return out


func _decode_frame_packet(body: PackedByteArray) -> Dictionary:
    if body.size() < FRAME_PACKET_HEADER_BYTES:
        return {}
    if body[0] != 70 or body[1] != 84 or body[2] != 66 or body[3] != 49:
        return {}
    var version := int(body.decode_u16(4))
    if version < FRAME_PACKET_VERSION_MIN or version > FRAME_PACKET_VERSION_MAX:
        return {}
    var size := int(body.decode_u16(6))
    var seed_v := int(body.decode_u32(8))
    var planet_size := int(body.decode_u16(12))
    var target_tiles := int(body.decode_u16(14))
    var cells := size * size
    var offset := FRAME_PACKET_HEADER_BYTES
    var fields := {}
    var biome_bytes := _copy_bytes(body, offset, cells * 2)
    offset += cells * 2
    fields["biome_id"] = biome_bytes
    var u8_names: Array = ["height", "sunlight", "river", "vegetation", "scorch"]
    if version >= 2:
        # v2 appends the environment planes the player character reads
        u8_names += ["flow_dir", "flow_speed", "fauna_herb", "fauna_pred",
                     "structure", "faction", "moisture", "temperature"]
    for field_name in u8_names:
        var plane := _copy_bytes(body, offset, cells)
        offset += cells
        if plane.size() != cells:
            return {}
        fields[field_name] = plane
    return {
        "seed": seed_v,
        "planet_size": planet_size,
        "view_tiles_target": target_tiles,
        "size": size,
        "time_days": float(body.decode_float(16)),
        "span": float(body.decode_float(20)),
        "center": {
            "cx": float(body.decode_float(24)),
            "cy": float(body.decode_float(28)),
        },
        "tile_world": float(body.decode_float(32)),
        "sea_level": float(body.decode_float(36)),
        "sea_effective": float(body.decode_float(40)),
        "sunlight_mean": int(body[44]),
        "cloud_mean": int(body[45]),
        "fields": fields,
    }


func _wrapped_delta(value: float, center: float) -> float:
    return fposmod(value - center + 0.5, 1.0) - 0.5


func _column_height_px_for(elev: float, sea_effective: float) -> float:
    if elev < sea_effective:
        return water_column_height
    return clampf(10.0 + (elev - sea_effective) * relief_scale, 8.0, 44.0)


func _draw_block(center: Vector2, height_px: float, top_color: Color, left_color: Color, right_color: Color) -> void:
    var half_w := tile_width * 0.5
    var half_h := tile_height * 0.5
    var top_center := center - Vector2(0.0, height_px)
    var top := top_center + Vector2(0.0, -half_h)
    var right := top_center + Vector2(half_w, 0.0)
    var bottom := top_center + Vector2(0.0, half_h)
    var left := top_center + Vector2(-half_w, 0.0)
    var drop := Vector2(0.0, height_px)

    draw_colored_polygon(PackedVector2Array([left, bottom, bottom + drop, left + drop]), left_color)
    draw_colored_polygon(PackedVector2Array([right, bottom, bottom + drop, right + drop]), right_color)
    draw_colored_polygon(PackedVector2Array([top, right, bottom, left]), top_color)


func _draw_water_surface(center: Vector2, height_px: float, scale: float, tint: Color) -> void:
    var draw_width := tile_width * clampf(scale, 0.24, 1.0)
    var draw_height := tile_height * clampf(scale, 0.22, 0.95)
    var top_center := center - Vector2(0.0, height_px + 1.0)
    var rect := Rect2(top_center - Vector2(draw_width * 0.5, draw_height * 0.5), Vector2(draw_width, draw_height))
    draw_texture_rect(_water_texture, rect, false, tint)


func _draw_tree(center: Vector2, height_px: float, biome_id: int, sun: float, veg: float) -> void:
    var trunk_height := 10.0 + veg * 12.0
    var top_center := center - Vector2(0.0, height_px)
    var trunk_rect := Rect2(top_center + Vector2(-2.0, -trunk_height + 6.0), Vector2(4.0, trunk_height))
    draw_rect(trunk_rect, Color8(92, 64, 44).lerp(Color.WHITE, sun * 0.12))

    var canopy := _tree_texture(biome_id)
    var size := Vector2(24.0 + veg * 18.0, 24.0 + veg * 16.0)
    var pos := top_center + Vector2(-size.x * 0.5, -size.y + 8.0)
    draw_texture_rect(canopy, Rect2(pos, size), false,
        Color(1.0, 1.0, 1.0, 0.94).lerp(Color(1.0, 1.0, 1.0, 1.0), sun * 0.14))


func _tree_texture(biome_id: int) -> Texture2D:
    var key := str(biome_id)
    if _tree_textures.has(key):
        return _tree_textures[key]
    var info: Dictionary = BIOME_INFO[biome_id]
    var base: Color = info["base"]
    var accent: Color = info["accent"]
    var texture: Texture2D = PixelArt.make_canopy_texture(base.lightened(0.06), accent)
    _tree_textures[key] = texture
    return texture


func _hash01(x: int, y: int, salt: int) -> float:
    var value := sin(float(x * 127 + y * 311 + salt * 29)) * 43758.5453
    return absf(fmod(value, 1.0))