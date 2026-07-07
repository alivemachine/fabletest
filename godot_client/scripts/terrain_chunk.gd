extends Node2D

const FRAME_PACKET_HEADER_BYTES := 48
const FRAME_PACKET_VERSION := 3

# ---------------------------------------------------------------------------
# TILE PALETTE — the single source of truth for what each color means in-game.
# Python sends a color_map bitmap; Godot looks up each pixel here to get:
#   water       — flat water surface rendered on top
#   foliage     — eligible for tree placement
#   walkable    — can the player walk on this tile
#   physics     — collision shape type ("walk", "solid", "water", "none")
#   resource    — what can be harvested here (empty = nothing)
# To remap a biome to a different texture or behavior: change it here in the
# Editor. Python never needs to change. New biomes = new entries.
# ---------------------------------------------------------------------------
const TILE_PALETTE: Dictionary = {
	# key = Color8(r,g,b) matching worldgen.BIOME_COLORS
	Color8( 30,  60, 120): {"name":"deep_ocean",   "water":true,  "foliage":false, "walkable":false, "physics":"water",  "resource":""},
	Color8( 45,  85, 155): {"name":"ocean",         "water":true,  "foliage":false, "walkable":false, "physics":"water",  "resource":""},
	Color8( 70, 130, 180): {"name":"shallow",       "water":true,  "foliage":false, "walkable":false, "physics":"water",  "resource":"fish"},
	Color8(210, 200, 150): {"name":"beach",         "water":false, "foliage":false, "walkable":true,  "physics":"walk",   "resource":"sand"},
	Color8(222, 200, 120): {"name":"desert",        "water":false, "foliage":false, "walkable":true,  "physics":"walk",   "resource":""},
	Color8(180, 190,  90): {"name":"savanna",       "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"hide"},
	Color8(120, 180,  90): {"name":"grassland",     "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"food"},
	Color8( 60, 140,  70): {"name":"forest",        "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"wood"},
	Color8( 30, 110,  55): {"name":"jungle",        "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"wood"},
	Color8( 90, 140, 110): {"name":"taiga",         "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"wood"},
	Color8(170, 180, 170): {"name":"tundra",        "water":false, "foliage":false, "walkable":true,  "physics":"walk",   "resource":""},
	Color8(235, 240, 245): {"name":"snow",          "water":false, "foliage":false, "walkable":true,  "physics":"walk",   "resource":""},
	Color8(130, 125, 120): {"name":"mountain",      "water":false, "foliage":false, "walkable":false, "physics":"solid",  "resource":"stone"},
	Color8(200, 200, 205): {"name":"high_peak",     "water":false, "foliage":false, "walkable":false, "physics":"solid",  "resource":"ore"},
	# sub-biomes
	Color8(100, 168,  62): {"name":"tall_grass",    "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"food"},
	Color8(146, 198, 108): {"name":"meadow",        "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"food"},
	Color8(196, 176,  98): {"name":"wheat_soil",    "water":false, "foliage":false, "walkable":true,  "physics":"walk",   "resource":"food"},
	Color8(162, 176,  74): {"name":"acacia_scrub",  "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"hide"},
	Color8(236, 216, 142): {"name":"dunes",         "water":false, "foliage":false, "walkable":true,  "physics":"walk",   "resource":"sand"},
	Color8(198, 170, 112): {"name":"reg_rock",      "water":false, "foliage":false, "walkable":true,  "physics":"walk",   "resource":"stone"},
	Color8(208, 196, 132): {"name":"shrub_steppe",  "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":""},
	Color8( 62, 168, 118): {"name":"oasis",         "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"food"},
	Color8( 98, 168,  88): {"name":"glade",         "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"wood"},
	Color8( 38,  98,  48): {"name":"dark_forest",   "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"wood"},
	Color8( 72, 148,  82): {"name":"jungle_clear",  "water":false, "foliage":true,  "walkable":true,  "physics":"walk",   "resource":"wood"},
	Color8(148, 152, 142): {"name":"rocky_tundra",  "water":false, "foliage":false, "walkable":true,  "physics":"walk",   "resource":"stone"},
	Color8(108, 102,  96): {"name":"scree",         "water":false, "foliage":false, "walkable":false, "physics":"solid",  "resource":"stone"},
}

# Fallback definition for colors that don't exactly match the palette
# (e.g. lit/blended colors from composite render slightly off-key).
const TILE_DEFAULT: Dictionary = {"name":"unknown", "water":false, "foliage":false, "walkable":true, "physics":"walk", "resource":""}

# Pre-built palette as sorted arrays for fast nearest-color lookup.
var _palette_colors: PackedColorArray
var _palette_keys: Array  # parallel array of Color keys into TILE_PALETTE
var _palette_cache: Dictionary = {}  # memoize lookup results

const BIOME_INFO := [  # kept for offline grid fallback only
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
# Persistent tile cache keyed by wrapped world-tile Vector2i -> draw record.
# Chunks are merged in as they arrive; drawing samples this cache around the
# player. Data can arrive late, partial, or never -- rendering never waits.
var _tile_cache: Dictionary = {}
var _build_thread: Thread
var _build_mutex := Mutex.new()
var _build_ready := false
var _build_result: Dictionary = {}
var _queued_payload = null
var _build_token := 0

var anchor := Vector2.ZERO

# ArrayMesh renderer: all tile geometry is baked into a MeshInstance2D once per
# chunk-data-change or once per tile-step. Moving sub-tile is just a position
# offset on the mesh node — zero GDScript draw work per frame at runtime.
var _mesh_node: MeshInstance2D
var _mesh_front: ArrayMesh
var _mesh_back: ArrayMesh
var _mesh_base := Vector2i(-99999, -99999)
var _mesh_buf_radius := 8
var _mesh_thread: Thread
var _mesh_mutex := Mutex.new()
var _mesh_ready := false
var _mesh_result: Dictionary = {}


func _ready() -> void:
	_build_palette_index()
	_setup_mesh_renderer()


func _build_palette_index() -> void:
	_palette_colors = PackedColorArray()
	_palette_keys = []
	for c in TILE_PALETTE.keys():
		_palette_colors.append(c)
		_palette_keys.append(c)


func _palette_lookup(color: Color) -> Dictionary:
	# Exact match first (memoized)
	var key := Color(roundf(color.r * 255) / 255.0, roundf(color.g * 255) / 255.0, roundf(color.b * 255) / 255.0, 1.0)
	if _palette_cache.has(key):
		return _palette_cache[key]
	# Try exact Color8 match
	var c8 := Color8(int(color.r * 255), int(color.g * 255), int(color.b * 255))
	if TILE_PALETTE.has(c8):
		_palette_cache[key] = TILE_PALETTE[c8]
		return TILE_PALETTE[c8]
	# Nearest-color fallback (lit composites won't be exact)
	var best_dist := 1e9
	var best: Dictionary = TILE_DEFAULT
	for i in range(_palette_keys.size()):
		var pc: Color = _palette_colors[i]
		var dr := color.r - pc.r
		var dg := color.g - pc.g
		var db := color.b - pc.b
		var d := dr * dr + dg * dg + db * db
		if d < best_dist:
			best_dist = d
			best = TILE_PALETTE[_palette_keys[i]]
	_palette_cache[key] = best
	return best


func _process(_delta: float) -> void:
	if _mesh_ready:
		_mesh_mutex.lock()
		var mresult := _mesh_result
		_mesh_result = {}
		_mesh_ready = false
		_mesh_mutex.unlock()
		if _mesh_thread != null:
			_mesh_thread.wait_to_finish()
			_mesh_thread = null
		_upload_mesh(mresult)
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
	if _mesh_thread != null:
		_mesh_thread.wait_to_finish()
		_mesh_thread = null


func set_focus(uv: Vector2, screen_anchor: Vector2) -> void:
	focus_uv = uv
	anchor = screen_anchor
	position = anchor
	if _mesh_node == null:
		return
	# Pixel offset from mesh_base to current player — continuous tile delta.
	# This is the ONLY per-frame work: one Vector2 assignment.
	var f := _world_tile_f(uv)
	var dx := f.x - float(_mesh_base.x)
	var dy := f.y - float(_mesh_base.y)
	_mesh_node.position = Vector2(-(dx - dy) * tile_width * 0.5,
		-(dx + dy) * tile_height * 0.5)
	# Trigger an off-thread mesh rebuild when player nears the buffer edge.
	var cur_ix := int(floor(f.x + 0.5))
	var cur_iy := int(floor(f.y + 0.5))
	if max(absi(cur_ix - _mesh_base.x), absi(cur_iy - _mesh_base.y)) > _mesh_buf_radius - 2:
		if _mesh_thread == null:
			_request_mesh_rebuild()


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
	if payload.is_empty():
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


func _setup_mesh_renderer() -> void:
	_mesh_front = ArrayMesh.new()
	_mesh_back = ArrayMesh.new()
	_mesh_node = MeshInstance2D.new()
	_mesh_node.mesh = _mesh_front
	var shader := Shader.new()
	shader.code = "shader_type canvas_item;\nvoid fragment() { COLOR = COLOR; }"
	var mat := ShaderMaterial.new()
	mat.shader = shader
	_mesh_node.material = mat
	add_child(_mesh_node)


func _request_mesh_rebuild() -> void:
	if _mesh_thread != null:
		return
	var f := _world_tile_f(focus_uv)
	var new_base := Vector2i(int(floor(f.x + 0.5)), int(floor(f.y + 0.5)))
	var r := render_radius + _mesh_buf_radius
	var gn := _grid_n
	# Snapshot relevant tiles on main thread — safe, no race with _merge_snapshot.
	var cache_snap := {}
	for oy in range(-r, r + 1):
		for ox in range(-r, r + 1):
			var tx := (((new_base.x + ox) % gn) + gn) % gn
			var ty := (((new_base.y + oy) % gn) + gn) % gn
			var rec = _tile_cache.get(Vector2i(tx, ty), null)
			if rec != null:
				cache_snap[Vector2i(tx, ty)] = rec
	_mesh_thread = Thread.new()
	_mesh_thread.start(Callable(self, "_build_mesh_thread_func").bind(new_base, cache_snap, gn))


func _build_mesh_thread_func(new_base: Vector2i, cache: Dictionary, gn: int) -> void:
	var result := _build_mesh_data(new_base.x, new_base.y, cache, gn)
	result["base"] = new_base
	_mesh_mutex.lock()
	_mesh_result = result
	_mesh_ready = true
	_mesh_mutex.unlock()


func _build_mesh_data(base_x: int, base_y: int, cache: Dictionary, gn: int) -> Dictionary:
	var r := render_radius + _mesh_buf_radius
	var hw := tile_width * 0.5
	var hh := tile_height * 0.5
	var verts_arr := []
	var colors_arr := []
	# Diagonal sweep gives iso back-to-front order (d = ox+oy ascending).
	for d in range(-2 * r, 2 * r + 1):
		for ox in range(-r, r + 1):
			var oy := d - ox
			if oy < -r or oy > r:
				continue
			var tx := (((base_x + ox) % gn) + gn) % gn
			var ty_i := (((base_y + oy) % gn) + gn) % gn
			var rec = cache.get(Vector2i(tx, ty_i), null)
			if rec == null:
				continue
			var hp: float = rec["height_px"]
			var tc: Color = rec["top"]
			var lc: Color = rec["left"]
			var rc_c: Color = rec["right"]
			if rec["water_scale"] > 0.0:
				tc = tc.lerp(Color(0.50, 0.80, 1.0), 0.40)
			var cx := float(ox - oy) * hw
			var cy := float(ox + oy) * hh
			var tyv := cy - hp  # Y of top-center
			# Left face
			_arr_quad(verts_arr, colors_arr,
				cx - hw, tyv,      cx, tyv + hh,
				cx, cy + hh,       cx - hw, cy, lc)
			# Right face
			_arr_quad(verts_arr, colors_arr,
				cx + hw, tyv,      cx, tyv + hh,
				cx, cy + hh,       cx + hw, cy, rc_c)
			# Top face
			_arr_quad(verts_arr, colors_arr,
				cx, tyv - hh,      cx + hw, tyv,
				cx, tyv + hh,      cx - hw, tyv, tc)
			# Tree canopy
			if rec["tree_biome"] >= 0:
				var veg: float = rec["tree_veg"]
				var sun: float = rec["tree_sun"]
				var cc: Color = rec["canopy_color"].lerp(Color.WHITE, sun * 0.10)
				var trunk_h := 10.0 + veg * 12.0
				var cw := 12.0 + veg * 10.0
				var ch_v := 8.0 + veg * 8.0
				var trunk_c: Color = Color8(92, 64, 44).lerp(Color.WHITE, sun * 0.12)
				_arr_quad(verts_arr, colors_arr,
					cx - 2.0, tyv - trunk_h + 6.0,   cx + 2.0, tyv - trunk_h + 6.0,
					cx + 2.0, tyv + 6.0,              cx - 2.0, tyv + 6.0, trunk_c)
				var cp := tyv - trunk_h + 4.0
				_arr_quad(verts_arr, colors_arr,
					cx, cp - ch_v,   cx + cw, cp,
					cx, cp + ch_v,   cx - cw, cp, cc)
			# Building
			if int(rec.get("structure", 0)) == 2:
				var wall: Color = rec.get("wall", Color8(200, 190, 170))
				var fw := tile_width * 0.30
				var fh_b := tile_height * 0.30
				var wh := 22.0
				_arr_quad(verts_arr, colors_arr,
					cx - fw, tyv,          cx, tyv + fh_b,
					cx, tyv + fh_b - wh,   cx - fw, tyv - wh, wall.darkened(0.30))
				_arr_quad(verts_arr, colors_arr,
					cx + fw, tyv,          cx, tyv + fh_b,
					cx, tyv + fh_b - wh,   cx + fw, tyv - wh, wall.darkened(0.14))
				_arr_quad(verts_arr, colors_arr,
					cx, tyv - fh_b - wh,   cx + fw, tyv - wh,
					cx, tyv + fh_b - wh,   cx - fw, tyv - wh,
					wall.lerp(Color8(96, 62, 52), 0.5).lightened(0.04))
	return {
		"verts": PackedVector2Array(verts_arr),
		"colors": PackedColorArray(colors_arr),
	}


func _arr_quad(verts: Array, colors: Array,
		x0: float, y0: float, x1: float, y1: float,
		x2: float, y2: float, x3: float, y3: float,
		color: Color) -> void:
	verts.append(Vector2(x0, y0)); colors.append(color)
	verts.append(Vector2(x1, y1)); colors.append(color)
	verts.append(Vector2(x2, y2)); colors.append(color)
	verts.append(Vector2(x0, y0)); colors.append(color)
	verts.append(Vector2(x2, y2)); colors.append(color)
	verts.append(Vector2(x3, y3)); colors.append(color)


func _upload_mesh(result: Dictionary) -> void:
	if result.is_empty():
		return
	var verts: PackedVector2Array = result.get("verts", PackedVector2Array())
	var colors: PackedColorArray = result.get("colors", PackedColorArray())
	var new_base: Vector2i = result.get("base", _mesh_base)
	if verts.is_empty():
		return
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_COLOR] = colors
	# Write into the back buffer — never touch the front mesh while GPU uses it.
	_mesh_back.clear_surfaces()
	_mesh_back.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	# Swap: point the node at the freshly written mesh, recycle the old front.
	_mesh_node.mesh = _mesh_back
	var tmp := _mesh_front
	_mesh_front = _mesh_back
	_mesh_back = tmp
	_mesh_base = new_base
	# Fix position immediately — main._process already ran this frame with the
	# old _mesh_base, so we must correct it here or get a 1-frame jump.
	var f := _world_tile_f(focus_uv)
	var dx := f.x - float(_mesh_base.x)
	var dy := f.y - float(_mesh_base.y)
	_mesh_node.position = Vector2(-(dx - dy) * tile_width * 0.5,
		-(dx + dy) * tile_height * 0.5)


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
	_request_mesh_rebuild()


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
	# v3 bitmap path: payload has color_map, property_map, data_map as raw bytes.
	# Fall back to legacy field path if bitmaps are absent (e.g. offline grid).
	if payload.has("color_map"):
		return _build_snapshot_bitmaps(payload)
	return _build_snapshot_legacy(payload)


func _build_snapshot_bitmaps(payload: Dictionary) -> Dictionary:
	# Decode the v3 three-bitmap payload. No world-model knowledge needed here:
	# color_map   -> palette lookup -> tile definition (physics, walkability, ...)
	# property_map-> height_px, water flag, structure, faction
	# data_map    -> vegetation, flow direction/speed, fauna
	var size: int = int(payload.get("size", 0))
	var seed: int = int(payload.get("seed", 0))
	var tile_world: float = max(float(payload.get("tile_world", 0.001)), 1e-6)
	var center: Dictionary = payload.get("center", {"cx": 0.5, "cy": 0.5})
	var center_uv := Vector2(float(center.get("cx", 0.5)), float(center.get("cy", 0.5)))
	var color_map: PackedByteArray = payload["color_map"]   # N*N*3 bytes RGB
	var prop_map: PackedByteArray  = payload["property_map"] # N*N*4 bytes RGBA
	var data_map: PackedByteArray  = payload["data_map"]     # N*N*4 bytes RGBA
	var grid_n: int = max(1, int(round(1.0 / tile_world)))
	var origin_x := int(floor(center_uv.x / tile_world + 0.5)) - size / 2
	var origin_y := int(floor(center_uv.y / tile_world + 0.5)) - size / 2
	var tiles: Array = []
	if size > 0:
		for gy in range(size):
			for gx in range(size):
				var i := gy * size + gx
				var ci := i * 3   # color_map offset
				var pi := i * 4   # property_map offset
				var di := i * 4   # data_map offset
				if ci + 2 >= color_map.size() or pi + 3 >= prop_map.size():
					continue

				# --- color & tile definition from palette
				var top_color := Color8(color_map[ci], color_map[ci+1], color_map[ci+2])
				var tile_def: Dictionary = _palette_lookup(top_color)
				var water: bool   = tile_def["water"]
				var foliage: bool = tile_def["foliage"]

				# --- property_map: R=height G=surface B=structure A=faction
				var height_norm := float(prop_map[pi])   / 255.0
				var surf_flag   := int(prop_map[pi + 1]) # 0=land 1=water 2=river
				var structure   := int(prop_map[pi + 2])
				var faction     := int(prop_map[pi + 3]) - 1  # -1 = unclaimed

				# Height in pixels: unified formula for land and water.
				# Python encodes land elevation for land tiles, and the
				# tide-adjusted waterline (sea_eff) for water tiles, so the
				# sea surface rises/falls with the tide.  No special case
				# needed; the same ramp works for both.
				var height_px: float = clampf(water_column_height + height_norm * relief_scale,
								water_column_height, water_column_height + relief_scale)

				# --- data_map: R=sunlight G=veg B=flow_dir A=flow_speed
				var sun   := float(data_map[di])     / 255.0 if di < data_map.size() else 0.8
				var veg   := float(data_map[di + 1]) / 255.0 if di+1 < data_map.size() else 0.0
				var f_dir := float(data_map[di + 2]) / 255.0 if di+2 < data_map.size() else 0.0
				var f_spd := float(data_map[di + 3]) / 255.0 if di+3 < data_map.size() else 0.0

				# Apply per-tile sunlight to the stable biome color.
				# color_map is now unlit (pure biome), so we darken/lighten here.
				# Sun=1 → full daylight, Sun=0 → night (dark but not black).
				var lit_color := top_color.lerp(Color(0.08, 0.10, 0.14), 1.0 - sun)
				var left_color  := lit_color.darkened(0.28 if not water else 0.10)
				var right_color := lit_color.darkened(0.16 if not water else 0.18)

				# Water flag: top face gets animated shimmer tint in _draw().
				var water_scale := 1.0 if surf_flag > 0 else 0.0

				# Tree placement: use hash on global tile coords for stability
				var gtx := origin_x + gx
				var gty := origin_y + gy
				var tree_biome := -1
				var tree_sun   := 0.8
				var tree_veg   := veg
				var canopy_color := Color.TRANSPARENT
				if structure == 0 and foliage and veg > 0.26 and _hash01(gtx, gty, seed) > 0.55:
					tree_biome = 6  # grassland index as generic foliage fallback
					canopy_color = lit_color.lightened(0.12)

				# Building wall color from faction tint
				var wall_color := Color(0.0, 0.0, 0.0, 0.0)
				if structure == 2:
					wall_color = lit_color.lerp(Color8(228, 214, 186), 0.55)
					if faction >= 0:
						var fi := faction % 6
						const FAC := [Color8(214,69,65),Color8(232,184,58),Color8(58,176,168),
									  Color8(150,96,210),Color8(232,128,52),Color8(96,178,84)]
						wall_color = FAC[fi].lerp(Color8(228, 214, 186), 0.58)

				var key := Vector2i(((gtx % grid_n) + grid_n) % grid_n,
					((gty % grid_n) + grid_n) % grid_n)
				tiles.append({
					"key":          key,
					"height":       height_norm,
					"height_px":    height_px,
					"top":          lit_color,
					"left":         left_color,
					"right":        right_color,
					"water_scale":  water_scale,
					"tree_biome":   tree_biome,
					"tree_sun":     tree_sun,
					"tree_veg":     tree_veg,
					"canopy_color": canopy_color,
					"structure":    structure,
					"wall":         wall_color,
					"env": {
						"biome":     tile_def["name"],
						"water":     surf_flag == 1,
						"river":     surf_flag == 2,
						"flow":      Vector2(cos(f_dir * TAU), sin(f_dir * TAU)) * f_spd,
						"sunlight":  sun,
						"vegetation":veg,
						"structure": structure,
						"faction":   faction,
						"height":    height_norm,
						"walkable":  tile_def["walkable"],
						"physics":   tile_def["physics"],
						"resource":  tile_def["resource"],
					},
				})
	return {
		"size":         size,
		"seed":         seed,
		"sea_effective": float(payload.get("sea_effective", 0.42)),
		"tile_world":   tile_world,
		"center_uv":    center_uv,
		"time_days":    float(payload.get("time_days", 0.0)),
		"sunlight_mean":int(payload.get("sunlight_mean", 180)),
		"cloud_mean":   int(payload.get("cloud_mean", 110)),
		"tiles":        tiles,
	}


func _build_snapshot_legacy(payload: Dictionary) -> Dictionary:
	# Legacy field-based decode kept for offline-grid fallback.
	var fields: Dictionary = payload.get("fields", {})
	var size: int = int(payload.get("size", 0))
	var seed: int = int(payload.get("seed", 0))
	var sea_effective: float = float(payload.get("sea_effective", 0.42))
	var tile_world: float = max(float(payload.get("tile_world", 0.001)), 1e-6)
	var center: Dictionary = payload.get("center", {"cx": 0.5, "cy": 0.5})
	var center_uv: Vector2 = Vector2(float(center.get("cx", 0.5)), float(center.get("cy", 0.5)))
	var heights = fields.get("height", PackedByteArray())
	var biomes  = fields.get("biome_id", PackedByteArray())
	var sunlight = fields.get("sunlight", PackedByteArray())
	var grid_n: int = max(1, int(round(1.0 / tile_world)))
	var origin_x := int(floor(center_uv.x / tile_world + 0.5)) - size / 2
	var origin_y := int(floor(center_uv.y / tile_world + 0.5)) - size / 2
	var tiles: Array = []
	if size > 0:
		for gy in range(size):
			for gx in range(size):
				var index := gy * size + gx
				if not _has_field_u8(heights, index):
					continue
				var biome_id := 6  # grassland default
				if _has_field_s16(biomes, index):
					biome_id = clampi(_field_s16(biomes, index), 0, BIOME_INFO.size() - 1)
				var info: Dictionary = BIOME_INFO[biome_id]
				var water: bool = info["water"]
				var sun := float(_field_u8(sunlight, index, 199)) / 255.0
				var elev := float(_field_u8(heights, index)) / 255.0
				var height_px := _column_height_px_for(elev, sea_effective)
				var top_color: Color = info["base"]
				var left_color := top_color.darkened(0.28 if not water else 0.10)
				var right_color := top_color.darkened(0.16 if not water else 0.18)
				var gtx := origin_x + gx
				var gty := origin_y + gy
				var key := Vector2i(((gtx % grid_n) + grid_n) % grid_n,
					((gty % grid_n) + grid_n) % grid_n)
				tiles.append({
					"key": key, "height": elev, "height_px": height_px,
					"top": top_color, "left": left_color, "right": right_color,
					"water_scale": 0.94 if water else 0.0,
						"tree_biome": -1, "tree_sun": sun, "tree_veg": 0.0,
						"canopy_color": Color.TRANSPARENT,
					"structure": 0, "wall": Color(0,0,0,0), "env": {},
				})
	return {
		"size": size, "seed": seed, "sea_effective": sea_effective,
		"tile_world": tile_world, "center_uv": center_uv,
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
	if version != FRAME_PACKET_VERSION:
		return {}
	var size := int(body.decode_u16(6))
	var seed_v := int(body.decode_u32(8))
	var planet_size := int(body.decode_u16(12))
	var target_tiles := int(body.decode_u16(14))
	var cells := size * size
	var offset := FRAME_PACKET_HEADER_BYTES
	# v3: three packed bitmaps
	var color_map   := _copy_bytes(body, offset, cells * 3);  offset += cells * 3
	var prop_map    := _copy_bytes(body, offset, cells * 4);  offset += cells * 4
	var data_map    := _copy_bytes(body, offset, cells * 4)
	if color_map.size() != cells * 3 or prop_map.size() != cells * 4:
		return {}
	return {
		"seed":          seed_v,
		"planet_size":   planet_size,
		"view_tiles_target": target_tiles,
		"size":          size,
		"time_days":     float(body.decode_float(16)),
		"span":          float(body.decode_float(20)),
		"center": {
			"cx": float(body.decode_float(24)),
			"cy": float(body.decode_float(28)),
		},
		"tile_world":    float(body.decode_float(32)),
		"sea_level":     float(body.decode_float(36)),
		"sea_effective": float(body.decode_float(40)),
		"sunlight_mean": int(body[44]),
		"cloud_mean":    int(body[45]),
		"color_map":     color_map,
		"property_map":  prop_map,
		"data_map":      data_map,
	}


func _wrapped_delta(value: float, center: float) -> float:
	return fposmod(value - center + 0.5, 1.0) - 0.5


func _column_height_px_for(elev: float, sea_effective: float) -> float:
	if elev < sea_effective:
		return water_column_height
	return clampf(10.0 + (elev - sea_effective) * relief_scale, 8.0, 44.0)


func _hash01(x: int, y: int, salt: int) -> float:
	var value := sin(float(x * 127 + y * 311 + salt * 29)) * 43758.5453
	return absf(fmod(value, 1.0))
