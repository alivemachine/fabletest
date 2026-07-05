extends Node2D

const TerrainChunkScene = preload("res://scripts/terrain_chunk.gd")
const PixelArt = preload("res://scripts/pixel_art.gd")

var bridge_url := "http://127.0.0.1:8765/frame.bin"
var seed := 42
var world_size := 192
var civ_count := 3
var view_tiles := 24
var stream_tiles := 64
var zoom := 12.0
var sim_speed := 0.35
var sea_level := 0.42
var season_amp := 0.18
var tide_amp := 0.012
var day_night := 0.65
var playing := true

# Offline test mode: draw a static grey grid and do zero fetching, so movement
# smoothness can be verified in isolation. Press G to toggle at runtime.
var offline_grid := false

var world_uv := Vector2(0.5, 0.5)
var move_speed_tiles := 8.5
var request_accum := 0.0
var request_in_flight := false
var refresh_after_response := false
var refresh_reset_pending := false
var walk_phase := 0.0
var status_line := "waiting for bridge"
var last_move_input := Vector2.ZERO
var prefetch_center_uv := Vector2(0.5, 0.5)
var stream_center_uv := Vector2(0.5, 0.5)
var stream_center_valid := false
var time_refresh_clock := 0.0
var time_refresh_interval := 0.25

var terrain: Node2D
var player_root: Node2D
var player_stack: Node2D
var player_shadow: Sprite2D
var hud_label: Label
var http: HTTPRequest
var sky_rect: ColorRect
var backdrop_layers: Array = []


func _ready() -> void:
    Engine.max_fps = 120
    _build_environment()
    _build_terrain()
    _build_player()
    _build_backdrop()
    _build_hud()
    _build_http()
    _layout_scene()
    prefetch_center_uv = world_uv
    if offline_grid:
        # Start on the static grey grid so smoothness can be verified before any
        # networking happens. Press G to switch to the live streamed world.
        if terrain.has_method("set_offline_grid"):
            terrain.set_offline_grid(true, stream_tiles)
    else:
        _request_frame(0.0, false)


func _process(delta: float) -> void:
    var move_input := _movement_input()
    last_move_input = move_input
    if move_input.length() > 0.01:
        var tile_world := _tile_world_size()
        var forward := Vector2(0.0, -1.0)
        var right := Vector2(1.0, 0.0)
        var world_move := (right * move_input.x + forward * move_input.y).normalized()
        world_uv += world_move * move_speed_tiles * tile_world * delta
        world_uv.x = wrapf(world_uv.x, 0.0, 1.0)
        world_uv.y = wrapf(world_uv.y, 0.0, 1.0)
        walk_phase += delta * 10.0
    else:
        walk_phase = 0.0

    player_stack.position.y = -30.0 + 2.5 * sin(walk_phase)

    request_accum += delta
    time_refresh_clock += delta

    # Offline grid mode: never fetch anything. Pure movement test on a static
    # grey grid to prove the render/movement loop is smooth on its own.
    if offline_grid:
        _layout_scene()
        _update_backdrop()
        _update_hud()
        return

    # Movement never triggers a fetch directly. The player glides continuously
    # inside the already-loaded chunk (the terrain slides every frame via
    # focus_screen_offset). We only stream a NEW chunk when the player nears the
    # buffered edge, and we let time-evolution refresh on a slow timer that
    # reuses the SAME sticky center so it never shifts the map under you.
    var have_chunk: bool = terrain != null and terrain.has_method("has_payload") and terrain.has_payload()
    var drift: float = terrain.focus_radius_tiles(world_uv) if have_chunk else 0.0
    var chunk_size: int = terrain.stream_size() if have_chunk else stream_tiles
    var buffer_tiles: float = max(0.0, (float(chunk_size) - float(view_tiles)) * 0.5)
    var edge_threshold: float = max(4.0, buffer_tiles * 0.6)

    var should_request := false
    var request_center := stream_center_uv
    if not have_chunk:
        # First load (or after reset): center on the player.
        should_request = true
        request_center = world_uv
    elif drift > edge_threshold:
        # Player is near the loaded edge: recenter the crop ahead of them.
        should_request = true
        request_center = _stream_request_center()
    elif playing and time_refresh_clock >= time_refresh_interval:
        # Time evolution only: refetch the SAME window so the sim advances but
        # the map does not move under the player.
        should_request = true
        request_center = stream_center_uv

    if should_request and not request_in_flight:
        if playing and time_refresh_clock >= time_refresh_interval:
            time_refresh_clock = 0.0
        _request_frame(request_accum, false, request_center)
        request_accum = 0.0

    _layout_scene()
    _update_backdrop()
    _update_hud()


func _unhandled_input(event: InputEvent) -> void:
    if event is InputEventKey and event.pressed and not event.echo:
        match event.keycode:
            KEY_G:
                offline_grid = not offline_grid
                if terrain != null and terrain.has_method("set_offline_grid"):
                    terrain.set_offline_grid(offline_grid, stream_tiles)
                if not offline_grid:
                    _request_frame(0.0, false, world_uv)
            KEY_SPACE:
                playing = not playing
                _request_frame(0.0, false)
            KEY_R:
                _request_frame(0.0, true)
            KEY_BRACKETLEFT:
                zoom = max(6.0, zoom * 0.88)
                _request_frame(0.0, false)
            KEY_BRACKETRIGHT:
                zoom = min(24.0, zoom * 1.14)
                _request_frame(0.0, false)


func _build_environment() -> void:
    sky_rect = ColorRect.new()
    sky_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
    sky_rect.color = Color8(36, 56, 78)
    sky_rect.set_anchors_preset(Control.PRESET_FULL_RECT)
    sky_rect.z_index = -1000
    add_child(sky_rect)


func _build_terrain() -> void:
    terrain = TerrainChunkScene.new()
    terrain.name = "Terrain"
    terrain.z_index = 20
    add_child(terrain)


func _build_player() -> void:
    player_root = Node2D.new()
    player_root.name = "Player"
    player_root.z_index = 40
    add_child(player_root)

    player_shadow = Sprite2D.new()
    player_shadow.texture = PixelArt.make_shadow_texture()
    player_shadow.centered = true
    player_shadow.scale = Vector2(1.15, 0.78)
    player_shadow.position = Vector2(0.0, 6.0)
    player_shadow.modulate = Color(0.0, 0.0, 0.0, 0.30)
    player_root.add_child(player_shadow)

    player_stack = Node2D.new()
    player_stack.name = "Stack"
    player_stack.position = Vector2(0.0, -30.0)
    player_root.add_child(player_stack)

    var marker := Polygon2D.new()
    marker.polygon = PackedVector2Array([
        Vector2(0.0, -88.0),
        Vector2(-10.0, -72.0),
        Vector2(0.0, -64.0),
        Vector2(10.0, -72.0),
    ])
    marker.color = Color8(252, 214, 92)
    player_root.add_child(marker)

    var avatar := PixelArt.make_avatar_texture(Color8(76, 128, 224), Color8(52, 62, 90), Color8(241, 207, 168))
    for layer in range(5):
        var sprite := Sprite2D.new()
        sprite.texture = avatar
        sprite.centered = true
        sprite.position = Vector2(0.0, -float(layer) * 2.0)
        sprite.scale = Vector2(3.4, 3.4)
        player_stack.add_child(sprite)


func _build_backdrop() -> void:
    _add_backdrop_layer(
        PixelArt.make_band_texture(256, 96, Color8(76, 118, 156), Color8(102, 132, 148), Color8(45, 62, 82)),
        0.30, 9.0, Color8(128, 156, 180), 1.24
    )
    _add_backdrop_layer(
        PixelArt.make_cloud_texture(256, 64, Color8(236, 242, 248), Color8(170, 182, 196)),
        0.18, 15.0, Color8(220, 228, 238), 1.42
    )
    _add_backdrop_layer(
        PixelArt.make_band_texture(256, 80, Color8(118, 152, 146), Color8(92, 118, 112), Color8(54, 74, 78)),
        0.46, 24.0, Color8(128, 168, 148), 1.10
    )


func _add_backdrop_layer(texture: Texture2D, y_ratio: float, factor: float, tint: Color, scale_factor: float) -> void:
    var sprite := Sprite2D.new()
    sprite.texture = texture
    sprite.centered = true
    sprite.modulate = tint
    sprite.z_index = -400 + backdrop_layers.size()
    add_child(sprite)
    backdrop_layers.append({
        "sprite": sprite,
        "y_ratio": y_ratio,
        "factor": factor,
        "tint": tint,
        "scale_factor": scale_factor,
    })


func _build_hud() -> void:
    var canvas := CanvasLayer.new()
    add_child(canvas)
    hud_label = Label.new()
    hud_label.position = Vector2(16.0, 14.0)
    hud_label.modulate = Color(0.96, 0.96, 0.96)
    canvas.add_child(hud_label)


func _build_http() -> void:
    http = HTTPRequest.new()
    http.timeout = 4.0
    http.use_threads = true
    add_child(http)
    http.request_completed.connect(_on_request_completed)


func _movement_input() -> Vector2:
    var x := 0.0
    var y := 0.0
    if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
        x -= 1.0
    if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
        x += 1.0
    if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
        y -= 1.0
    if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
        y += 1.0
    return Vector2(x, y).normalized()


func _tile_world_size() -> float:
    if terrain != null and terrain.has_method("tile_world") and terrain.has_payload():
        return terrain.tile_world()
    return 1.0 / (zoom * float(view_tiles))


func _request_frame(dt: float, reset: bool, center_override = null) -> void:
    if http == null:
        return
    if request_in_flight:
        refresh_after_response = true
        refresh_reset_pending = refresh_reset_pending or reset
        return
    var center_uv: Vector2 = center_override if center_override is Vector2 else world_uv
    prefetch_center_uv = center_uv
    var query := {
        "seed": seed,
        "size": world_size,
        "civ_count": civ_count,
        "cx": snappedf(center_uv.x, 0.0001),
        "cy": snappedf(center_uv.y, 0.0001),
        "zoom": _stream_zoom(),
        "view_tiles": stream_tiles,
        "dt": dt,
        "speed": sim_speed,
        "playing": 1 if playing else 0,
        "sea_level": sea_level,
        "season_amp": season_amp,
        "tide_amp": tide_amp,
        "day_night": day_night,
        "reset": 1 if reset else 0,
    }
    var url := bridge_url + "?" + _encode_query(query)
    var err := http.request(url)
    if err != OK:
        status_line = "request failed: %s" % err
        return
    request_in_flight = true
    if reset:
        request_accum = 0.0


func _stream_request_center() -> Vector2:
    var center := world_uv
    var input_len := last_move_input.length()
    if input_len <= 0.01:
        return center
    var tile_world := _tile_world_size()
    var forward := Vector2(0.0, -1.0)
    var right := Vector2(1.0, 0.0)
    var world_move := (right * last_move_input.x + forward * last_move_input.y).normalized()
    var chunk_size: int = terrain.stream_size() if terrain != null and terrain.has_payload() else stream_tiles
    var buffer_tiles: float = max(0.0, (float(chunk_size) - float(view_tiles)) * 0.5)
    var look_ahead_tiles: float = clampf(buffer_tiles * 0.22, 0.0, 4.0)
    center += world_move * tile_world * look_ahead_tiles
    return Vector2(wrapf(center.x, 0.0, 1.0), wrapf(center.y, 0.0, 1.0))


func _stream_zoom() -> float:
    return zoom * float(view_tiles) / float(max(stream_tiles, view_tiles))


func _encode_query(query: Dictionary) -> String:
    var parts: PackedStringArray = []
    for key in query.keys():
        parts.append(str(key).uri_encode() + "=" + str(query[key]).uri_encode())
    return "&".join(parts)


func _on_request_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
    request_in_flight = false
    if response_code != 200:
        status_line = "bridge offline or invalid response (%d)" % response_code
        if refresh_after_response:
            var pending_reset := refresh_reset_pending
            refresh_after_response = false
            refresh_reset_pending = false
            _request_frame(0.0, pending_reset)
        return
    terrain.apply_payload_bytes(body)
    # Remember the window we actually loaded. This is the sticky stream center:
    # the player can now glide anywhere inside this buffered chunk without any
    # further fetch until they approach its edge.
    stream_center_uv = prefetch_center_uv
    stream_center_valid = true
    status_line = "bridge ok"
    if refresh_after_response:
        var pending_reset_ok := refresh_reset_pending
        refresh_after_response = false
        refresh_reset_pending = false
        _request_frame(0.0, pending_reset_ok)


func _apply_payload_tint() -> void:
    var sun := float(terrain.sunlight_mean_byte()) / 255.0 if terrain != null and terrain.has_payload() else 0.7
    var cloud := float(terrain.cloud_mean_byte()) / 255.0 if terrain != null and terrain.has_payload() else 0.43
    if sky_rect != null:
        sky_rect.color = Color8(30, 46, 64).lerp(Color8(128, 170, 205), sun * 0.9) * Color(1.0 - cloud * 0.06, 1.0 - cloud * 0.04, 1.0, 1.0)


func _layout_scene() -> void:
    var viewport_size := get_viewport_rect().size
    if terrain == null:
        return
    # The player stays pinned to a fixed screen anchor; the terrain layer moves
    # underneath it. The terrain node owns its own placement (see set_focus /
    # _update_placement) so tile data and position always update atomically.
    var anchor := Vector2(viewport_size.x * 0.5, viewport_size.y * 0.60)
    terrain.set_focus(world_uv, anchor)
    if terrain.has_payload():
        player_root.position = anchor + Vector2(0.0, -terrain.ground_height_for_focus(world_uv) - 6.0)
    else:
        player_root.position = anchor + Vector2(0.0, -18.0)


func _update_backdrop() -> void:
    var viewport_size := get_viewport_rect().size
    _apply_payload_tint()
    var sun := float(terrain.sunlight_mean_byte()) / 255.0 if terrain != null and terrain.has_payload() else 0.7
    for layer in backdrop_layers:
        var sprite: Sprite2D = layer["sprite"]
        var y_ratio := float(layer["y_ratio"])
        var factor := float(layer["factor"])
        var scale_factor := float(layer["scale_factor"])
        var tint: Color = layer["tint"]
        var tex_size := sprite.texture.get_size()
        var fit_scale: float = max(viewport_size.x / max(tex_size.x, 1.0), viewport_size.y / max(tex_size.y, 1.0)) * scale_factor
        sprite.scale = Vector2(fit_scale, fit_scale)
        sprite.position = Vector2(
            viewport_size.x * 0.5 + sin(world_uv.x * TAU * factor) * 14.0,
            viewport_size.y * y_ratio + cos(world_uv.y * TAU * factor * 0.5) * 8.0
        )
        sprite.modulate = tint.lerp(Color.WHITE, sun * 0.22) * Color(1.0, 1.0, 1.0, 0.84)


func _update_hud() -> void:
    if hud_label == null:
        return
    var day: float = terrain.time_days() if terrain != null and terrain.has_payload() else 0.0
    var chunk: int = terrain.stream_size() if terrain != null and terrain.has_payload() else stream_tiles
    var mode_line: String = "OFFLINE GRID (no fetching)" if offline_grid else status_line
    hud_label.text = "Fabletest Godot client\n" \
        + "WASD move  |  [ ] zoom  |  Space pause  |  R reset  |  G grid test\n" \
        + "seed %d  |  day %.1f  |  view %d / chunk %d  |  zoom %.1fx  |  %s" % [seed, day, view_tiles, chunk, zoom, mode_line] 