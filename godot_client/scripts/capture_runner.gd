extends SceneTree

const DEFAULT_POINTS := 10
const WAIT_TIMEOUT_SECONDS := 4.0
const WAIT_TIMEOUT_MSEC := int(WAIT_TIMEOUT_SECONDS * 1000.0)
const MAIN_SCENE_PATH := "res://scenes/main.tscn"

var _output_dir := ""
var _points_path := ""
var _width := 1920
var _height := 1080


func _initialize() -> void:
	_parse_args()
	if _output_dir == "":
		push_error("capture_runner: missing --output-dir")
		quit(2)
		return

	DirAccess.make_dir_recursive_absolute(_output_dir)
	get_root().size = Vector2i(_width, _height)

	var scene := load(MAIN_SCENE_PATH)
	var main = scene.instantiate()
	get_root().add_child(main)
	await process_frame
	await process_frame

	var points := _load_points()
	if points.is_empty():
		push_error("capture_runner: no points provided")
		quit(2)
		return

	var index := 0
	for p in points:
		var cx := float(p.get("cx", 0.5))
		var cy := float(p.get("cy", 0.5))
		main.teleport_to(Vector2(cx, cy), true)
		var ok := await _wait_for_frame(main)
		if not ok:
			push_warning("capture_runner: timed out waiting for payload at #%d" % index)
		await process_frame
		await process_frame
		var tex := get_root().get_texture()
		var image: Image = tex.get_image() if tex != null else null
		if image == null:
			push_warning("capture_runner: viewport capture unavailable at #%d" % index)
			continue
		var frame_name := "frame_%02d.png" % index
		var path := _output_dir.path_join(frame_name)
		image.save_png(path)
		index += 1
		print("captured %s %.5f %.5f" % [frame_name, cx, cy])
	quit(0)


func _wait_for_frame(main: Node) -> bool:
	var deadline := Time.get_ticks_msec() + WAIT_TIMEOUT_MSEC
	while Time.get_ticks_msec() < deadline:
		if main.has_live_payload() and not main.is_request_pending():
			return true
		await process_frame
	return false


func _load_points() -> Array:
	var points: Array = []
	if _points_path == "":
		for i in range(DEFAULT_POINTS):
			points.append({"cx": randf(), "cy": randf(), "kind": "random"})
		return points

	var file := FileAccess.open(_points_path, FileAccess.READ)
	if file == null:
		return points
	var parsed = JSON.parse_string(file.get_as_text())
	if parsed is Array:
		for item in parsed:
			if item is Dictionary:
				points.append(item)
	return points


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		match args[i]:
			"--output-dir":
				if i + 1 < args.size():
					_output_dir = args[i + 1]
					i += 2
					continue
			"--points-json":
				if i + 1 < args.size():
					_points_path = args[i + 1]
					i += 2
					continue
			"--width":
				if i + 1 < args.size():
					_width = int(args[i + 1])
					i += 2
					continue
			"--height":
				if i + 1 < args.size():
					_height = int(args[i + 1])
					i += 2
					continue
		i += 1
