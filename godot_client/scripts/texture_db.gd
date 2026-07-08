extends Node
## Supabase-backed texture database with a local disk cache.
##
## The generation pipeline publishes the sprite/texture database to a public
## Supabase Storage bucket:
##
##     <base_url>/index.json                 manifest of every ready key
##     <base_url>/img/<key_hash>_v<i>.png    one file per variation
##
## This node downloads the manifest once, then serves textures on demand:
## disk cache hit -> instant; miss -> HTTP fetch, save to
## user://texture_cache/<generation>/, then load. Old generations are
## dropped when the manifest's generated_at stamp changes, so regenerated
## art (same filenames, new content) is never served stale.
##
## Usage:
##     var db := preload("res://scripts/texture_db.gd").new()
##     add_child(db)
##     db.index_loaded.connect(_on_index)
##     db.load_index("https://<ref>.supabase.co/storage/v1/object/public/textures")
##     db.get_texture("7ca36f30f2106c9e", 0, func(tex): sprite.texture = tex)

signal index_loaded(count: int)
signal index_failed(reason: String)

const CACHE_ROOT := "user://texture_cache"

var base_url := ""
var generation := ""          ## generated_at stamp of the loaded manifest
var assets := {}              ## key_hash -> {key, subject, lod, tags, files}
var _pending := {}            ## url -> [callbacks] (dedupes concurrent asks)

func load_index(url: String) -> void:
	base_url = url.trim_suffix("/")
	_fetch(base_url + "/index.json", func(body: PackedByteArray):
		if body.is_empty():
			index_failed.emit("index.json unreachable")
			return
		var idx = JSON.parse_string(body.get_string_from_utf8())
		if idx == null or not (idx is Dictionary):
			index_failed.emit("index.json unparsable")
			return
		if idx.has("base_url"):
			base_url = String(idx["base_url"]).trim_suffix("/")
		generation = String(idx.get("generated_at", "unknown"))
		assets.clear()
		for a in idx.get("assets", []):
			assets[a["hash"]] = a
		_evict_stale_generations()
		index_loaded.emit(assets.size())
	)

## Fetch one variation as a Texture2D. cb receives Texture2D or null.
func get_texture(key_hash: String, variation: int, cb: Callable) -> void:
	var a = assets.get(key_hash)
	if a == null or variation >= a["files"].size():
		cb.call(null)
		return
	var fname: String = a["files"][variation]
	var local := _cache_dir() + "/" + fname
	if FileAccess.file_exists(local):
		cb.call(_texture_from_file(local))
		return
	_fetch(base_url + "/img/" + fname, func(body: PackedByteArray):
		if body.is_empty():
			cb.call(null)
			return
		DirAccess.make_dir_recursive_absolute(_cache_dir())
		var f := FileAccess.open(local, FileAccess.WRITE)
		if f:
			f.store_buffer(body)
			f.close()
		cb.call(_texture_from_buffer(body))
	)

## Every hash for a subject ("tree.oak", "ground.taiga", ...) — lets game
## code pick the best-matching key by tags without a second query.
func hashes_for_subject(subject: String) -> Array:
	var out: Array = []
	for h in assets:
		if assets[h]["subject"] == subject:
			out.append(h)
	return out

# -- internals ----------------------------------------------------------------

func _cache_dir() -> String:
	# one directory per manifest generation; stale ones are deleted
	return CACHE_ROOT + "/" + generation.sha1_text().substr(0, 12)

func _evict_stale_generations() -> void:
	var keep := _cache_dir().get_file()
	var dir := DirAccess.open(CACHE_ROOT)
	if dir == null:
		return
	dir.list_dir_begin()
	var name := dir.get_next()
	while name != "":
		if dir.current_is_dir() and name != keep and not name.begins_with("."):
			_remove_dir_recursive(CACHE_ROOT + "/" + name)
		name = dir.get_next()
	dir.list_dir_end()

func _remove_dir_recursive(path: String) -> void:
	var dir := DirAccess.open(path)
	if dir == null:
		return
	dir.list_dir_begin()
	var name := dir.get_next()
	while name != "":
		if not name.begins_with("."):
			if dir.current_is_dir():
				_remove_dir_recursive(path + "/" + name)
			else:
				dir.remove(name)
		name = dir.get_next()
	dir.list_dir_end()
	DirAccess.remove_absolute(path)

func _texture_from_file(path: String) -> Texture2D:
	var img := Image.load_from_file(path)
	return null if img == null else ImageTexture.create_from_image(img)

func _texture_from_buffer(body: PackedByteArray) -> Texture2D:
	var img := Image.new()
	return null if img.load_png_from_buffer(body) != OK \
		else ImageTexture.create_from_image(img)

func _fetch(url: String, cb: Callable) -> void:
	if _pending.has(url):          # already in flight: piggyback
		_pending[url].append(cb)
		return
	_pending[url] = [cb]
	var req := HTTPRequest.new()
	req.timeout = 30.0
	req.use_threads = true
	add_child(req)
	req.request_completed.connect(
		func(_result: int, code: int, _headers: PackedStringArray, body: PackedByteArray):
			var callbacks: Array = _pending.get(url, [])
			_pending.erase(url)
			req.queue_free()
			var payload := body if code == 200 else PackedByteArray()
			for c in callbacks:
				c.call(payload)
	)
	if req.request(url) != OK:
		var callbacks: Array = _pending.get(url, [])
		_pending.erase(url)
		req.queue_free()
		for c in callbacks:
			c.call(PackedByteArray())
