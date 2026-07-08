extends RefCounted

# ---------------------------------------------------------------------------
# SPRITE ATLAS — every world sprite (trees, buildings, fauna, resources) lives
# in one generated texture so the whole terrain renders as a single textured
# ArrayMesh. Cell 0 is a solid white block: terrain faces UV into it, so plain
# vertex-color quads and textured sprites share one surface and one draw call.
# Index constants below are the atlas cell ids terrain_chunk.gd keys on.
# ---------------------------------------------------------------------------
const CELL := 32
const ATLAS_COLS := 8

const SPR_WHITE := 0
const SPR_OAK := 1
const SPR_PINE := 2
const SPR_PALM := 3
const SPR_CACTUS := 4
const SPR_ACACIA := 5
const SPR_JUNGLE := 6
const SPR_FIR := 7
const SPR_BUSH := 8
const SPR_HOUSE := 9
const SPR_DEER := 10
const SPR_SHEEP := 11
const SPR_CAMEL := 12
const SPR_WOLF := 13
const SPR_LION := 14
const SPR_FISH := 15
const SPR_WHEAT := 16
const SPR_BERRY := 17
const SPR_ROCK := 18
const SPR_ORE := 19
const SPR_COUNT := 20


static func _to_texture(image: Image) -> Texture2D:
	return ImageTexture.create_from_image(image)


# Filled ellipse blob with light top / dark bottom shading and speckle.
static func _blob(img: Image, ox: int, oy: int, cx: float, cy: float,
		rx: float, ry: float, color: Color) -> void:
	for y in range(CELL):
		for x in range(CELL):
			var dx := (float(x) - cx) / maxf(rx, 0.5)
			var dy := (float(y) - cy) / maxf(ry, 0.5)
			if dx * dx + dy * dy > 1.0:
				continue
			var c := color
			if dy < -0.35:
				c = c.lightened(0.14)
			elif dy > 0.4:
				c = c.darkened(0.16)
			if (x * 5 + y * 3) % 7 == 0:
				c = c.darkened(0.10)
			img.set_pixel(ox + x, oy + y, c)


static func _rect(img: Image, ox: int, oy: int, x0: int, y0: int,
		x1: int, y1: int, color: Color) -> void:
	for y in range(y0, y1 + 1):
		for x in range(x0, x1 + 1):
			if x >= 0 and x < CELL and y >= 0 and y < CELL:
				img.set_pixel(ox + x, oy + y, color)


# One quadruped body plan reused for every land animal: body blob, legs,
# head; per-species extras are painted by the caller.
static func _animal(img: Image, ox: int, oy: int, body: Color, head_dx: int = 7) -> void:
	_blob(img, ox, oy, 16.0, 18.0, 8.0, 5.0, body)
	var leg := body.darkened(0.28)
	_rect(img, ox, oy, 11, 22, 12, 27, leg)
	_rect(img, ox, oy, 19, 22, 20, 27, leg)
	_blob(img, ox, oy, float(16 + head_dx), 12.0, 3.5, 3.0, body.lightened(0.05))


static func make_sprite_atlas() -> Dictionary:
	var rows := (SPR_COUNT + ATLAS_COLS - 1) / ATLAS_COLS
	var img := Image.create(ATLAS_COLS * CELL, rows * CELL, false, Image.FORMAT_RGBA8)
	img.fill(Color(0, 0, 0, 0))
	for id in range(SPR_COUNT):
		var ox := (id % ATLAS_COLS) * CELL
		var oy := (id / ATLAS_COLS) * CELL
		_draw_sprite(img, id, ox, oy)
	# Normalized UV rects, inset half a texel so sampling never bleeds between cells.
	var uvs: Array = []
	var w := float(img.get_width())
	var h := float(img.get_height())
	for id in range(SPR_COUNT):
		var ox := float((id % ATLAS_COLS) * CELL)
		var oy := float((id / ATLAS_COLS) * CELL)
		uvs.append(Rect2((ox + 0.5) / w, (oy + 0.5) / h,
			(CELL - 1.0) / w, (CELL - 1.0) / h))
	# White cell: collapse its UV rect to the cell center so terrain quads of
	# any shape sample a single solid texel.
	var wc: Rect2 = uvs[SPR_WHITE]
	uvs[SPR_WHITE] = Rect2(wc.position + wc.size * 0.5, Vector2.ZERO)
	return {"texture": _to_texture(img), "uv": uvs}


static func _draw_sprite(img: Image, id: int, ox: int, oy: int) -> void:
	var trunk := Color8(92, 64, 44)
	match id:
		SPR_WHITE:
			_rect(img, ox, oy, 0, 0, CELL - 1, CELL - 1, Color.WHITE)
		SPR_OAK:
			_rect(img, ox, oy, 14, 18, 17, 29, trunk)
			_blob(img, ox, oy, 16.0, 11.0, 11.0, 9.5, Color8(88, 152, 76))
		SPR_PINE:
			_rect(img, ox, oy, 15, 24, 16, 29, trunk.darkened(0.15))
			for tier in range(3):
				var ty := 8 + tier * 6
				var half := 4 + tier * 3
				for y in range(6):
					var wdt := int(float(half) * float(y + 1) / 6.0)
					_rect(img, ox, oy, 16 - wdt, ty + y, 15 + wdt, ty + y, Color8(52, 110, 78))
		SPR_PALM:
			for i in range(12):
				_rect(img, ox, oy, 15 + i / 5, 17 + i, 16 + i / 5, 17 + i, Color8(148, 112, 72))
			for f in range(5):
				var a := -PI + float(f) * PI * 0.25
				for r in range(9):
					var fx := 15 + int(cos(a) * r)
					var fy := 16 + int(sin(a) * r * 0.55)
					_rect(img, ox, oy, fx, fy, fx + 1, fy, Color8(96, 168, 88))
		SPR_CACTUS:
			var green := Color8(96, 158, 92)
			_rect(img, ox, oy, 14, 10, 17, 29, green)
			_rect(img, ox, oy, 9, 14, 13, 16, green.darkened(0.06))
			_rect(img, ox, oy, 9, 10, 11, 16, green.darkened(0.06))
			_rect(img, ox, oy, 18, 17, 22, 19, green.darkened(0.10))
			_rect(img, ox, oy, 20, 13, 22, 19, green.darkened(0.10))
		SPR_ACACIA:
			_rect(img, ox, oy, 15, 15, 16, 29, trunk)
			_rect(img, ox, oy, 12, 14, 19, 16, trunk)
			_blob(img, ox, oy, 16.0, 10.0, 13.0, 4.5, Color8(122, 158, 74))
		SPR_JUNGLE:
			_rect(img, ox, oy, 14, 16, 17, 29, trunk.darkened(0.10))
			_blob(img, ox, oy, 10.0, 12.0, 8.0, 6.5, Color8(48, 128, 68))
			_blob(img, ox, oy, 22.0, 11.0, 8.0, 6.5, Color8(56, 138, 72))
			_blob(img, ox, oy, 16.0, 7.0, 9.0, 6.0, Color8(66, 150, 80))
		SPR_FIR:
			_rect(img, ox, oy, 15, 25, 16, 29, trunk.darkened(0.25))
			for tier in range(4):
				var ty := 5 + tier * 5
				var half := 3 + tier * 3
				for y in range(5):
					var wdt := int(float(half) * float(y + 1) / 5.0)
					_rect(img, ox, oy, 16 - wdt, ty + y, 15 + wdt, ty + y, Color8(34, 82, 58))
		SPR_BUSH:
			_blob(img, ox, oy, 16.0, 22.0, 9.0, 6.5, Color8(96, 148, 82))
		SPR_HOUSE:
			_rect(img, ox, oy, 8, 16, 23, 28, Color8(226, 212, 184))     # walls
			_rect(img, ox, oy, 8, 16, 23, 17, Color8(196, 180, 150))     # eave shade
			for y in range(8):                                            # roof
				_rect(img, ox, oy, 8 + y, 8 + y, 23 - y, 8 + y, Color8(164, 92, 70))
			_rect(img, ox, oy, 14, 22, 17, 28, Color8(110, 78, 56))      # door
			_rect(img, ox, oy, 19, 20, 21, 22, Color8(96, 128, 160))     # window
		SPR_DEER:
			_animal(img, ox, oy, Color8(150, 108, 70), 7)
			_rect(img, ox, oy, 22, 6, 22, 10, Color8(96, 70, 46))        # antlers
			_rect(img, ox, oy, 25, 6, 25, 10, Color8(96, 70, 46))
			_rect(img, ox, oy, 22, 6, 25, 6, Color8(96, 70, 46))
		SPR_SHEEP:
			_animal(img, ox, oy, Color8(228, 226, 218), 7)
			_blob(img, ox, oy, 23.0, 12.0, 3.0, 2.6, Color8(70, 62, 58)) # dark face
		SPR_CAMEL:
			_animal(img, ox, oy, Color8(196, 160, 104), 8)
			_blob(img, ox, oy, 14.0, 12.0, 4.0, 3.5, Color8(196, 160, 104))  # hump
		SPR_WOLF:
			_animal(img, ox, oy, Color8(126, 128, 134), 8)
			_rect(img, ox, oy, 7, 16, 9, 17, Color8(126, 128, 134))      # tail
			_rect(img, ox, oy, 23, 8, 24, 10, Color8(110, 112, 118))     # ears
		SPR_LION:
			_animal(img, ox, oy, Color8(202, 156, 84))
			_blob(img, ox, oy, 23.0, 12.0, 5.0, 4.5, Color8(150, 96, 50))  # mane
			_blob(img, ox, oy, 24.0, 12.0, 2.6, 2.2, Color8(210, 170, 100))
		SPR_FISH:
			_blob(img, ox, oy, 16.0, 20.0, 10.0, 2.0, Color8(210, 232, 244))  # ripple
			_blob(img, ox, oy, 15.0, 16.0, 5.0, 2.6, Color8(88, 122, 158))    # back
			_rect(img, ox, oy, 13, 12, 15, 15, Color8(88, 122, 158))          # fin
		SPR_WHEAT:
			for s in range(6):
				var sx := 6 + s * 4
				_rect(img, ox, oy, sx, 14, sx, 28, Color8(212, 178, 92))
				_rect(img, ox, oy, sx - 1, 10, sx + 1, 15, Color8(228, 198, 110))
		SPR_BERRY:
			_blob(img, ox, oy, 16.0, 21.0, 9.5, 7.0, Color8(74, 124, 66))
			for b in range(7):
				var bx := 10 + (b * 7) % 12
				var by := 17 + (b * 5) % 7
				_rect(img, ox, oy, bx, by, bx + 1, by + 1, Color8(198, 62, 74))
		SPR_ROCK:
			_blob(img, ox, oy, 16.0, 21.0, 9.0, 7.0, Color8(138, 134, 128))
			_blob(img, ox, oy, 11.0, 24.0, 4.5, 3.2, Color8(120, 116, 112))
		SPR_ORE:
			_blob(img, ox, oy, 16.0, 21.0, 9.0, 7.0, Color8(96, 92, 90))
			for g in range(4):
				var gx := 12 + (g * 4) % 9
				var gy := 18 + (g * 5) % 6
				_rect(img, ox, oy, gx, gy, gx + 1, gy + 1, Color8(120, 214, 224))


static func make_tile_texture(base: Color, accent: Color, shadow: Color) -> Texture2D:
	var image := Image.create(16, 16, false, Image.FORMAT_RGBA8)
	for y in range(16):
		for x in range(16):
			var color := base
			if y < 3:
				color = color.lightened(0.08)
			if x > 11 or y > 11:
				color = color.darkened(0.08)
			if (x + y) % 5 == 0:
				color = color.lerp(accent, 0.34)
			elif (x * 3 + y * 5) % 7 == 0:
				color = color.lerp(shadow, 0.24)
			image.set_pixel(x, y, color)
	return _to_texture(image)


static func make_water_texture(base: Color, foam: Color) -> Texture2D:
	var image := Image.create(16, 16, false, Image.FORMAT_RGBA8)
	for y in range(16):
		for x in range(16):
			var ripple := 0.5 + 0.5 * sin(float(x) * 0.62 + float(y) * 0.31)
			var color := base.lerp(foam, 0.16 * ripple)
			if y < 2:
				color = color.lightened(0.10)
			if (x + y) % 6 == 0:
				color = color.lerp(foam, 0.26)
			image.set_pixel(x, y, color)
	return _to_texture(image)


static func make_canopy_texture(base: Color, accent: Color) -> Texture2D:
	var image := Image.create(18, 18, false, Image.FORMAT_RGBA8)
	image.fill(Color(0, 0, 0, 0))
	for y in range(18):
		for x in range(18):
			var dx := (float(x) - 8.5) / 7.6
			var dy := (float(y) - 8.5) / 7.1
			var dist := dx * dx + dy * dy
			if dist > 1.0:
				continue
			var color := base.lerp(accent, clampf(1.0 - dist, 0.0, 1.0) * 0.35)
			if y < 5:
				color = color.lightened(0.12)
			if (x * 5 + y * 3) % 7 == 0:
				color = color.darkened(0.12)
			image.set_pixel(x, y, color)
	return _to_texture(image)


static func make_avatar_texture(primary: Color, secondary: Color, skin: Color) -> Texture2D:
	var image := Image.create(16, 24, false, Image.FORMAT_RGBA8)
	image.fill(Color(0, 0, 0, 0))
	for y in range(24):
		for x in range(16):
			var color := Color(0, 0, 0, 0)
			if y >= 2 and y <= 7 and x >= 5 and x <= 10:
				color = skin
			elif y >= 8 and y <= 15 and x >= 4 and x <= 11:
				color = primary
			elif y >= 16 and y <= 22 and x >= 5 and x <= 10:
				color = secondary
			elif y >= 9 and y <= 12 and (x == 3 or x == 12):
				color = primary.darkened(0.10)
			if color.a > 0.0 and y < 6:
				color = color.lightened(0.06)
			image.set_pixel(x, y, color)
	return _to_texture(image)


static func make_shadow_texture() -> Texture2D:
	var image := Image.create(32, 16, false, Image.FORMAT_RGBA8)
	image.fill(Color(0, 0, 0, 0))
	for y in range(16):
		for x in range(32):
			var dx := (float(x) - 15.5) / 14.0
			var dy := (float(y) - 7.5) / 5.8
			var alpha := clampf(1.0 - dx * dx - dy * dy, 0.0, 1.0) * 0.42
			image.set_pixel(x, y, Color(0, 0, 0, alpha))
	return _to_texture(image)


static func make_band_texture(width: int, height: int, sky: Color, ridge: Color, ink: Color) -> Texture2D:
	var image := Image.create(width, height, false, Image.FORMAT_RGBA8)
	for y in range(height):
		var t := float(y) / float(max(height - 1, 1))
		var row := sky.lerp(ridge, pow(t, 1.45))
		for x in range(width):
			image.set_pixel(x, y, row)
	var ridge_y := int(height * 0.65)
	for x in range(width):
		var wobble := int(4.0 * sin(float(x) * 0.07) + 3.0 * sin(float(x) * 0.19))
		for y in range(ridge_y + wobble, height):
			var mix := clampf(float(y - ridge_y - wobble) / 18.0, 0.0, 1.0)
			image.set_pixel(x, y, ridge.lerp(ink, mix * 0.55))
	return _to_texture(image)


static func make_cloud_texture(width: int, height: int, light: Color, shadow: Color) -> Texture2D:
	var image := Image.create(width, height, false, Image.FORMAT_RGBA8)
	image.fill(Color(0, 0, 0, 0))
	for y in range(height):
		for x in range(width):
			var wave := 0.55 + 0.45 * sin(float(x) * 0.09 + float(y) * 0.21)
			var puff := 0.55 + 0.45 * sin(float(x) * 0.03)
			var alpha := clampf((wave * puff) - 0.42, 0.0, 1.0) * 0.82
			if alpha <= 0.0:
				continue
			var color := shadow.lerp(light, wave)
			image.set_pixel(x, y, Color(color.r, color.g, color.b, alpha))
	return _to_texture(image)
