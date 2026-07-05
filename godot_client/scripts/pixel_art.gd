extends RefCounted


static func _to_texture(image: Image) -> Texture2D:
    return ImageTexture.create_from_image(image)


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