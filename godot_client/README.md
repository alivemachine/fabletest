# Godot Client Prototype

This client is a Godot 4 prototype for walking the evolving world in third person.
It keeps the world simulation in Python and streams a player-centered chunk into
Godot over HTTP.

## What is in here

- `project.godot` boots a minimal Godot 4.2+ project.
- `scenes/main.tscn` starts the prototype scene.
- `scripts/main.gd` handles input, bridge polling, camera, HUD, and parallax.
- `scripts/terrain_chunk.gd` turns streamed biome/height/light fields into a
  local chunk of columns, water overlays, vegetation stacks, and small props.
- `scripts/pixel_art.gd` generates placeholder pixel-art textures in code so
  the prototype runs without an external tileset yet.

## Run

From the repository root, start the bridge:

```powershell
python godot_bridge.py --seed 42 --size 192 --civ-count 3
```

Then open `godot_client/` in Godot 4.2 or newer and run the main scene.

To auto-capture 10 Full HD gameplay screenshots (including teleports near
settlements) for the web page gallery, run from the repository root:

```bash
python3 generate_godot_screenshots.py
```

Run that script after any Python or Godot update so the browser gallery stays
current.

## Controls

- `WASD` or arrow keys: move across the streamed world.
- `[` / `]`: zoom the streamed terrain chunk out or in.
- `Space`: pause or resume world evolution.
- `R`: reset the ecosystem to day 0.

## Why this does not use Better Terrain yet

`better-terrain` is a strong fit for a `TileMapLayer`-driven 2D or 2.5D autotile
pipeline. Its API is centered on writing terrain ids into `TileMapLayer` cells
and then calling `BetterTerrain.update_terrain_area(...)` so Godot chooses the
best atlas tile.

This prototype is doing something else on purpose:

- the streamed view is rendered as a 3D chunk with height columns,
- the player is in a third-person camera rig,
- vegetation is shown as sprite-stacked billboard props,
- parallax layers sit behind the 3D terrain.

That means Better Terrain would not replace the current chunk renderer. It would
only help after a deliberate pivot to a `TileMapLayer` terrain surface.

## If you want to switch to Better Terrain next

Keep the bridge exactly as-is and replace only the Godot terrain surface:

1. Copy `addons/better-terrain` into `godot_client/addons/` and enable the plugin.
2. Replace `TerrainChunk.gd` with one or more `TileMapLayer` nodes using an
   authored tileset.
3. Map streamed `biome_id` values from `godot_bridge.py` to Better Terrain
   terrain ids.
4. For each new chunk, write the terrain ids into the visible `TileMapLayer`
   cells and call `BetterTerrain.update_terrain_area(tilemap_layer, rect)`.
5. Keep the current bridge polling, player rig, camera, and parallax layers.

That route will save code once you have a real 16-bit tileset and want editor-
driven autotiling. The current prototype is the faster path to validate the
streaming seam, third-person navigation, and evolving-world loop first.