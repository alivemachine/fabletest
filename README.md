# World Engine — prototype (M0–M2)

The world is a pure function: `world(seed, x, y, t) → layers`. See `DESIGN.md`
for the full design. This repo is the Python prototype stage of the roadmap:

- `worldgen.py` — the layer stack itself (M0 fields → biomes, M1 D8 water).
  Runs standalone: `python3 worldgen.py` writes `world_<seed>.png`.
- `world_core.py` — the shared render core: vectorized biomes, the M2 time
  layer (day/night, seasons, tides), per-layer RGB renderers. numpy in,
  pixels out; no UI, no I/O.
- `world_viewer.py` — **the desktop layer console** (matplotlib) around the
  core: every layer animating, sliders, PNG-sequence exporter.
- `web/` — **the same console in the browser** (works on a phone):
  `worldgen.py` + `world_core.py` run unmodified in the browser through
  Pyodide (Python on WebAssembly); the page is only canvas + sliders + a
  zip download. Deployed to GitHub Pages by `.github/workflows/pages.yml`,
  which also vendors the Pyodide runtime so there is no CDN dependency.

**Live console:** https://alivemachine.github.io/fabletest/
(first visit downloads the Python runtime, ~12 MB, cached afterwards)

![console](docs/console.png)

## Run

```bash
pip install -r requirements.txt
python3 world_viewer.py                 # default: seed 42, 256² (fluid)
python3 world_viewer.py --seed 7 --size 512   # prettier, heavier rebuilds
```

## The interface

- **Big map + 7 thumbnails** (world composite, elevation, temperature,
  moisture, flow, biomes, daylight) — all animating. **Click a thumbnail**
  to promote that layer to the main view.
- **World**: seed textbox + `rnd` button, sea level, river threshold.
- **Time & climate**: sim speed (days/sec), day/night depth, season
  amplitude (opposite per hemisphere), tide amplitude (sea level breathes).
- **Transport**: pause / run, reset to day 0.
- **Export**: writes `exports/world_s<seed>_<layer>_<N>f/frame_0000.png …`
  of the *current view layer* at the current seed and settings, starting at
  the current sim time, plus a `meta.json` recording every parameter — so any
  sequence is exactly reproducible.

Time model: 1 sim day = one day/night cycle, year = 96 days, tide ≈ 12.5 h.

## Turning a sequence into a video

```bash
ffmpeg -framerate 24 -i exports/world_s42_composite_96f/frame_%04d.png -pix_fmt yuv420p world.mp4
```

## Notes

- Everything time-dependent is recomputed per frame, vectorized in numpy;
  only elevation/moisture/flow are rebuilt (once) when seed or size changes.
- `compute_rivers` in the original M0 file had a D8 sign bug (`np.roll`
  sampled the neighbor at `−offset` but recorded the index at `+offset`),
  which silently broke flow accumulation — max accumulation was ~14, so no
  river ever crossed the threshold. Fixed here; rivers now accumulate
  properly and the flow layer shows the drainage network.
- The default river threshold scales with resolution (`350` is tuned for
  512²).
