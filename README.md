# World Engine — prototype (M0–M2)

The world is a pure function: `world(seed, x, y, t) → layers`. See `DESIGN.md`
for the full design. This repo is the Python prototype stage of the roadmap:

- `worldgen.py` — the layer stack itself (M0 fields → biomes, M1 D8 water).
  Runs standalone: `python3 worldgen.py` writes `world_<seed>.png`.
- `world_core.py` — the shared render core: the full layer stack (fields →
  clouds → water → biomes → flora → fauna → civilization), the M2 time layer
  (day/night, seasons, tides), and per-layer RGB renderers. numpy in, pixels
  out; no UI, no I/O.
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

## The layers

| Layer | What it is | Tool (per DESIGN.md) |
|---|---|---|
| Elevation / Temperature / Moisture | static fields; temp carries seasons | noise + gradient |
| Clouds | two noise sheets advected by wind, gated by moisture, piled on windward slopes | advected noise |
| Flow | D8 flow accumulation → rivers | flow algorithm |
| Biomes | lookup on (elevation, temperature, moisture) | table |
| Flora | vegetation density = warmth × water, pulsing with seasons | field |
| Fauna | herbivore/predator biomass on a Lotka–Volterra limit cycle (a resource/game map) | population CA (far form) |
| Civilization | 1–6 factions: territory + population from the M3 history, tinted by faction; each depletes local game | history CA (M3) |
| History | the chronicle: same territory with war-fronts (red) and famine/pest zones (violet) drawn on | history CA (M3) |
| Daylight | the day/night terminator | sin(t) |

### M3 — the history simulation

Civilization is no longer a closed-form curve; it is a coarse cellular
automaton (`HIST_SIZE`² grid, one step per sim-week) that actually runs:

- **stocks per cell** — population, food capacity, faction influence, unrest.
- **dynamics** — population grows logistically toward food capacity and
  colonises reachable land; faction *influence* diffuses outward (blocked by
  sea and mountains) and whoever has the most influence owns the cell.
- **war is emergent** — where two factions' influence meet and contest a
  border, that contest is a mortality term: population dies along the front
  and ownership flips when one side overtakes. Nobody scripted a war.
- **shocks feed the food chain** — deterministic **pests/blights**,
  **droughts** (arid, long), and **cold spells / ice** (polar, long) each
  depress food capacity in a region for a while. Capacity drops below
  population → famine (the logistic term goes negative) → unrest → war and
  migration. That is the pest → famine → war → migration cascade, run through
  the food web rather than authored.

The CA is **integrated once at build time** into ~110 keyframes over a
24-year horizon, stored compactly (uint8). `render(t)` interpolates the
timeline, so history stays fully **seekable and exportable** — scrub to any
year, export any year — even though the underlying process is stateful. Past
the horizon the final state holds. The **Civilization** layer shows territory
and population; the **History** layer overlays the wars and famines.

### Far form vs near form (why these are seekable)

Every time-dependent layer here is the **far form** — a pure, *seekable*
function of `t` (a stock/statistic). You can jump to day 500 without
simulating days 0–499, which is exactly what the exporter relies on. Fauna
rides the Lotka–Volterra *limit cycle* directly instead of integrating the
ODE; civilization applies logistic growth in closed form. Cross-layer
coupling that can be written as an algebraic function of the current fields
is included (moisture→flora→fauna carrying capacity; flora+water+climate→
civilization habitability; civilization→local game depletion).

The **near form** — live agents that integrate over time, where a pest, a
war, a storm, or an advancing ice front is a *shock injected into a stock and
propagated through the food-web graph* — is the M4 Resolver. It is stateful
(not seekable) by nature, so it is deliberately a separate build, not part of
this pure-function core.

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
