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
- **Pan & zoom the big map** — scroll / pinch to zoom, drag to pan,
  double-click to zoom in or reset. This is not an image zoom: the world is a
  pure function of *continuous* `(x, y)`, so diving in re-samples the same field
  at a finer step and **adds** higher-frequency detail, coherent with the planet
  above it — the coastline you saw from orbit is exactly where you left it, you
  just see the individual bays now. Thumbnails stay whole-planet so you keep
  your bearings, and **export captures the current window**, so a zoomed patch
  exports as easily as the whole globe.
- **World**: seed textbox + `rnd` button, sea level, river threshold. The
  **tide** is a waterline that *sweeps a fixed beach* (the sand is as wide as
  the tidal range and stays put; only the water covering it moves); the
  **sea-level slider** still moves the whole coast — and, on the Vitality layer,
  floods or exposes land with lasting ecological consequences.
- **Time & climate**: **logarithmic** sim speed (crawl to thousands of days/sec
  — fast-forward centuries), day/night depth, season amplitude (opposite per
  hemisphere; crank it for hot summers that spark fires/drought), tide amplitude.
- **Transport**: pause / run, **reset to day 0** (also rewinds the living
  ecosystem to pristine).
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
| Vitality | the **living, stateful** world: soil fertility, vegetation, fauna, civilisation and burn/salt scars that integrate forward and *remember* — floods drown, droughts burn, and recovery is slow | ecosystem sim (M4, stateful) |
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

### Vitality — the living, stateful world (first M4 layer)

Every layer above is a pure function of `t`; **Vitality is not.** It is a coarse
ecosystem that *integrates forward as the clock runs and has memory*, so events
leave lasting consequences instead of looping. Per cell it tracks **soil
fertility, vegetation, fauna, civilisation, and burn/salt scars**:

- **Your sliders are the triggers.** Shoving the **sea-level** slider up floods
  land relative to the level the ecosystem is *adapted* to (its `sea_ref`,
  which drifts only slowly) → mass die-off and **salted soil**; pull it back and
  the exposed seabed is barren, greening only from the edges. High **season
  amplitude** drives hot summers that **ignite fires** in grass/savanna and
  **desertify** bare hot ground.
- **Slow variables give irreversibility.** Fertility and scars heal on a
  timescale of *years*, and vegetation/fauna/civilisation only recover by
  **colonisation from surviving neighbours** — so a wiped, isolated region stays
  dead until life spreads back in. Some worlds never come back; that is the
  point — you can watch which worlds last.
- **Consequences ⇒ not scrubbable backward.** The state at day 900 depends on
  the whole path of what you did, not on a formula of `t`, and the dynamics are
  dissipative (a burned and a drowned forest both end at `veg=0` — the present
  can't tell you which). So Vitality only runs **forward or resets**; the *other*
  layers stay fully seekable. The stateful mess is quarantined to this one layer.
- **Fast-forward centuries.** The speed control is logarithmic (up to thousands
  of days/sec, with internal sub-stepping) so you can wreck a world and watch,
  over simulated centuries, whether it ever greens again.

This is the far→near step of the design made real: the pure layers are the
seekable substrate; Vitality is the stateful stock field every future M4 agent
(herds as boids, settlements, NPCs) will expand out of and collapse back into.

### Continuous zoom — the pure function at any scale (the step before M4)

The 512² grid was never "the world" — it is one *sampling* of `world(seed, x,
y, t)`, which is defined for continuous `x, y`. So a "world map" and a "tile-
accurate patch of one beach" are the same function sampled over different
windows, not two different data structures:

- **Windowed noise** (`worldgen.noise_window`) samples any window
  `[cx±span/2, cy±span/2]` of the unit torus at any zoom. Its value at each
  integer lattice corner is a *hash* of `(seed, octave, i, j)`, so we only ever
  evaluate the corners a window touches — O(window pixels), independent of zoom
  depth. A million-cell-wide high-frequency octave is never allocated; it is
  sampled four corners at a time.
- **Detail is added, not revealed.** Each octave is divided by a *fixed* total
  (Σ½ⁿ = 2), never by "how many octaves we summed," so the low octaves
  contribute identically at every zoom. The planet's coastline stays put; diving
  in only *adds* the finer octaves that a wider window couldn't resolve. This is
  why zoom is coherent rather than a blur.
- **`WorldSlice.view(cx, cy, zoom)`** re-samples the fields for a window and
  reuses this planet's history timeline, faction cores and normalization by
  reference — so panning and zooming never re-runs the M3 CA. Latitude,
  longitude and hillshade are recomputed from the window's *world* coordinates,
  and the coarse history grid is sub-sampled to the window, so temperature,
  seasons, day/night and territory all stay correct as you dive in.

A planet is *supposed* to be finite (it wraps — it is a globe); what you
actually wanted was infinite **detail**, not infinite **extent**, and that is
what zoom gives: finite world, bottomless zoom, computed only for the window on
screen.

**Current limits (the next tasks, on the road to M4):** rivers are global D8 and
can't yet be windowed without upstream boundary conditions, so a zoomed view
draws no rivers; civilization is still the coarse HIST grid upsampled, so zoom
does not yet *expand* a cell's population into individual settlements and
buildings. Both are `expand()` work — turning a coarse cell's summary (faction,
population, stress) into a deterministic settlement grammar keyed by the cell's
seed — which is exactly the M4 Resolver this zoom pipeline is the skeleton for.

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
