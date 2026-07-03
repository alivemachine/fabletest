# World Engine — One-Page Design

## The one principle
**The world is a pure function.** `world(seed, x, y, t) → layers`.
Nothing is stored except player-caused changes (**deltas**). Everything else is
recomputed from the seed on demand, cached while near, discarded when far.
That is why the download is tiny and the world is infinite.

## The layer stack (each layer reads only the layers *above* it)
1. **Fields** — elevation, temperature, moisture, wind. Tileable noise → wraps forever.
2. **Water** — D8 flow on the height field → rivers, lakes, drainage. Rain from wind + terrain.
3. **Climate** — a cellular automaton on a coarse grid: droughts, floods, fire, sea ice, storms.
4. **Ecology** — vegetation field + animal *populations* (Lotka–Volterra CA per cell).
5. **Society** — settlements, factions, economy, culture (society CA + settlement grammar).
6. **Individuals** — one person/animal: schedule, storylets, relationships, birth, death.

Downward = constraint. Upward = summation only. This never changes; depth just
re-runs the same stack at a finer scale.

## Tool-selection rule (ask two questions of anything)
**A — what kind of process is it?**

| Phenomenon | Tool |
|---|---|
| Static, continuous | **Noise** (terrain, plant potential, minerals) |
| Downhill flow from a height field | **Flow algorithm** (rivers, lava, migration) |
| Spreads / diffuses / phase-changes over time on a grid | **Cellular automaton** (fire, weather, populations, pheromones) |
| Discrete, mobile, individual state | **Agent** (person, animal, herd, storm, army, meteor) |
| Bounded space filled by local fit-rules | **WFC / grammar** (interiors, nest tunnels) |
| "Who connects / eats / allies / routes to whom" | **Graph** — *derived & consulted, never a generator* |
| Many trivial visual parts, near only | **Particles** (cosmetic rain, sparks) |

**B — how far is the player?** Far → collapse to a statistic (a stock).
Near → instantiate the agent. Orthogonal to A; always applies.

Keep this primitive set **tiny** (7–8 tools). Express every new thing as a
*combination* of them, never a new primitive.

## One agent, five scales
An agent is always the same abstraction: *consume a stock → transform → produce a
stock, on some timescale.* A wolf, a storm, a city, a meteor are the same model at
different sizes. Bigger/slower agents set boundary conditions for smaller/faster ones.

| Tier | Simulated live? |
|---|---|
| Cosmic (star, moon, tides, impactor) | Never — pure function of `t` |
| Climate (winds, storms, climate CA) | Only near player, else keyframed |
| Ecology (vegetation, populations, herds) | Populations always (cheap); herds near |
| Society (settlements, factions, economy) | Aggregate always; detailed near |
| Individual (one person/animal) | Only on screen; else a statistic |

Human/animal verbs (reproduction, death, job, day-night, relationships, culture,
contentment) all live at the Individual tier, each with a **far form** (a rate/number
in a stock) and a **near form** (an agent event/storylet). The **relationship graph**
and the **food web** are your only two graphs — both derived data the sim consults.

## Infinite depth = one Resolver
The engine never "renders a new system." Every entity has a `type` tag and a
deterministic `seed` (= parent seed + coords) and two methods:
- **expand()** — cross the detail threshold → instantiate the finer stack beneath it.
- **collapse()** — leave → summarize back to a statistic, keep only deltas.

Plus a **budget governor** capping how many things are expanded at once (compute
homeostasis). Meeting ants = the ground becomes terrain at a finer noise frequency,
the colony expands into agents (ants) + a CA field (pheromone trail) + WFC (nest) +
stock-flow (foraging). You didn't write an ant system — you ran the stack one scale down.
**Honest caveat:** the engine recombines a *finite* primitive set infinitely; a truly
novel mechanic needs *you* to author a new resolver.

## Build roadmap
- **M0 ✅ Fields → biomes, rendered as a map.** *(worldgen.py — done)*
- **M1  Water.** D8 rivers ✅ + rainfall advection (wind carries moisture, rain shadows), lakes via priority-flood.
- **M2  Time.** Animate it: day/night, seasons (temp offset), tides (`sin(t)`). Export frames → **watch worlds evolve; reject collapsing ones.**
- **M3  History CA.** Coarse dynamic grid: civilizations, populations, resources, events. Keyframe for time-travel.
- **M4  LOD + Resolver.** expand/collapse, settlements, NPCs near/far, budget cap. Agents = people + animals + storms, one code path.
- **M5  Ship in Godot.** Port the deterministic core, add the player, interiors (WFC), rendering. Exports to phone / PC / console.

**Stack:** prototype M0–M3 in Python (numpy + PIL) so you can *see* every layer.
Move to **Godot** at M5. Not a function graph — a fixed stack with the "read upward
only" rule stays understandable.
