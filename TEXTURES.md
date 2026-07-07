# Tag-Driven Textures — design & strategy

**The goal:** every sprite in the game is *generated* (diffusion model), keyed by
the exact state of the tile it skins — layer, biome, season, time of day,
temperature, growth stage, ecosystem condition, and zoom distance — and served
fast enough that panning, zooming, or fast-forwarding a season never waits on
a GPU.

**The constraint:** generation is slow (seconds) and costs money per image.
The world's state space is continuous, therefore infinite. So the system's one
job is to make the infinite finite, generate each finite appearance exactly
once, and serve everything else from disk.

Implementation: `texgen.py` (the whole pipeline) + `/tiles`, `/texture/…`
endpoints in `godot_bridge.py`. Tests: `test_texgen.py`. Demo:
`python3 texgen.py` → contact sheet in `exports/`.

---

## 1. The three collapses

```
continuous state ──quantize──▶ tags ──canonicalize──▶ key ──dedup──▶ few keys/frame
   (infinite)                (finite)              (one per look)   (dozens, not 10⁴ tiles)
```

1. **Quantize.** Each continuous field becomes a small discrete axis:
   temperature 0.371 → `mild`, season phase 0.61 → `autumn`, vegetation
   0.72 → `mature`. Buckets are the *artistically distinguishable* steps —
   a texture for temp 0.37 and one for 0.39 would look identical, so they
   must share a key.

2. **Canonicalize.** `subject + its relevant tags` → one deterministic key
   string, e.g.

   ```
   tree.oak|lod=group3|season=winter|tod=dusk|temp=cold|growth=mature|cond=pristine|density=dense
   ```

   Axes irrelevant to a subject are **dropped from its key** (the ocean has no
   growth stage; a rock has no ecosystem condition). This relevance mask is the
   main combinatorial control — see §6.

3. **Dedup.** For a chunk on screen, every tile's tags bit-pack into one
   integer; `np.unique` collapses ~16 000 tiles to **typically 10–90 distinct
   keys per frame** (measured on real chunks). Keys — never tiles — are what
   gets cached, generated, counted, and evicted.

## 2. The tag vocabulary (`texgen.AXES`)

| axis | values | notes |
|---|---|---|
| `lod` | obj8x8 … single … group81 | signed zoom ladder, §3 |
| `season` | spring summer autumn winter | per-tile (hemispheres are opposed) |
| `tod` | night dawn day dusk | global per frame (`sun_x`) |
| `temp` | freezing cold mild warm hot | from the live temperature field |
| `wet` | arid dry damp wet | from moisture |
| `growth` | bare sprout young mature lush | from living vegetation |
| `cond` | pristine stressed withered scorched | from `EcoSim` health + scars |
| `density` | sparse patchy dense | group lods only |

Subjects (`texgen.SUBJECTS`): every biome (and sub-biome) as a `ground.*`
tile — generated 1:1 from `BIOME_COLORS`, so new sub-biomes get textures for
free — plus `ground.river`, and props: six tree species mapped from biomes,
`tree.dead`, `shrub`, `cactus`, `rock`, `house`, `road`, and `player`.
Adding an axis = one row in `AXES` + one quantizer line in `derive()`.
Adding a subject = one row. Keys stay canonical because the schema lives in
one place.

## 3. The LOD ladder — every thing at every distance

Signed, anchored at **lod 0 = one object per screen tile**:

```
lod +4  group81   one sprite = a whole forest / town district
lod +3  group27   dozens of trees, a hamlet
lod +2  group9    ~9 trees, a cluster of houses
lod +1  group3    the "3 trees" sprite
lod  0  single    one tree / one house / the player, one tile each
lod -1  obj2x2    the same tree now spans 2×2 tiles
lod -2  obj4x4    4×4 tiles
lod -3  obj8x8    8×8 tiles, full-detail art
```

Zooming **out** aggregates ×3 per step: a sprite *represents more objects*
(the group's `density` tag carries how full it looks). Zooming **in** past
lod 0 magnifies ×2 per step: a single object *covers more tiles* — buildings,
trees, rivers, the player all keep a representation at every zoom. Ground
tiles stay one-per-tile at every lod (close-up grounds are just another key);
props switch from "one instance per tile" to an **instance list**
(`anchor tile + footprint`) pinned to the fixed lod-0 object lattice, so the
same oak stays the same oak while it grows from 1 tile to 64. Close-up
sprites render at `tile_px × footprint` resolution — detail is *added* with
zoom, like the terrain noise octaves.

## 4. Identity is deterministic all the way down

The world is a pure function; its skin must be too. Same tile state → same
key → same prompt → same generation seeds (derived from the key's hash).
Which of a key's N **variations** a tile shows is a hash of its world-lattice
cell (`variation_grid`) — revisit the same beach and you see the exact same
palm, with *nothing stored per tile*. Regeneration after eviction reproduces
the same images (same seeds), so the whole asset store is a disposable cache.

## 5. Lifecycle, fallbacks, and why nothing ever blocks

```
resolve(key):  ready? ──▶ serve exact art
                 │
                 ▼ (miss: queue generation, priority = tiles on screen)
               nearest READY neighbor, same subject, weighted tag distance
                 │            (winter oak missing? serve autumn oak dimmed-by-engine)
                 ▼ (nothing close enough)
               deterministic procedural placeholder (instant, cached, tinted
               by the real tags — season/night/scorch visibly work today)
```

Asset states: `pending → generating → ready` (or `failed`, kept with the
error, retryable; or `evicted`, regenerates on demand). The renderer just
re-requests `/tiles` next frame; art **upgrades in place** as the queue
drains. The tag distance is weighted per axis (wrong `cond` is worse than
wrong `tod`; ordinal axes count steps) with a cutoff, so substitutes are
plausible, never absurd.

**Pre-warming** makes "any time, any season, instantly" true: the clock and
camera are predictable, so for every appearance on screen the service queues
its next-season, next-tod, and lod±1 twins at low priority
(`prewarm_neighbors`). By the time dusk falls, the dusk art is already on
disk. Visible work always outranks speculation in the queue.

## 6. Why this doesn't explode

Naive product: ~45 subjects × 8 lods × 4 seasons × 4 tods × 5 temps × 4 wets
× 5 growths × 4 conds × 3 densities ≈ **11 million** combos. Three cuts:

1. **Relevance masks** — each subject keys only on its own axes (water: 3
   axes ≈ 160 combos; a tree: 7). The *reachable* space is ~10⁵.
2. **Correlation** — tags come from one coherent world: `freezing` co-occurs
   with `winter`/`snow` biomes, `scorched` only where fires happened. Most of
   the 10⁵ is physically unreachable.
3. **Laziness** — only combos the camera actually *visits* are generated.
   A long play session touches thousands of keys, not millions. At ~$0.01 an
   image and 3 variations each, a thousand keys ≈ $30 — and it's a one-time,
   shared-by-all-players cost that accretes into a permanent library.

The measured numbers from the demo: a whole-continent view = 85 keys; a
deep-zoom view = 3 keys; the winter-night revisit of the same spot = +2 keys.

## 7. Storage — the manifest is the system of record

```
texture_store/
├── store.db                  SQLite manifest
├── assets/<subject>/<key_hash>/v0.png v1.png v2.png     real art
└── placeholders/<key_hash>/v0.png …                     instant stand-ins
```

`assets` table: key, hash, subject, lod, tags (json), prompt, negative,
status, variations, px, backend, error, created/generated/last-used, use
count. `variations` table: per-file path, bytes, sha1, provisional flag.

- **Usage stats** (`touch` on every resolve, weighted by tiles served) drive
  both eviction and pre-warming priority.
- **Eviction** (`evict_lru(budget_bytes)`) deletes LRU art but keeps rows —
  prompts and history survive, files regenerate identically on demand. The
  store can therefore run under any disk budget.
- Everything a human curator needs is queryable: "show me all failed keys",
  "the 50 most-served sprites", "every scorched-condition asset".

## 8. Serving fast

Today: placeholders are instant (procedural, cached), fallbacks are disk
reads, `/tiles` answers with whatever exists *now* and never waits. The
legend is tiny (dozens of entries); the codes grid is ints; PNGs are served
once and cached by the client.

Next steps when volume demands (in order):
1. **Atlas packing** — bake the N most-used keys per (subject, lod) into
   sprite sheets at store level; Godot loads one texture per subject instead
   of hundreds of small PNGs.
2. **HTTP caching** — `ETag: sha1` on `/texture/…` (hashes are already in the
   manifest) so clients revalidate for free.
3. **Shared store** — the store dir is already portable; put it behind a CDN
   and every player contributes to and draws from one library.

## 9. The far side of the API

`texgen.Backend` protocol: `generate(GenJob) → [png bytes]`, where `GenJob`
carries prompt, negative, deterministic seed, pixel size, and variation
count. Three implementations ship:

- **`PlaceholderBackend`** — the procedural painter promoted to a backend;
  the whole pipeline runs end-to-end with no API attached (default).
- **`RestBackend(url)`** — the "any image gen api" adapter. Contract:

  ```json
  POST  {"prompt", "negative_prompt", "seed", "width", "height",
         "num_images", "key"}
  200   {"images": ["<base64 png>", "..."]}
  ```
- **`RunPodComfyUIBackend`** — direct RunPod Serverless SDXL integration.
  By default it submits asynchronously and polls (survives multi-minute
  cold starts while the pod spins up):

  `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run` then
  `GET  https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}`

  (`RUNPOD_MODE=runsync` switches to the single blocking call — only safe
  when a worker is already warm.) With bearer auth and payload shape:

  ```json
  {
   "input": {
     "workflow": {
       "...": "ComfyUI workflow JSON"
     }
   }
  }
  ```

  For RunPod's quick-deploy SDXL worker (no ComfyUI), set
  `RUNPOD_INPUT_FORMAT=prompt` to send instead:

  ```json
  {
   "input": {
     "prompt": "...", "negative_prompt": "...",
     "width": 1024, "height": 1024, "seed": 42, "num_images": 1
   }
  }
  ```

  Run `python runpod_smoke.py` once after configuring credentials — it
  health-checks the endpoint, tries both input formats, and saves a test
  image so you know which format the endpoint speaks.

  The generated workflow uses: `CheckpointLoaderSimple`, `CLIPTextEncode`,
  `LoraLoader`, `KSampler`, `VAEDecode`, `SaveImage` (plus `EmptyLatentImage`
  to provide sampler input). Default prompt prefix:

  `"isometric stylized setting, tiny fantasy village on a cliff, tile-game environment, soft sunlight, clean shapes, SDXL, high detail"`

  Required env vars:

  - `RUNPOD_ENDPOINT_ID`
  - `RUNPOD_API_KEY`

  Useful optional env vars:

  - `RUNPOD_DRY_RUN=1` (returns deterministic placeholder images; no credentials)
  - `RUNPOD_MODE` (`run` = async submit + poll, default; `runsync` = blocking)
  - `RUNPOD_INPUT_FORMAT` (`comfyui` default; `prompt` for quick-deploy SDXL workers)
  - `RUNPOD_POLL_TIMEOUT_SEC` (default 600 — total wait incl. cold start),
    `RUNPOD_POLL_INTERVAL_SEC` (default 3)
  - `RUNPOD_TIMEOUT_SEC`, `RUNPOD_RETRIES`, `RUNPOD_RETRY_BACKOFF_SEC`
  - `RUNPOD_SDXL_CHECKPOINT` (default `sd_xl_base_1.0.safetensors`)
  - `RUNPOD_LORA_NAME` (default `stylized-setting-isometric-sdxl-and-sd15.safetensors`)
  - `RUNPOD_LORA_PATH` (default `/workspace/ComfyUI/models/loras/`)
  - `RUNPOD_PROMPT_PREFIX`

  To fetch the Civitai LoRA into the worker path:

  ```python
  import texgen
  texgen.download_civitai_lora(
     source_url="https://civitai.com/models/118775/stylized-setting-isometric-sdxl-and-sd15",
     dest_dir="/workspace/ComfyUI/models/loras/",
     filename="stylized-setting-isometric-sdxl-and-sd15.safetensors",
  )
  ```

  Ensure the SDXL base checkpoint configured via `RUNPOD_SDXL_CHECKPOINT`
  exists on the worker (typically in `ComfyUI/models/checkpoints/`).

Prompts come from `build_prompt()` — one table maps tags to phrases
(`season=winter` → "winter, snow-dusted"; `lod=group3` → "a small cluster of
three …"), with a global style prefix (`isometric pixel art, 16-bit …`).
Changing art direction = editing tables, keys untouched.

### Which API for now (recommendation)

- **Now (cheapest path to real art, zero ops): fal.ai** — SDXL endpoints with
  LoRA support (load an isometric pixel-art LoRA by URL), seed control,
  ~$0.01–0.03/image, plain REST. A ~30-line shim maps its JSON to the
  `RestBackend` contract. **Replicate** is the equivalent alternative
  (per-second billing, huge model/LoRA catalog) if you prefer its ecosystem.
- **At volume (your original instinct, and correct): RunPod serverless +
  ComfyUI + SDXL + isometric 16-bit LoRA.** Cheapest per image once you're
  generating thousands; total control of the workflow graph. The shim is a
  tiny FastAPI wrapper that feeds the prompt/seed into a fixed ComfyUI
  workflow JSON and returns base64 PNGs. Nothing in this repo changes —
  same `RestBackend`, different URL.
- Avoid the big-lab image APIs (gpt-image, Gemini) for this: 10–40× the
  price, no LoRA (style lock is the hard problem), no seed determinism.

Two things the shim must own regardless of vendor: **transparent
backgrounds** for props (SDXL has no alpha — run `rembg`/background removal
server-side before returning), and **style lock** (one LoRA + fixed style
prefix; later, IPAdapter reference images for cross-key consistency).

## 10. Bridge API (what Godot consumes)

- `GET /tiles?cx&cy&zoom&view_tiles&…` (same params as `/frame`, plus
  `prewarm=0/1`) →

  ```json
  {"lod": -1, "lod_name": "obj2x2",
   "ground_codes": [...],           // N² ints, row-major
   "ground_variations": [...],      // N² ints, which v#.png per tile
   "props": [[i, j, code, variation, footprint], ...],
   "legend": {"<code>": {"key", "key_hash", "status", "served",
                          "tags", "footprint", "urls": ["/texture/<hash>/0.png"]}},
   "queue": 3}
  ```

- `GET /texture/<key_hash>/<idx>.png` — always the best art that exists for
  that key right now (exact → fallback → placeholder).
- `GET /texture_stats` — store + queue counts.
- CLI:
  - `--texture-backend rest --texture-url http://…` for generic REST
  - `--texture-backend runpod-comfyui` for direct RunPod worker-comfyui
  - plus `--texture-store`, `--texture-variations`, `--tile-px`.

## 11. Roadmap

1. ✅ Schema, quantizers, canonical keys, chunk dedup (`derive`)
2. ✅ Store + manifest + lifecycle + LRU eviction
3. ✅ Fallback chain + placeholder painter + variation determinism
4. ✅ Priority queue + worker + pre-warming
5. ✅ Bridge endpoints (`/tiles`, `/texture/…`, `/texture_stats`)
6. ✅ RunPod worker-comfyui SDXL backend + workflow payload path
7. ☐ Godot: render chunks from `/tiles` (swap placeholder → real art live)
8. ☐ Atlas packing + ETags
9. ☐ Curation pass: review grid (the contact-sheet, but per subject),
     regenerate-with-edited-prompt, pin/ban variations
