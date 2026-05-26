# Rubber-Sheet Oblique-3D Field Viz — Architect Blueprint

Generated 2026-05-26 by code-architect agent. Source of truth for the
implementation that lands across multiple commits.

## What we're building

Oblique 45° "rubber sheet" view: when `state.fieldStyle === 'rubber-sheet'`,
the ENTIRE rendered scene (mesh + bodies + trails + drag preview +
prediction line + particles) gets oblique-projected so bodies visually
sit at the bottom of their gravity wells. Physics still operates on
world (x, y) — only the renderer transforms positions.

## Math

- `screen_x = world_x`
- `screen_y = world_y * cos(45°) + sag(world_x, world_y) * sin(45°)`
- `sag(x, y) = SAG_MAX * clamp(-φ(x,y) * dispScale / refDepth, 0, 1)`
  (matches the 3D branch of GRID_WARP.VS's formula incl. anti-fold cap)
- `SAG_MAX = 140 px` (chosen by user from tilt=45° dense mockup)

## Design decisions (architectural)

1. **No new module.** Projection logic lives in `renderer-webgl.js`.
2. **Per-shader uniforms, no shared GLSL include.** Use JS template
   literal interpolation to share a `SAG_VS_HELPER` string across 6
   modified shaders.
3. **CPU computes sag texture (128×128 R32F), GPU samples.**
4. **Trail FBO: project per-dot at write time.** On toggle to/from
   'rubber-sheet', call `resetTrailCanvas()`.
5. **Painter's algorithm**, not depth buffer (GL ctx is `depth: false`).
6. **Toggle: snap, no fade.**

## Files to modify

- `src/state.js` — add `'rubber-sheet'` to `fieldStyle` enum
- `src/shaders.js` — modify 6 VS (TRAIL_DOT, ENTITY, CIRCLE_FILL,
  CIRCLE_RING, LINE_SEG, PARTICLE_FLOW) to sample sag texture
- `src/renderer-webgl.js` — sag texture infrastructure, painter sort,
  trail projection, sag upload in each draw path
- `src/input.js` — `_worldFromScreen(sx, sy)` with 2-step Newton
- `src/ui.js` — toggle wiring + new field-style option

## Build sprints (8 steps)

1. **`state.js`**: add `'rubber-sheet'` to `fieldStyle` enum
2. **`renderer-webgl.js` sag-texture infrastructure**:
   - `_sagTexture` GL handle, `_sagPixels` Float32Array scratch (128×128)
   - `_updateSagTexture(mode)` — per-frame CPU compute + GPU upload
   - Hook into `_initPrograms` (create), `resizeRenderer` (reallocate),
     context-loss restore
3. **`shaders.js` — sag sampling in 6 VS**:
   - Define `SAG_VS_HELPER` string with sample+project function
   - Interpolate into TRAIL_DOT, ENTITY, CIRCLE_FILL, CIRCLE_RING,
     LINE_SEG, PARTICLE_FLOW VS via template literal
   - Each VS adds `uniform sampler2D uSagTex`, `uniform float uSagMode`,
     `uniform vec2 uSagViewport`
4. **`renderer-webgl.js` — wire sag uniforms into draw paths**
5. **`renderer-webgl.js` — painter sort + trail projection**:
   - Y-sort `state.entities` slice before bucket loop in `_drawEntities`
   - Modify trail dot push to apply forward projection at write time
6. **`input.js` — inverse projection**:
   - `_worldFromScreen(sx, sy)` two-iteration Newton:
     - `y_0 = sy / cos45`
     - `y_k+1 = (sy - sag(sx, y_k) * sin45) / cos45`
   - Identity when `fieldStyle !== 'rubber-sheet'`
7. **`ui.js` — toggle wiring**:
   - Add `'rubber-sheet'` to fieldStyle button group
   - On switch to/from 'rubber-sheet': call `resetTrailCanvas()` and
     `_particlesSeeded = false`
8. **Tests** (`tests/potential.test.html`):
   - Fixture A: `_updateSagTexture` single-body sag at body center vs
     2ε distance
   - Fixture B: `_worldFromScreen` round-trips with <0.5 px error

## Performance budget

- 128×128 sag eval × N entities: ~2 M ops at N=128, ~500 K at N=30.
  Matches `_computeFieldIntensityRange`'s current 192×N cost (already
  benchmarked acceptable).
- `_sagPixels` pre-allocated once at `_initFbos` — no per-frame GC.
- `texSubImage2D` upload of 64 KB once per frame — trivial.

## Wrap mode

`_fieldEntityData` already populated with 9-ghost copies by `drawField()`
before `_updateSagTexture` runs. No additional min-image logic needed.

## Open implementation questions (resolve as you go)

- Exact `SAG_MAX` constant (start at 140 px, tune)
- `_sagTexW × _sagTexH` (start 128×128)
- Newton iteration count (start at 2)
- Texture unit slot allocation (dynamic per-draw bind matches existing
  sprite pattern)
