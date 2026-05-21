# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running and testing

**No build step.** ESM modules load directly. `d3-quadtree` is the sole runtime dependency, pulled from `https://esm.sh/d3-quadtree@3` via the import map in [index.html](index.html). Do **not** add a bundler, package.json, or `node_modules` — the GitHub Pages deploy in [.github/workflows/deploy.yml](.github/workflows/deploy.yml) uploads the repo root verbatim.

```bash
# Local preview (any static server — file:// breaks ESM)
python -m http.server 8000
# then open http://localhost:8000
```

**Tests:** open [tests/potential.test.html](tests/potential.test.html) directly in a browser. No test runner — it's a hand-fixture page that prints pass/fail in the DOM. The fixtures cross-check `computePotentialAt` / `computeForceDirAt` in [src/potential.js](src/potential.js) against hand-computed expected values.

## Architecture

### Simulation loop (main.js)

Fixed-dt accumulator. Every wall-clock frame:

1. `prepareFrame(entities)` — builds the per-frame Barnes-Hut quadtree **once** if `N ≥ state.bhThreshold`. Does **not** build the spatial hash; that's per-substep (see broadphase note below).
2. Drain accumulator with up to `MAX_SUBSTEPS=8` calls to `stepPBD` → `updateAbsorptions` → `applyBoundary`.
3. `updateTrailCanvas(simDelta)` — phosphor FBO decay, keyed on simulated (not wall) time so pausing freezes trails.
4. `drawSceneGL()` → `drawField()` → `drawUI()` — single `#stage` WebGL 2 canvas. `drawField` is gated by `state.showField` (zero cost when off).

Effective time ratio is overridden to `EDIT_MODE_TIME_RATIO` when `state.isEditMode` — the slider value is ignored in edit mode by design.

### Physics pipeline (physics.js → stepPBD)

Box2D-style **split-impulse** solver. Per substep:

- **A.** Reset pseudo-velocity accumulators (`_pvx`, `_pvy`).
- **B–C.** `computeAccelerations` → gravity kick (`v += a·dt`).
- **D.** Predict positions (`x += v·dt`).
- **E.** `handleCollisions` — broadphase at *predicted* positions. **Rebuilds the spatial hash per-substep** (NOT per-frame) because the broadphase is a binary classifier where a <1px drift across a cell boundary causes a 100% miss for that pair. Pushes planet-planet contacts onto `_contacts[]`; BH-pair contacts fire `beginAbsorption` immediately.
- **F.** `_solveContactVelocities` — Sequential Impulses, 8 iterations, **warm-started** from `_prevPairImpulses` (Map keyed by symmetric `_pairKey(a,b)`). Accumulated impulse clamped ≥ 0.
- **F'.** `_rebuildPrevPairImpulses` — persist this substep's converged impulses for next substep's warm-start. Stores normal direction alongside magnitude so wrap-boundary teleports can be detected via dot-product sign flip and the stale warm-start discarded.
- **G.** `_solvePositionConstraints` — pseudo-velocity NGS, 3 iterations, Baumgarte slop (`LINEAR_SLOP_FRAC × rSum`) + `MAX_CORRECTION_FRAC` cap. Integrates `_pv*` into `x/y` at the end. **Real velocity is never touched by the position pass** — this is the whole point: it prevents position correction from pumping phantom PE into the gravity field.

Things deliberately **not** in the pipeline (removed in the 2026-05-20 Box2D refactor — don't re-add without re-reading the long block comment above `stepPBD`): centripetal projection, energy refund (`_dxGrav`), Catto relaxation pass, static-contact damping. The split-impulse pattern made all four redundant.

### Charge-asymmetric force model

**Newton's 3rd law is intentionally broken.** Force on A from B exists *only* if `B.charge !== 0`:

```
a_on_A = B.charge · G · m_B / r² · unit(B - A)
```

A `charge=0` body still *receives* force but applies none. This is the spec, not a bug. Implications throughout the code:

- The pairwise loop in `computeAccelerations` checks each direction independently (lines 113 / 119 of [physics.js](src/physics.js)).
- Barnes-Hut node moments use **signed** `mq = Σ(m·q)` — see [physics-barneshut.js](src/physics-barneshut.js).

### Direct O(N²) vs Barnes-Hut switch

`computeAccelerations` dispatches to `computeAccelerationsBH` when `entities.length ≥ state.bhThreshold` (default 256). **The direct path is the correct path.** BH does per-target tree traversal with no pairwise accounting, so for any pair the force A reads off the tree is not the exact equal-and-opposite of what B reads. The residual drifts cluster CoM — empirically ~2.4 px/s on a 143-body hex cluster at G=80, amplified by the 9-ghost PBC sum in wrap mode. The threshold was raised from 64 → 256 (2026-05-20) so typical scenes use the exact path; users who scale up accept the trade-off. **Do not silently lower this default.**

### Wrap mode (`state.boundaryMode === 'wrap'`)

Wrap is treated as toroidal PBC. Every distance calculation uses the **minimum-image convention** (`minImageDelta`), and every long-range sum is over the **9-ghost neighbourhood** (the identity image + 8 mirrors at `±W, ±H, ±W±H`):

- Pair force loop: `dx/dy` clamped to ±half-span (physics.js lines 92–95).
- Barnes-Hut: 9 separate tree descents per target, one per image offset.
- Spatial hash: modular cell indexing; cell size must be `≥ 2 × maxR` to cover the 3×3 neighbourhood. Cells are sized **per-axis** so they tile the viewport exactly — the previous single-`cellSize` impl missed Y-wrap pairs when `viewport.height % cellSize !== 0` (see [physics-spatial-hash.js](src/physics-spatial-hash.js:36–40) for the regression note).
- Field viz (`computePotentialAt` / `computeForceDirAt`): 9-ghost sum, passed `boundary={width,height}` from the renderer.

If you add a new force/field calculation, it MUST handle wrap correctly or the world becomes discontinuous at the edges.

### Plummer softening — the CPU/GPU oracle contract

`φ(x,y) = Σ_i  -G · q_i · m_i / sqrt(r_i² + ε²)` is implemented in **three places** that must stay bit-identical:

1. [src/potential.js](src/potential.js) — JS reference.
2. The GLSL `EQUIPOTENTIAL.FS` fragment shader in [src/shaders.js](src/shaders.js) — production render path.
3. The acceleration loop in [src/physics.js](src/physics.js) (the `r2 = r2Raw + minR*minR` line and the magnitude formula derived from it).

[tests/potential.test.html](tests/potential.test.html) is the cross-check oracle. The earlier `max(r, ε)` hard-floor softening was abandoned because it produced a force/potential kink (force constant, ∇φ=0 inside the floor) → energy non-conservation in close approaches + visible contour wobble. If you change one site, change all three and re-run the fixture page.

### Renderer (renderer-webgl.js, shaders.js)

Single WebGL 2 canvas. All UI overlays (hover ghost, drag rubber band, prediction line, selection ring, absorbing fallback) are GL too — there is no Canvas2D overlay layer anymore. Coordinates are CSS-pixel logical units; the vertex shader maps `[0..W] × [0..H] → NDC` via a uniform ortho matrix; backing store is DPR-scaled.

Sprite path: [sprite-cache.js](src/sprite-cache.js) bakes each unique `(type, color, radius, charge, pinned)` combo into an off-screen Canvas2D once (radius is quantized to 0.5 px to coalesce slider drag variants). Renderer lazily uploads each canvas to a GL texture on first use, then issues one instanced draw per sprite bucket. Absorbing entities bypass the cache (alpha changes every frame) and use a fallback arc path.

Trail FBO: two RGBA8 textures ping-ponged each frame — decay pass (`a -= uDec`) then instanced dot plot with `gl.MAX` blend (emulates "newer wins"). Composited at frame end.

### State (state.js)

Single mutable `state` object — every other module imports and mutates through small helpers. `DEFAULTS_TUNING` is the frozen "factory reset" source for the 高级调参 panel; `state.G / epsilon / launchSpeedK / ...` are runtime-mutable and read at call time (not at module load) so slider changes take effect on the next frame. `state.radiusBase` is recomputed on every `setupCanvas` so the default body radius scales with viewport; existing entities keep their creation-time radius (only `state.pending` rescales).

## Conventions worth knowing

- **No comments explaining what code does** — block comments at the top of each module explain *why*, and the physics.js header explains the pipeline. Don't add narrating comments inline.
- **Zero build / zero npm.** If you find yourself wanting to add a dep, first try inlining what you need. The import map in `index.html` is the only allowed dep mechanism, and it's currently used only for `d3-quadtree`.
- **`prepareFrame` is the per-frame setup hook.** New per-frame data structures (anything that can be reused across substeps without violating correctness) go there. Per-substep state goes inside `stepPBD`.
- The workspace `CLAUDE.md` (one level up) covers agent/slash-command etiquette, delegation matrix, and the workspace-wide rules. Read it once if you haven't.
