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

## Preview MCP — recurring-mistake checklist (READ BEFORE TOUCHING PREVIEW)

This trap has been hit **at least 5 times** across sessions. The lesson decays between sessions. Follow this checklist verbatim.

### Symptom

`preview_eval` reports `url: "chrome-error://chromewebdata/"`, dynamic imports fail with `Failed to fetch dynamically imported module: chrome-error://chromewebdata/src/state.js`, HUD/DOM probes return null/false.

### Correct recovery (in order)

1. **Navigate via `window.open(url, '_self')` — NOT `location.href = ...`, NOT `location.reload()`**. Chrome-error pages disable `location.*` assignments for security.
2. **Append a UNIQUE query (e.g., `?perf=1&t=${Date.now()}`)** so the navigation isn't a same-URL no-op.
3. **Wait AT LEAST 3-5 seconds** in the SAME eval before checking. Rapier WASM init alone is 130-160 ms; module fetch + parse adds another 500-1500 ms.
4. **If the eval returns `"Inspected target navigated or closed"` — THAT IS THE SUCCESS SIGNAL**, not an error. It means the page navigated away mid-eval, which is what we wanted. **Issue a FRESH eval to query the now-loaded state**.
5. **DO NOT `preview_stop` then `preview_start`** unless `preview_list` shows the server is genuinely dead. Stop+start orphans the browser tab and forces it deeper into chrome-error. The fix for "tab stuck" is window.open, not server restart.

### Mistakes I keep making (call them out before repeating)

- Treating `"Inspected target navigated or closed"` as failure → it's success.
- Checking `location.href` ≤ 500 ms after `window.open` → navigation hasn't finished. Wait longer.
- Calling `preview_stop` + `preview_start` when the page is just stuck → makes it worse, not better.
- Using `location.href = ...`, `location.assign(...)`, `location.replace(...)`, or `<a>.click()` from chrome-error → all of these silently no-op in that context.
- Forgetting that ESM module cache is keyed by URL **without query string** — main.js cache-buster `?v=X` does NOT invalidate sub-module imports. If a sub-module change isn't taking effect, hard-reload via window.open with new query on the entry URL, not by bumping main.js version.

### One-shot probe template (always use this)

```js
(async () => {
  window.open('http://localhost:8123/?perf=1&t=' + Date.now(), '_self');
  await new Promise(r => setTimeout(r, 5000));   // patience — Rapier WASM init
  return { url: location.href, hud: !!document.getElementById('perf-hud') };
})();
// If eval rejects with "Inspected target navigated or closed" — it WORKED.
// Issue a fresh eval to read state. Don't conclude failure from that error.
```

## /feature-dev default mode: autonomous iteration

**User-declared default (2026-05-25):** any `/feature-dev` invocation in this
project SHOULD use autonomous problem-discovery iteration unless the user
explicitly says otherwise. That means:

- Skip the "wait for approval after planning" step. Make reasonable design
  calls and proceed.
- Skip clarifying-question rounds for low-stakes decisions. Pick the
  recommended option and document the choice in a comment or commit msg.
- DO ask via `AskUserQuestion` ONLY when the choice is genuinely high-stakes,
  irreversible, or the user-supplied request is missing essential info that
  cannot be inferred from codebase patterns.
- Drive the loop yourself: hypothesis → measurement → fix → verify → repeat.
  Use the per-iteration discipline from the Optimization-loop methodology
  section below.
- code-reviewer pass after each implementation batch is still mandatory.
- Commit + deploy at the end without asking permission (unless commit is
  on a protected branch or the change is genuinely destructive).

The user's mental model: `/feature-dev` = "go figure it out, ship it, tell
me what you did." Not "draft a plan, wait for me, then maybe build it."

If the project's CLAUDE.md, `/feature-dev` skill, or workspace rules
ever conflict with this default, the workspace rules win — but a friendly
inline reminder of this preference is fine.

## Optimization-loop methodology (LEARN FROM PAST WINS)

The session ending 2026-05-25 had a **6-stage thrash** (Stages 0-6 produced
modest gains despite many changes) followed by a **3-iteration breakthrough**
(Iter 1-3 achieved 7-8× FPS improvement at the user's target scenario,
30 → 233 FPS at N = 200 dense). What flipped wasn't WHAT was changed but
HOW. These are the engineering-method lessons, not the specific facts.

### Methodology principles

1. **Build the right instrumentation before changing anything.** The
   thrash phase had a "FPS meter" that measured RAF callback rate —
   meaningless when every frame was throwing inside a try / catch. The
   wins started when perf-monitor got per-phase breakdown + `interFrame`
   capture + `worstFrames` attribution. Without finer measurement, you
   are guessing — and guesses compound badly across many iterations.

2. **Verify the measurement is honest before trusting any data.** A
   silent renderer crash made the FPS read 218 while no frame actually
   completed. Sanity check: when a metric correlates badly with the
   user's lived experience, suspect the metric first — not the code
   under measurement.

3. **One iteration = one hypothesis from one piece of data → one
   targeted fix → one verification.** Don't bundle multiple guesses.
   Iter 1: "where is the 49 ms unaccounted gap?" → add interFrame →
   confirmed browser-side. Iter 2: "what's 27 ms of contactsTrace?" →
   split full readManifold vs fast count → 27 → 0.3 ms. Iter 3: "why
   is gravity 17 ms with GPU enabled?" → measure CPU equivalent → raise
   threshold → 17 → 0.6 ms. Each step reversible; each verifies the
   hypothesis before the next moves.

4. **When the measurement-loop friction is high, fix the loop first.**
   Lost ~3 cycles fighting preview-MCP chrome-error before recognizing
   the recovery pattern was already documented. Slow / broken inner
   loop costs compound exponentially across many iterations — drop
   physics work and fix the loop instead.

5. **Recognize "metric vs reality" mismatches early.** "Inspected
   target navigated or closed" looks like an error, is actually a
   success signal (page navigated mid-eval — what you wanted).
   "longTaskCount = 0" doesn't mean "no stalls" — a 79 ms frame can
   split across multiple short tasks via await boundaries.

6. **Small composable steps beat big architectural reaches.** Three
   ~30-min iterations produced 7-8× speedup. The "big" alternatives
   (Stage 7 Worker, ~2-week rewrite) weren't needed yet. Don't reach
   for the structural refactor while small targeted fixes still produce
   measurable wins. The path is: small-cheap-fix → measure → if the
   metric stops improving, THEN reach bigger.

7. **Stop iterating when the algorithmic floor is the wall.** At
   N = 700, CPU O(N²/2) gravity hit ~10 ms and Rapier internal solver
   hit ~10 ms. Both are physics-layer floors, not JS overhead. Honest
   acknowledgment > another speculative tweak. The next move from
   there is structural (worker, different engine), not parametric.

### When to escalate to `code-architect`

- **Don't** call it for measurement instrumentation, data-driven
  one-line constant changes, or surgical bug fixes. Those are
  mechanical and well-served by self-plan.
- **Do** call it for structural decisions: changing the threading
  model (Worker), switching physics engines, redesigning the substep
  loop. These have wide blast radius and the architect's "would this
  break X / Y / Z" surfacing is genuinely valuable.
- Iter 1-3 used zero architect calls because data was clear and fixes
  were targeted. Over-using architect for small surgical changes is a
  stalling tactic disguised as caution.

### Just-in-time pivot patterns (recognize these signals)

| Signal | Pivot to |
|---|---|
| All metrics look fine but user reports breakage | The meter is lying. Verify what each number REPRESENTS, not what it looks like. |
| Optimization X "should" help but doesn't | Hypothesis was based on a phase you're not measuring. Add finer instrumentation, retest. |
| Inner loop (preview / build / test) takes > 30 s | Stop pushing changes; spend 1 hour fixing the loop. |
| Multiple consecutive iterations show diminishing returns | You hit the floor — admit it and document. Don't fake a 0.1 % win. |
| A measured phase is < 1 ms but the frame is 30 ms | Work is happening in unmeasured space (GC, browser, sub-modules). Add a catch-all phase like `interFrame`. |
| Eval / probe error message looks like failure | Disambiguate by reading the actual error text. Some look bad but mean success. |
| You're tempted to call architect for a 5-line change | Just do it. Architect is for "what breaks across X / Y / Z" not "should I lower this constant." |
| Same problem appears N+1 times with N already > 2 | Add the lesson to CLAUDE.md as METHOD (not fact) before doing the work. Decay is real. |
