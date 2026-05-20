// main.js — Entry point. Wires the WebGL stage canvas, input, UI, and the
// simulation loop.
//
// Time stepping uses a fixed physics dt (SIM_DT) with an accumulator scaled
// by `state.timeScale`. This decouples physics stability from frame rate
// and lets the user slow/speed/pause time without affecting accuracy.
//
// V9.0b rendering pipeline (single canvas, all WebGL):
//   - `#stage` canvas → WebGL 2. renderer-webgl.js exports drawScene
//     (background + trail FBO + entity sprites + 9-ghost wrap mirrors) and
//     drawUI (hover ghost + drag preview + prediction line + selection
//     ring + absorbing fallback). Both write to the same default framebuffer
//     in CSS-pixel logical coords via a shared ortho matrix.
//   - Pointer events go straight to #stage (no separate overlay canvas to
//     intercept them).

import { state, SIM_DT, BASE_TIME_SCALE, EDIT_MODE_TIME_RATIO, computeRadiusBase } from './state.js';
import { stepPBD, updateAbsorptions, applyBoundary, prepareFrame } from './physics.js';
import {
  initWebGL,
  drawScene as drawSceneGL,
  drawField,
  drawUI,
  updateTrailCanvas,
  resizeRenderer,
} from './renderer-webgl.js';
import { attachInput } from './input.js';
import { bindUI, syncFromSelection, updateEntityCount } from './ui.js';
import { maybeLogEnergy } from './debug-energy.js';

const MAX_FRAME_DT = 0.1;      // s — cap to prevent spiral-of-death after a stall
const MAX_SUBSTEPS = 8;        // safety net: never run more than N physics steps per frame

const stageCanvas = document.getElementById('stage');

// Initialize WebGL 2 on the stage canvas. On failure renderer-webgl.js
// surfaces the error UX (#webgl-error overlay) and disables itself; the
// loop still ticks so panel sliders remain reactive even with no rendering.
const webglOk = initWebGL(stageCanvas);
if (!webglOk) {
  console.error('[main] WebGL 2 unavailable — stage cannot render.');
}

setupCanvas();
window.addEventListener('resize', setupCanvas);

bindUI();
attachInput(stageCanvas, _newId => syncFromSelection());

let lastTime = performance.now();
let accumulator = 0;
requestAnimationFrame(frame);

// ─── Loop ──────────────────────────────────────────────────────────
function frame(now) {
  const realDt = Math.min(MAX_FRAME_DT, (now - lastTime) / 1000);
  lastTime = now;

  // V8.2: build the Barnes-Hut quadtree and spatial hash once per frame so
  // all substeps share them. Worth up to 7x savings on tree-build cost at
  // high substep count. Position drift between substeps is bounded by
  // SIM_DT * |v| and stays within Verlet's existing integration tolerance.
  prepareFrame(state.entities);

  // Effective time ratio = user's slider value, UNLESS edit mode is active
  // (then a fixed override applies, independent of the slider).
  const effectiveRatio = state.isEditMode ? EDIT_MODE_TIME_RATIO : state.timeScale;
  accumulator += realDt * effectiveRatio * BASE_TIME_SCALE;
  let steps = 0;
  while (accumulator >= SIM_DT && steps < MAX_SUBSTEPS) {
    // PBD: stepPBD now bundles broadphase (handleCollisions) inside its
    // pipeline so contact detection happens at predicted positions, not
    // at pre-step ones. main.js only sequences the higher-level phases.
    stepPBD(state.entities, SIM_DT);
    updateAbsorptions(state.entities, SIM_DT);
    applyBoundary(state.entities, state.viewport, state.boundaryMode);
    accumulator -= SIM_DT;
    steps++;
  }
  if (steps >= MAX_SUBSTEPS) accumulator = 0;

  // V8.1: update the phosphor-decay trail FBO once per visual frame.
  // The fade rate is keyed on simulation time (not wall time), so pausing
  // freezes trails and 3× time-scale fades 3× faster.
  const simDelta = steps * SIM_DT;
  updateTrailCanvas(simDelta);

  // Drop selection silently if the selected entity is gone OR mid-absorption.
  if (state.selectedId !== null) {
    const sel = state.entities.find(e => e.id === state.selectedId);
    if (!sel || sel.absorbing) {
      state.selectedId = null;
      syncFromSelection();
    }
  }

  // V9.0b: WebGL draws everything. drawScene clears, blits trail, draws
  // entity sprites. V9.1 inserts drawField between scene and UI (gated by
  // state.showField → zero cost when off). drawUI then overlays hover /
  // drag / prediction / selection / absorbing on the same framebuffer.
  drawSceneGL();
  drawField();
  drawUI();

  updateEntityCount();
  maybeLogEnergy();   // diagnostic — no-op unless state.__debugEnergy is true
  requestAnimationFrame(frame);
}

// ─── Canvas / DPR ──────────────────────────────────────────────────
// We render in CSS-pixel units so input coordinates (from clientX/Y minus
// bounding rect) match the physics coord space 1:1. The canvas backing
// store is DPR-scaled; the WebGL stage handles DPR via gl.viewport +
// an ortho matrix that maps CSS-px → NDC.
function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = stageCanvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  stageCanvas.width  = w * dpr;
  stageCanvas.height = h * dpr;

  state.viewport.width = w;
  state.viewport.height = h;

  // Notify WebGL renderer so it can rebuild the trail FBO at the new
  // viewport size + update its ortho matrix.
  resizeRenderer(w, h, dpr);

  // Recompute the dynamic radius base. Existing entities keep their per-
  // instance radius (no rescale = no surprise resize on browser-zoom or
  // window drag). The "next-to-place" pending body rescales proportionally
  // so the user's slider ratio stays meaningful across viewport changes.
  const prevBase = state.radiusBase;
  const newBase = computeRadiusBase(state.viewport);
  if (prevBase > 0 && newBase !== prevBase && state.pending.radius > 0) {
    const ratio = state.pending.radius / prevBase;
    state.pending.radius = ratio * newBase;
  }
  state.radiusBase = newBase;
  // Refresh slider DOM so the displayed ratio reflects the new base. Safe
  // to call before bindUI has run because syncFromSelection itself is no-op
  // until DOM refs are cached.
  syncFromSelection();
}
