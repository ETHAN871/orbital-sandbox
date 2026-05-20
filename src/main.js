// main.js — Entry point. Wires canvases (WebGL stage + Canvas2D overlay),
// input, UI, and the simulation loop.
//
// Time stepping uses a fixed physics dt (SIM_DT) with an accumulator
// scaled by `state.timeScale`. This decouples physics stability from frame
// rate and lets the user slow/speed/pause time without affecting accuracy.
//
// V9.0a rendering split:
//   - `#stage` canvas → WebGL 2 hot path (background, trail FBO, entity
//     sprites with 9-ghost wrap mirrors). See renderer-webgl.js.
//   - `#overlay` canvas → Canvas2D UI lines (hover ghost, drag preview,
//     prediction line, selection ring, absorbing fallback). See renderer.js.
//   Both canvases share the same CSS-pixel coordinate space (DPR is applied
//   to each backing store independently). Pointer events go to #stage only;
//   #overlay has pointer-events:none.

import { state, SIM_DT, BASE_TIME_SCALE, EDIT_MODE_TIME_RATIO, computeRadiusBase } from './state.js';
import { stepVerlet, handleCollisions, updateAbsorptions, applyBoundary, prepareFrame } from './physics.js';
import {
  initWebGL,
  drawScene as drawSceneGL,
  updateTrailCanvas,
  resizeRenderer,
} from './renderer-webgl.js';
import { drawOverlay } from './renderer.js';
import { attachInput } from './input.js';
import { bindUI, syncFromSelection, updateEntityCount } from './ui.js';

const MAX_FRAME_DT = 0.1;      // s — cap to prevent spiral-of-death after a stall
const MAX_SUBSTEPS = 8;        // safety net: never run more than N physics steps per frame

const stageCanvas = document.getElementById('stage');
const overlayCanvas = document.getElementById('overlay');
const overlayCtx = overlayCanvas.getContext('2d');

// Initialize WebGL 2 on the stage canvas. On failure renderer-webgl.js
// surfaces the error UX and disables itself; we let the loop still tick so
// the (now-static) overlay still renders and panel sliders remain reactive.
const webglOk = initWebGL(stageCanvas);
if (!webglOk) {
  console.error('[main] WebGL 2 unavailable — UI overlay still active, but no stage rendering.');
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
    stepVerlet(state.entities, SIM_DT);
    handleCollisions(state.entities);
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

  // V9.0a: render WebGL first (clears stage to bgColor and draws trails +
  // entities), then clear+draw overlay 2D on top.
  drawSceneGL();
  overlayCtx.clearRect(0, 0, state.viewport.width, state.viewport.height);
  drawOverlay(overlayCtx);

  updateEntityCount();
  requestAnimationFrame(frame);
}

// ─── Canvas / DPR ──────────────────────────────────────────────────
// We render in CSS-pixel units so input coordinates (from clientX/Y minus
// bounding rect) match the physics coord space 1:1. Each canvas backing
// store is DPR-scaled; the overlay's 2D ctx transform compensates, while
// the WebGL stage handles DPR internally via its ortho matrix + viewport.
function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = stageCanvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  stageCanvas.width  = w * dpr;
  stageCanvas.height = h * dpr;
  overlayCanvas.width  = w * dpr;
  overlayCanvas.height = h * dpr;
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

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
