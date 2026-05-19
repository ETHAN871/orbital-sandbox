// main.js — Entry point. Wires canvas, input, UI, and the simulation loop.
//
// Time stepping uses a fixed physics dt (SIM_DT) with an accumulator
// scaled by `state.timeScale`. This decouples physics stability from frame
// rate and lets the user slow/speed/pause time without affecting accuracy.

import { state, SIM_DT, BASE_TIME_SCALE, EDIT_MODE_TIME_RATIO, computeRadiusBase } from './state.js';
import { stepVerlet, handleCollisions, appendTrail, updateAbsorptions, applyBoundary } from './physics.js';
import { drawScene } from './renderer.js';
import { attachInput } from './input.js';
import { bindUI, syncFromSelection, updateEntityCount } from './ui.js';

const MAX_FRAME_DT = 0.1;      // s — cap to prevent spiral-of-death after a stall
const MAX_SUBSTEPS = 8;        // safety net: never run more than N physics steps per frame

const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d');

setupCanvas();
window.addEventListener('resize', setupCanvas);

bindUI();
attachInput(canvas, _newId => syncFromSelection());

let lastTime = performance.now();
let accumulator = 0;
requestAnimationFrame(frame);

// ─── Loop ──────────────────────────────────────────────────────────
function frame(now) {
  const realDt = Math.min(MAX_FRAME_DT, (now - lastTime) / 1000);
  lastTime = now;

  // Effective time ratio = user's slider value, UNLESS edit mode is active
  // (then a fixed override applies, independent of the slider). This keeps the
  // slider as user's persistent "main rate" and the edit slowdown as a
  // temporary constant overlay.
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
  // Drop accumulated lag if we hit the substep cap (avoids permanent slowdown).
  if (steps >= MAX_SUBSTEPS) accumulator = 0;

  // Sample trails once per visual frame, BUT only when at least one physics
  // substep actually ran. If paused (timeScale=0 → steps=0), entities haven't
  // moved, so pushing a duplicate trail point is pure waste (and would still
  // pay the ring-buffer advance + size cap update).
  if (steps > 0) {
    for (const e of state.entities) appendTrail(e, state.trailLength);
  }

  // Drop selection silently if the selected entity is gone OR mid-absorption
  // (it's effectively dying — editing its sliders would be pointless).
  if (state.selectedId !== null) {
    const sel = state.entities.find(e => e.id === state.selectedId);
    if (!sel || sel.absorbing) {
      state.selectedId = null;
      syncFromSelection();
    }
  }

  drawScene(ctx);
  updateEntityCount();
  requestAnimationFrame(frame);
}

// ─── Canvas / DPR ──────────────────────────────────────────────────
// We render in CSS-pixel units so input coordinates (from clientX/Y minus
// bounding rect) match the physics coord space 1:1. The canvas backing
// store is DPR-scaled, and the context transform compensates.
function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  state.viewport.width = w;
  state.viewport.height = h;

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
