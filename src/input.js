// input.js — Pointer interaction.
//
// Two distinct interaction modes, gated by `state.isEditMode`:
//
//   Placement mode (default):
//     pointerdown  → record drag origin
//     pointermove  → update drag end + recompute 5-second prediction
//     pointerup    → spawn an entity using drag vector as initial velocity
//
//   Edit mode:
//     pointerdown  → record candidate-click position
//     pointerup    → if pointer barely moved, treat as click → hit-test entities
//                    → set state.selectedId (or null if clicked empty)
//
// We use pointer events so mouse, touch, and pen all work.

import { state, LAUNCH_SPEED_K } from './state.js';
import { createEntity, randomPlanetColor } from './entities.js';
import { predictTrajectory } from './physics.js';

const CLICK_MOVEMENT_THRESHOLD = 5;     // px — distinguishes click vs drag in edit mode
const HIT_TEST_BUFFER_PX = 8;           // px — expands tap zone past visible edge for ergonomic touch

let canvasRef = null;
let onSelectionChangeCb = () => {};
let clickStart = null;                  // { x, y } for edit-mode click detection
let predictionRafToken = 0;             // RAF id, 0 when no prediction recompute is pending

export function attachInput(canvas, onSelectionChange) {
  canvasRef = canvas;
  onSelectionChangeCb = onSelectionChange || (() => {});

  canvas.addEventListener('pointerdown', handlePointerDown);
  canvas.addEventListener('pointermove', handlePointerMove);
  canvas.addEventListener('pointerup', handlePointerUp);
  canvas.addEventListener('pointerleave', handlePointerLeave);
  canvas.addEventListener('contextmenu', e => e.preventDefault());
}

// ─── Coordinate helpers ────────────────────────────────────────────
// Canvas backing-store is DPR-scaled in main.js. We convert client px → CSS px
// (the logical coord system the physics uses) by dividing out DPR.
function eventToCanvasCoords(ev) {
  const rect = canvasRef.getBoundingClientRect();
  return {
    x: ev.clientX - rect.left,
    y: ev.clientY - rect.top,
  };
}

// ─── Event handlers ────────────────────────────────────────────────
function handlePointerDown(ev) {
  // Only respond to primary button (touch / pen also report button=0).
  if (ev.button !== 0) return;
  canvasRef.setPointerCapture(ev.pointerId);
  const { x, y } = eventToCanvasCoords(ev);

  if (state.isEditMode) {
    clickStart = { x, y };
    return;
  }

  // Placement: open a drag with zero-velocity initial prediction.
  state.drag = {
    startX: x,
    startY: y,
    currentX: x,
    currentY: y,
    previewBaseColor: state.pending.type === 'planet' ? randomPlanetColor() : '#000000',
    predictionPath: [],
  };
  schedulePrediction();
}

function handlePointerMove(ev) {
  const { x, y } = eventToCanvasCoords(ev);
  // Always track hover for the placement-mode ghost outline. Renderer
  // gates on `!isEditMode && !drag` itself.
  state.hoverPos = { x, y };

  if (state.isEditMode) return;
  if (!state.drag) return;

  state.drag.currentX = x;
  state.drag.currentY = y;
  schedulePrediction();
}

function handlePointerUp(ev) {
  const { x, y } = eventToCanvasCoords(ev);
  if (canvasRef.hasPointerCapture(ev.pointerId)) {
    canvasRef.releasePointerCapture(ev.pointerId);
  }

  if (state.isEditMode) {
    if (!clickStart) return;
    const dx = x - clickStart.x;
    const dy = y - clickStart.y;
    if (dx * dx + dy * dy <= CLICK_MOVEMENT_THRESHOLD * CLICK_MOVEMENT_THRESHOLD) {
      const hit = hitTestEntity(x, y);
      const newId = hit ? hit.id : null;
      if (newId !== state.selectedId) {
        state.selectedId = newId;
        onSelectionChangeCb(newId);
      } else if (newId !== null) {
        // Click the already-selected entity → deselect.
        state.selectedId = null;
        onSelectionChangeCb(null);
      }
    }
    clickStart = null;
    return;
  }

  // Placement commit — slingshot semantics: drag *back* to launch *forward*.
  // The initial velocity is the *negative* of the drag vector.
  if (!state.drag) return;
  const dragVecX = state.drag.currentX - state.drag.startX;
  const dragVecY = state.drag.currentY - state.drag.startY;
  const ent = createEntity({
    type: state.pending.type,
    x: state.drag.startX,
    y: state.drag.startY,
    vx: -dragVecX * LAUNCH_SPEED_K,
    vy: -dragVecY * LAUNCH_SPEED_K,
    mass: state.pending.mass,
    radius: state.pending.radius,
    charge: state.pending.charge,
    pinned: state.pending.pinned,
  });
  state.entities.push(ent);
  state.drag = null;
  cancelPendingPrediction();
}

function handlePointerLeave() {
  // Cancel any in-flight drag if the pointer leaves the canvas; commit nothing.
  if (state.drag) state.drag = null;
  clickStart = null;
  cancelPendingPrediction();
  state.hoverPos = null;
}

function cancelPendingPrediction() {
  if (predictionRafToken) {
    cancelAnimationFrame(predictionRafToken);
    predictionRafToken = 0;
  }
}

// ─── Helpers ───────────────────────────────────────────────────────
// `pointermove` can fire many times per frame, especially with coalescing.
// We defer the (relatively expensive) 5-second trajectory recompute to
// at-most-once-per-RAF so the work tracks display refresh, not input rate.
function schedulePrediction() {
  if (predictionRafToken) return;
  predictionRafToken = requestAnimationFrame(() => {
    predictionRafToken = 0;
    updatePrediction();
  });
}

function updatePrediction() {
  const d = state.drag;
  if (!d) return;
  // Slingshot reversal: launch velocity is opposite the drag vector, so the
  // prediction curve goes the opposite way from the rubber-band handle.
  const ghost = {
    x: d.startX,
    y: d.startY,
    vx: -(d.currentX - d.startX) * LAUNCH_SPEED_K,
    vy: -(d.currentY - d.startY) * LAUNCH_SPEED_K,
    radius: state.pending.radius,
  };
  d.predictionPath = predictTrajectory(ghost, state.entities);
}

// Topmost-first hit test: iterate in reverse so visually-front entities win.
// Skip entities mid-absorption — they're shrinking out of existence and
// shouldn't accept clicks. Hit radius is visible radius + buffer so users
// can tap slightly outside the body on touch / small targets.
function hitTestEntity(x, y) {
  for (let i = state.entities.length - 1; i >= 0; i--) {
    const e = state.entities[i];
    if (e.absorbing) continue;
    const dx = x - e.x;
    const dy = y - e.y;
    const hitR = e.radius + HIT_TEST_BUFFER_PX;
    if (dx * dx + dy * dy <= hitR * hitR) return e;
  }
  return null;
}
