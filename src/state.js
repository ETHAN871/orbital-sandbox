// state.js — Global mutable state + physics/UI constants.
// Single source of truth. Other modules import this and mutate
// it through small helper functions (no direct deep-mutation from UI).

// ─── Physics constants ────────────────────────────────────────────
export const G = 80;            // Gravitational constant, tuned for visual orbits at default params.
export const EPSILON = 4;       // Softening floor for r² to prevent singularity (px).
export const SIM_DT = 1 / 60;   // Fixed physics timestep (s) — independent of frame rate.
export const PREDICT_HORIZON = 5;          // Prediction line spans 5 seconds.
export const PREDICT_DT = 1 / 60;          // Prediction integration step (s).
export const PREDICT_STEPS = Math.floor(PREDICT_HORIZON / PREDICT_DT); // 300 points.
export const LAUNCH_SPEED_K = 0.7;         // Drag-vector → initial-velocity multiplier.
export const ABSORPTION_DURATION = 0.3;    // Seconds (sim time) for the black-hole devour animation.
export const ELASTIC_RESTITUTION = 1.0;    // 1 = perfectly elastic planet-planet collision.

// ─── UI defaults & limits ─────────────────────────────────────────
export const DEFAULTS = Object.freeze({
  type: 'planet',
  mass: 100,
  radius: 20,
  charge: 1,
  trailLength: 100,
  timeScale: 1,
  editTimeScale: 0.2,
});

export const LIMITS = Object.freeze({
  mass:   { min: 1,   max: 1000 },
  radius: { min: 5,   max: 80 },
  trail:  { min: 0,   max: 500 },
  time:   { min: 0,   max: 3 },
});

// ─── Live state ───────────────────────────────────────────────────
// `entities` is the live list of bodies in the simulation.
// `nextId` provides stable identity across re-renders/selection.
// `drag` tracks the current pointer-drag (for placement preview).
// `pending` is the "next-to-place" template editable when nothing is selected.
// `selectedId` is the currently-edited entity in edit mode (or null).
// `prevTimeScale` lets the pause button restore the previous non-zero rate.

export const state = {
  entities: [],
  nextId: 1,

  // UI mode
  isEditMode: false,
  selectedId: null,
  prevTimeScaleBeforeEdit: null,

  // Placement defaults (used when not editing)
  pending: {
    type: DEFAULTS.type,
    mass: DEFAULTS.mass,
    radius: DEFAULTS.radius,
    charge: DEFAULTS.charge,
  },

  // Display / sim
  trailLength: DEFAULTS.trailLength,
  timeScale: DEFAULTS.timeScale,
  prevTimeScale: DEFAULTS.timeScale,

  // Drag-to-place transient state
  drag: null, // null or { startX, startY, currentX, currentY, predictionPath: [{x,y}, ...] }

  // Cached viewport (set by main.js on resize)
  viewport: { width: 0, height: 0 },
};

// ─── Helpers ──────────────────────────────────────────────────────
export function nextEntityId() {
  return state.nextId++;
}

export function findEntityById(id) {
  return state.entities.find(e => e.id === id) || null;
}

export function removeEntityById(id) {
  const idx = state.entities.findIndex(e => e.id === id);
  if (idx >= 0) state.entities.splice(idx, 1);
}

export function clearEntities() {
  state.entities.length = 0;
  state.selectedId = null;
}
