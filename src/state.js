// state.js — Global mutable state + physics/UI constants.
// Single source of truth. Other modules import this and mutate
// it through small helper functions (no direct deep-mutation from UI).

// ─── Physics timing constants (fixed, not user-tunable) ──────────
// SIM_DT and PREDICT_DT are integration step sizes; changing them would
// break Verlet stability and animation timing. Other "constants" that
// USED to live here (G, EPSILON, PREDICT_HORIZON, LAUNCH_SPEED_K,
// ABSORPTION_DURATION, ELASTIC_RESTITUTION, BH_THRESHOLD) are now
// runtime-mutable state fields (see `state.xxx` below), driven by the
// "高级调参" UI panel. `DEFAULTS_TUNING` keeps their original values
// so the "恢复默认" button can reset.
export const SIM_DT = 1 / 60;              // Fixed physics timestep (s).
export const PREDICT_DT = 1 / 60;          // Prediction integration step (s).
// Maximum prediction step count: predictHorizon max = 15s × 60Hz = 900 steps.
// The _predictBuf is pre-allocated at this size so the slider can shrink
// without re-allocation.
export const PREDICT_STEPS_MAX = Math.floor(15 / PREDICT_DT);

// Tunable physics defaults. Frozen so "恢复默认" button reads stable values.
export const DEFAULTS_TUNING = Object.freeze({
  G: 80,                       // Gravitational constant.
  epsilon: 4,                  // Softening floor for r² (px).
  predictHorizon: 5,           // Prediction line span (sec sim time).
  launchSpeedK: 0.7,           // Drag-vector → initial-velocity multiplier.
  absorptionDuration: 0.3,     // Black-hole devour animation length (sec sim).
  elasticRestitution: 0.3,     // 0=inelastic, 1=fully elastic.
  bhThreshold: 64,             // N≥ this → Barnes-Hut, else direct O(N²).
});

// Slider semantics: the time-scale and radius sliders display *ratios* — the
// value the user sees is multiplied by the corresponding base constant to get
// the physical value. Slider 1 = base, slider 2 = 2×base, slider 0.5 = half base.
export const BASE_TIME_SCALE = 2;          // Slider 1.0× = effective 2× real time.

// ─── Dynamic radius base ──────────────────────────────────────────
// Slider 1.0× corresponds to a viewport-relative radius — bigger screens
// get bigger default bodies, but a hard floor keeps them tappable on
// touch / small screens.
//
//   base_px = clamp(min(viewport_w, viewport_h) / RADIUS_DIVISOR,
//                   MIN_RADIUS_PX, MAX_RADIUS_PX)
//
// main.js recomputes state.radiusBase on every setupCanvas (init + resize).
// Existing entities keep their creation-time radius; only `state.pending`
// rescales proportionally so the next-placed body fits the new viewport.

export const RADIUS_DIVISOR = 60;          // Default radius ≈ min(W,H) / 60.
export const MIN_RADIUS_PX  = 14;          // Floor — diameter 28 px, tappable.
export const MAX_RADIUS_PX  = 80;          // Ceiling — avoids defaults eating 4K screens.

export function computeRadiusBase(viewport) {
  const minEdge = Math.max(1, Math.min(viewport.width || 0, viewport.height || 0));
  const dyn = minEdge / RADIUS_DIVISOR;
  return Math.max(MIN_RADIUS_PX, Math.min(MAX_RADIUS_PX, dyn));
}
// Edit-mode *overrides* the effective time ratio to a constant value (slider
// is unchanged — it represents the user's chosen rate for non-edit play).
// effective_in_edit_mode = EDIT_MODE_TIME_RATIO × BASE_TIME_SCALE = 0.4× real.
export const EDIT_MODE_TIME_RATIO = 0.2;

// Boundary: entities are destroyed once they pass beyond
//   max(viewport_w, viewport_h) × BOUNDARY_BUFFER_FACTOR
// away from the viewport edge (gives them some off-screen room before despawn).
// In 'wrap' mode they teleport to the opposite side instead.
export const BOUNDARY_BUFFER_FACTOR = 0.5;

// ─── UI defaults & limits ─────────────────────────────────────────
export const DEFAULTS = Object.freeze({
  type: 'planet',
  mass: 100,
  // Initial effective radius in px (fallback before first viewport setup).
  // After main.js setupCanvas runs, state.pending.radius is rescaled to
  // state.radiusBase × current ratio, so this static value only matters for
  // the brief window before the first canvas measure.
  radius: 14,
  charge: 1,
  trailLength: 100,
  // timeScale is a *ratio*; effective time = timeScale × BASE_TIME_SCALE
  // (overridden to EDIT_MODE_TIME_RATIO × BASE_TIME_SCALE while editing).
  // Default 1.0× ratio → 2× real-time.
  timeScale: 1,
});

export const LIMITS = Object.freeze({
  mass:   { min: 1,   max: 1000 },
  // radius slider is a ratio: 0.1 = tiny (1 px), 10 = huge (100 px).
  radius: { min: 0.1, max: 10 },
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

  // Placement defaults (used when not editing). When `pinned` is true,
  // newly-spawned entities are immediately frozen in place.
  pending: {
    type: DEFAULTS.type,
    mass: DEFAULTS.mass,
    radius: DEFAULTS.radius,
    charge: DEFAULTS.charge,
    pinned: false,
  },

  // Display / sim
  trailLength: DEFAULTS.trailLength,
  timeScale: DEFAULTS.timeScale,
  prevTimeScale: DEFAULTS.timeScale,

  // Drag-to-place transient state
  drag: null, // null or { startX, startY, currentX, currentY, predictionPath: [{x,y}, ...] }

  // Pointer hover position in canvas coords (or null when off-canvas).
  // Used by renderer to draw a ghost outline preview for placement mode.
  hoverPos: null,

  // 'destroy' (default) → entities past viewport+buffer are removed.
  // 'wrap'              → entities wrap to the opposite edge.
  boundaryMode: 'destroy',

  // Cached viewport (set by main.js on resize)
  viewport: { width: 0, height: 0 },

  // Dynamic radius base — see computeRadiusBase(). Recomputed each setupCanvas.
  // Slider value 1.0 corresponds to this many px.
  radiusBase: MIN_RADIUS_PX,

  // Tunable physics params — driven by 高级调参 sliders; reset via
  // DEFAULTS_TUNING.
  G:                  DEFAULTS_TUNING.G,
  epsilon:            DEFAULTS_TUNING.epsilon,
  predictHorizon:     DEFAULTS_TUNING.predictHorizon,
  launchSpeedK:       DEFAULTS_TUNING.launchSpeedK,
  absorptionDuration: DEFAULTS_TUNING.absorptionDuration,
  elasticRestitution: DEFAULTS_TUNING.elasticRestitution,
  bhThreshold:        DEFAULTS_TUNING.bhThreshold,

  // Canvas background color — toggled by the 深/浅 button.
  // '#0a0a0f' = dark default; '#ececf0' = near-white-gray light.
  bgColor: '#0a0a0f',

  // Trail dot width in px (diameter). Renderer derives radius R = trailWidth / 2.
  // Default 3 px → solid 3×3 with 1 px AA edge.
  trailWidth: 3,

  // V9.1: when true, drawField() runs after drawScene() — renders the
  // gravitational equipotential contour lines + synchronized pulsing
  // streamlines as a topographic-style "field flow" visualization. Default
  // OFF so the GPU work is fully gated (zero overhead when not displayed).
  showField: false,
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
