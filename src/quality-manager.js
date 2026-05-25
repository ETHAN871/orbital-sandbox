// quality-manager.js — Adaptive Quality Manager (AQM v1).
//
// Replaces a dozen hand-tuned hardcoded constants with a single dial:
// "target frame time". The manager observes rolling per-frame dt (the
// real user-perceived latency, not RAF-rate "FPS") and adjusts solver
// iter caps, substep frequency, and contactsTrace sampling on a slow
// ramp to keep dt close to target.
//
// Why this instead of more hardcoded constants:
//   - User hardware varies (laptop vs desktop vs mobile)
//   - Scene complexity varies (10 vs 500 bodies)
//   - GPU availability varies (#1 crossover detector handles GPU path)
//   - The "right" iter cap at N=200 is wrong at N=500 — and vice versa
// A single dial trades quality for FPS smoothly across both axes.
//
// Quality level semantics:
//   1.0 = full quality (iter cap 8/4, 2 substeps/frame, contactSample 3)
//   0.5 = mid     (iter cap 6/3, ~1.5 effective substeps, contactSample 7)
//   0.0 = minimum (iter cap 4/2, 1 substep, contactSample 10)
// Substep budget is fractional — at budget 1.6 we get 2 substeps when
// accumulator allows, but the substep-throttle in main.js also gates
// on previous-frame slowness.
//
// Ramp:
//   - p95 > targetFrameMs × 1.5: lower quality by 0.05
//   - p95 < targetFrameMs × 0.7 AND quality < 1.0: raise by 0.02
//   - Lower ramp is 2.5× faster than raise (don't overcorrect upwards)
//   - Window: 60 frames; first 30 are warmup before any adjustment

const DEFAULT_TARGET_FRAME_MS = 1000 / 60;
const WINDOW_FRAMES = 60;
const WARMUP_FRAMES = 30;
const LOWER_TRIGGER_RATIO = 1.5;
const RAISE_TRIGGER_RATIO = 0.7;
const LOWER_STEP = 0.05;
const RAISE_STEP = 0.02;
const ADJUST_COOLDOWN_FRAMES = 15;

// Knob bounds at quality=0 and quality=1.
const KNOB_BOUNDS = {
  velIterCap:           { lo: 4, hi: 8 },
  posIterCap:           { lo: 2, hi: 4 },
  substepBudget:        { lo: 1.0, hi: 2.0 },
  contactSampleEveryN:  { lo: 10, hi: 3 },   // higher Q → more frequent
  sleepLinearThreshold: { lo: 8.0, hi: 5.0 }, // higher Q → tighter
};

let _enabled = true;
let _targetMs = DEFAULT_TARGET_FRAME_MS;
let _quality = 1.0;
let _frameMsWindow = [];
let _framesSinceAdjust = 0;

function lerp(a, b, t) { return a + (b - a) * t; }

function p95(arr) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
}

export function setTargetFps(fps) {
  _targetMs = 1000 / Math.max(15, Math.min(240, fps));
}

export function setEnabled(on) {
  _enabled = !!on;
  if (!on) _quality = 1.0;
}

// Call once per RAF with the most recent frame's duration in ms.
// Adjusts quality on a slow ramp.
export function tickQuality(frameMs) {
  if (!_enabled) return;
  _frameMsWindow.push(frameMs);
  if (_frameMsWindow.length > WINDOW_FRAMES) _frameMsWindow.shift();
  _framesSinceAdjust++;
  if (_frameMsWindow.length < WARMUP_FRAMES) return;
  if (_framesSinceAdjust < ADJUST_COOLDOWN_FRAMES) return;
  const p = p95(_frameMsWindow);
  if (p > _targetMs * LOWER_TRIGGER_RATIO && _quality > 0.0) {
    _quality = Math.max(0.0, _quality - LOWER_STEP);
    _framesSinceAdjust = 0;
  } else if (p < _targetMs * RAISE_TRIGGER_RATIO && _quality < 1.0) {
    _quality = Math.min(1.0, _quality + RAISE_STEP);
    _framesSinceAdjust = 0;
  }
}

// Returns the live knob set. Physics + main.js read this each frame
// instead of hardcoded constants.
//
// Interaction note: when AQM lowers velIterCap below the manager's
// requested value, the manager's recordPostStep counter keeps rising
// and decideIterations keeps returning the higher number — which AQM
// silently clamps down. The two systems do NOT oscillate (manager
// doesn't observe its own clamped output) but contact residuals can
// be higher than either system alone would predict. This is by
// design: trading penetration accuracy for frame budget.
//
// Wired knobs (consumed by callers):
//   - velIterCap, posIterCap → physics-rapier.js step() solver clamp
//   - contactSampleEveryN → contactsTrace sampling rate
//   - substepBudget → main.js maxSubstepsThisFrame cap
// Unwired knobs (returned for future use, no caller reads them yet):
//   - sleepLinearThreshold → would need world.integrationParameters
//     setter call each frame; not worth the complexity vs payoff.
export function getQualityKnobs() {
  if (!_enabled) {
    return {
      quality: 1.0,
      velIterCap: KNOB_BOUNDS.velIterCap.hi,
      posIterCap: KNOB_BOUNDS.posIterCap.hi,
      substepBudget: KNOB_BOUNDS.substepBudget.hi,
      contactSampleEveryN: KNOB_BOUNDS.contactSampleEveryN.hi,
      sleepLinearThreshold: KNOB_BOUNDS.sleepLinearThreshold.hi,
    };
  }
  const q = _quality;
  return {
    quality: q,
    velIterCap: Math.round(lerp(KNOB_BOUNDS.velIterCap.lo, KNOB_BOUNDS.velIterCap.hi, q)),
    posIterCap: Math.round(lerp(KNOB_BOUNDS.posIterCap.lo, KNOB_BOUNDS.posIterCap.hi, q)),
    substepBudget: lerp(KNOB_BOUNDS.substepBudget.lo, KNOB_BOUNDS.substepBudget.hi, q),
    contactSampleEveryN: Math.max(1, Math.round(lerp(KNOB_BOUNDS.contactSampleEveryN.lo, KNOB_BOUNDS.contactSampleEveryN.hi, q))),
    sleepLinearThreshold: lerp(KNOB_BOUNDS.sleepLinearThreshold.lo, KNOB_BOUNDS.sleepLinearThreshold.hi, q),
  };
}

export function getQualityStats() {
  return {
    enabled: _enabled,
    quality: _quality,
    targetFrameMs: _targetMs,
    windowFrames: _frameMsWindow.length,
    p95: p95(_frameMsWindow),
  };
}

export function resetQuality() {
  _quality = 1.0;
  _frameMsWindow.length = 0;
  _framesSinceAdjust = 0;
}
