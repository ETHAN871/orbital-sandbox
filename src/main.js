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
import {
  initWebGL,
  drawScene as drawSceneGL,
  drawField,
  drawUI,
  updateTrailCanvas,
  resizeRenderer,
  prepareFrameRenderer,
  updateFieldFades,
} from './renderer-webgl.js?v=20260603-seamfix';
import { attachInput } from './input.js';
import { bindUI, syncFromSelection, updateEntityCount } from './ui.js?v=20260603-seamfix';
import { createBackend } from './physics-backend.js';
import { createFpsMeter } from './fps-meter.js';
import {
  initPerfMonitor,
  markPhase as perfMarkPhase,
  endFrame as perfEndFrame,
  setContext as perfSetContext,
} from './perf-monitor.js';
import { tickQuality, getQualityKnobs, setEnabled as setAqmEnabled } from './quality-manager.js';

// URL escape hatch: `?aqm=off` disables AQM (full quality always).
// Useful when debugging physics artifacts — confirms whether an issue
// stems from AQM's iter/substep clamping or from physics itself.
try {
  if (new URLSearchParams(window.location.search).get('aqm') === 'off') {
    setAqmEnabled(false);
    console.info('[main] AQM disabled via ?aqm=off');
  }
} catch {}

// Field-viz style override via URL. Default is 'screen' (punched fly-screen).
// ?field=screen       — punched fly-screen dimples (default).
// ?field=2d           — centered radial in-plane warp + particle dust.
// ?field=rubber-sheet — oblique rubber-sheet (A/B compare).
// ?field=3d           — bare oblique mesh.
// ?field=legacy       — equipotential contour rings only.
// ?field=curvilinear  — rings + radial field lines.
try {
  const fs = new URLSearchParams(window.location.search).get('field');
  if (fs === 'screen' || fs === '2d' || fs === 'legacy' || fs === '3d' ||
      fs === 'rubber-sheet' || fs === 'curvilinear') {
    state.fieldStyle = fs;
    console.info('[main] field viz style =', fs);
  }
} catch {}
import {
  installRecorder,
  installKeyHandler as installDumpKeyHandler,
  installPersistentHint as installDumpHint,
} from './state-dump.js';

initPerfMonitor();

const MAX_FRAME_DT = 0.1;      // s — cap to prevent spiral-of-death after a stall
const MAX_SUBSTEPS = 8;        // safety net: never run more than N physics steps per frame
const EMBED_TAU = 0.4;         // s — field-viz well sink-in time constant (rubber-sheet settle)
// Phase C: when last frame was visibly slow (>30 ms = below 33 FPS),
// throttle this frame to a single substep. Prevents the heavy-frame
// chain reaction where one expensive dense-cluster frame inflates
// realDt → accumulator overflow → more substeps next frame → spiral.
// Cost: physics runs at half real-time while load persists (visible
// slo-mo). Benefit: cursor stays responsive instead of stuttering.
const SLOW_FRAME_THRESHOLD_MS = 30;
let _lastFrameWasSlow = false;

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

// Phase 1 WebGPU: physics-backend.js selects CPU or WebGPU at startup based
// on navigator.gpu + URL params (?backend=force-cpu / ?backend=verbose).
// On `device.lost` the wrapper transparently swaps the active backend to CPU.
// init() is async — adapter / device / pipeline / priming dispatch — so the
// RAF loop is kicked off only after the backend resolves.
const backend = await createBackend();
await backend.init(state.entities);

// state-dump.js: keyboard D + UI button. The ring buffer is now
// substep-granular — physics-rapier.js calls recordSubstep() at the
// end of each backend.step(), capturing pre-state, gravity vectors,
// post-state, and contact manifolds. The tunables getter just
// snapshots the runtime physics knobs at dump time so the saved JSON
// is self-contained for offline analysis.
installRecorder(state, () => ({
  G:                        state.G,
  epsilon:                  state.epsilon,
  elasticRestitution:       state.elasticRestitution,
  launchSpeedK:             state.launchSpeedK,
  absorptionDuration:       state.absorptionDuration,
  predictHorizon:           state.predictHorizon,
  overlapEscalateThreshold: state.overlapEscalateThreshold,
  overlapCooldownFrames:    state.overlapCooldownFrames,
  overlapBulletThreshold:   state.overlapBulletThreshold,
  timeScale:                state.timeScale,
  trailLength:              state.trailLength,
}));
installDumpKeyHandler();
installDumpHint();

let lastTime = performance.now();
let accumulator = 0;
const fpsMeter = createFpsMeter();   // bottom-right widget + RAF-delta EWMA cross-check
scheduleFrame(frame);

// ─── Loop ──────────────────────────────────────────────────────────
// RAF callbacks themselves are synchronous; the substep loop awaits the
// backend (CPU resolves immediately; GPU awaits the pipelined mapAsync).
// finally → scheduleFrame so a thrown error in runFrame still keeps
// the loop alive (sliders remain reactive; the user sees the console error
// rather than a silently frozen canvas).
function frame(now) {
  fpsMeter.begin();
  runFrame(now)
    .catch(err => console.error('[main] frame failed:', err))
    .finally(() => {
      fpsMeter.end();
      scheduleFrame(frame);
    });
}

// V9.7: dual-driver to keep the renderer alive when the tab is hidden.
//
// Chrome throttles requestAnimationFrame to ~1Hz (sometimes pauses
// entirely) on tabs with document.hidden=true. This kills the
// preview MCP inspector tab — it loads the page successfully but
// then RAF stops firing, so the canvas stays black and
// preview_screenshot returns black. The previous workaround
// (open URL in real browser via PowerShell Start-Process) bypassed
// the inspector but lost the eval/screenshot pipeline.
//
// Root fix: when document.hidden=true, fall back to setTimeout(33ms,
// ~30Hz) so the loop keeps ticking. Visible tab still gets RAF for
// vsync-aligned 60Hz. The 33ms fallback is a deliberate trade — full
// 60Hz when hidden would burn battery on real-user backgrounded
// tabs; 30Hz is smooth enough for headless debugging without
// doubling CPU when the user actually wants the tab idle.
function scheduleFrame(fn) {
  if (document.hidden) {
    setTimeout(() => fn(performance.now()), 33);
  } else {
    requestAnimationFrame(fn);
  }
}

async function runFrame(now) {
  const realDt = Math.min(MAX_FRAME_DT, (now - lastTime) / 1000);
  lastTime = now;

  perfMarkPhase('prepareFrame');
  backend.prepareFrame(state.entities);

  const effectiveRatio = state.isEditMode ? EDIT_MODE_TIME_RATIO : state.timeScale;
  accumulator += realDt * effectiveRatio * BASE_TIME_SCALE;
  // Substep cap is the floor of three signals:
  //   (1) MAX_SUBSTEPS (8) — hard safety net
  //   (2) _lastFrameWasSlow → 1 (binary spiral-of-death guard)
  //   (3) AQM substepBudget → 1-2 (smooth quality knob: at quality=0
  //       we cap at 1 substep, at quality=1 we allow up to 2)
  // AQM gives finer-grained control than the binary (2) guard; both
  // can fire simultaneously, the tighter wins.
  const aqmKnobs = getQualityKnobs();
  const aqmSubstepCap = Math.max(1, Math.round(aqmKnobs.substepBudget));
  const maxSubstepsThisFrame = _lastFrameWasSlow
    ? 1
    : Math.min(MAX_SUBSTEPS, aqmSubstepCap);
  let steps = 0;
  while (accumulator >= SIM_DT && steps < maxSubstepsThisFrame) {
    const isLast = (accumulator - SIM_DT < SIM_DT) || (steps + 1 >= maxSubstepsThisFrame);
    perfMarkPhase('worldStep');
    await backend.step(state.entities, SIM_DT, state.viewport, state.boundaryMode, isLast);
    accumulator -= SIM_DT;
    steps++;
  }
  if (steps >= maxSubstepsThisFrame) accumulator = 0;

  perfMarkPhase('post');
  const simDelta = steps * SIM_DT;
  // Field-viz sink-in: ease each body's `embed` 0→1 (exponential settle) on
  // simulated time, so its membrane well grows smoothly to steady state.
  if (simDelta > 0) {
    const settle = 1 - Math.exp(-simDelta / EMBED_TAU);
    for (let i = 0; i < state.entities.length; i++) {
      const e = state.entities[i];
      if (e.embed === undefined) e.embed = 1;
      else if (e.embed < 1) e.embed = Math.min(1, e.embed + (1 - e.embed) * settle);
    }
  }
  // Field-viz spring-back: detect bodies that left this frame and relax their
  // wells 1→0 (runs unconditionally so UI deletes while paused are caught).
  updateFieldFades(simDelta);
  updateTrailCanvas(simDelta);

  if (state.selectedId !== null) {
    const sel = state.entities.find(e => e.id === state.selectedId);
    if (!sel || sel.absorbing) {
      state.selectedId = null;
      syncFromSelection();
    }
  }

  perfMarkPhase('render');
  // V10: rubber-sheet sag-texture prep must run BEFORE drawSceneGL so
  // entity / trail / UI shaders sample valid sag data. No-op for non-
  // rubber-sheet field styles.
  prepareFrameRenderer();
  drawSceneGL();
  drawField();
  drawUI();

  updateEntityCount();
  perfSetContext({
    N: state.entities.length,
    substepsThisFrame: steps,
    backend: state.backendName,
  });
  // Track whether this frame was slow → next frame will throttle to 1
  // substep to prevent chained slow frames. performance.now() - now is
  // the entire RAF callback duration (includes physics + render + post).
  const frameMs = performance.now() - now;
  _lastFrameWasSlow = frameMs > SLOW_FRAME_THRESHOLD_MS;
  // Feed AQM for adaptive quality ramping. AQM observes p95 dt over
  // a 60-frame rolling window and adjusts iter caps / substep budget
  // / contactsTrace sampling to keep dt close to target FPS.
  tickQuality(frameMs);
  perfEndFrame();
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
