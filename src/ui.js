// ui.js — DOM binding for sliders, buttons, and the segmented type/charge selectors.
//
// Slider semantics depend on whether an entity is selected (edit mode):
//   - No selection → adjust `state.pending` (template for next placement).
//   - Selection    → adjust the selected entity in place; visuals update next frame.
//
// `syncFromSelection()` is called by main.js whenever selection changes, so the
// slider thumbs reflect the chosen entity's current values.

import {
  state, DEFAULTS, DEFAULTS_TUNING,
  BASE_TIME_SCALE, EDIT_MODE_TIME_RATIO,
  clearEntities, findEntityById, removeEntityById,
} from './state.js';
import { setStateDumpEnabled } from './state-dump.js';
import { refreshEntityColor } from './entities.js';
import { resetTrailCanvas } from './renderer-webgl.js?v=20260603-veil3';
import { clearContactState } from './physics.js';

const els = {};

export function bindUI() {
  cacheElements();
  bindTypeSegment();
  bindChargeSegment();
  bindRangeSlider('mass', val => applyAttribute('mass', val));
  bindRangeSlider('radius', val => applyAttribute('radius', val), 2);
  bindRangeSlider('trail', val => { state.trailLength = val; });
  bindRangeSlider('time-scale', val => applyTimeScale(val), 2);
  els.pauseBtn.addEventListener('click', togglePause);
  els.clearBtn.addEventListener('click', () => {
    clearEntities();
    // V8.1: also wipe the phosphor trail canvas — otherwise old dots
    // would linger and fade slowly even after entities are gone.
    resetTrailCanvas();
    // 2026-05-21 memory-leak fix: drop physics module-level caches that
    // hold strong refs to the just-cleared entities. Without these,
    // _contacts[] slots, _prevPairImpulses, and the BH quadtree pin
    // every removed entity until they're overwritten next substep —
    // which never happens when the user pauses + clears, so memory
    // accumulates across long sessions. See physics.js clearContactState
    // for the full rationale.
    clearContactState();
    syncFromSelection();
  });
  els.editBtn.addEventListener('click', toggleEditMode);
  els.pinBtn.addEventListener('click', togglePinSelected);
  els.deleteBtn.addEventListener('click', deleteSelectedEntity);
  els.pendingPinBtn.addEventListener('click', togglePendingPin);
  els.boundaryBtn.addEventListener('click', toggleBoundaryMode);
  els.fieldBtn.addEventListener('click', toggleFieldMode);
  els.bgThemeBtn.addEventListener('click', toggleBgTheme);
  if (els.dumpStateBtn) {
    refreshStateDumpBtn();
    els.dumpStateBtn.addEventListener('click', () => {
      setStateDumpEnabled(!state.stateDumpEnabled);
      refreshStateDumpBtn();
    });
  }

  // Trail-width slider (collapsible "线宽设置" section).
  bindRangeSlider('trail-width', val => { state.trailWidth = val; }, 1);

  // V9.8: field-viz brightness contrast (场可视化 section).
  bindRangeSlider('field-contrast', val => { state.fieldContrast = val; }, 2);
  // V9.9: Jobard-Lefer streamline spacing (CSS-px) for curvilinear grid.
  bindRangeSlider('field-spacing', val => { state.fieldLineSpacing = val | 0; }, 0);
  // Membrane field opacity (0..1), shown as a percentage.
  bindRangeSlider('membrane-opacity', val => { state.membraneOpacity = val; }, 2);
  // V11.1: rubber-sheet viewing tilt (degrees). 90°=top-down (flat),
  // 45°=classic oblique, 30°=strong oblique. Renderer reads as
  // cos(viewTilt * π / 180) and binds as uSagYFactor uniform.
  bindRangeSlider('view-tilt', val => { state.viewTilt = val; }, 0);

  // 7 sliders in the "高级调参" collapsible — each writes directly to its
  // state.* field. Handlers read state.* at use site, so changes take
  // effect on the next frame without any extra wiring.
  bindRangeSlider('tune-G',           val => { state.G = val; },                  0);
  bindRangeSlider('tune-restitution', val => { state.elasticRestitution = val; }, 2);
  bindRangeSlider('tune-K',           val => { state.launchSpeedK = val; },       2);
  bindRangeSlider('tune-absorb',      val => { state.absorptionDuration = val; }, 1);
  bindRangeSlider('tune-eps',         val => { state.epsilon = val; },            1);
  bindRangeSlider('tune-predict',     val => { state.predictHorizon = val; },     1);
  bindRangeSlider('tune-overlap',     val => { state.overlapEscalateThreshold = val; }, 0);
  bindRangeSlider('tune-stiffness',   val => { state.contactStiffness = val; },   0);
  els.tuneResetBtn.addEventListener('click', resetTuning);

  // Initial sync — populate slider-side labels from state defaults so the HTML
  // hardcodes don't silently diverge if DEFAULTS change.
  els.timeVal.textContent = formatVal('time-scale', state.timeScale, 2);
  els.radiusVal.textContent = formatVal('radius', state.pending.radius / state.radiusBase, 2);
  els.trailVal.textContent = formatVal('trail', state.trailLength, 0);
  // Initial sync for new sliders + bg theme button.
  els.trailWidthVal.textContent = formatVal('trail-width', state.trailWidth, 1);
  for (const spec of els.tuneInputs) {
    const v = state[spec.stateKey];
    if (spec.input) spec.input.value = String(v);
    if (spec.valEl) spec.valEl.textContent = formatVal(spec.id, v, spec.decimals);
  }
  refreshBgThemeBtn();
  refreshPendingPinBtn();
  refreshBoundaryBtn();
  syncFromSelection();
  updateModeHint();
  updateHeaderTime();
}

function cacheElements() {
  els.modeHint   = document.getElementById('mode-hint');
  els.headerTime = document.getElementById('header-time');     // prominent time-rate badge
  els.massInput  = document.getElementById('mass');
  els.massVal    = document.getElementById('mass-val');
  els.radiusInput= document.getElementById('radius');
  els.radiusVal  = document.getElementById('radius-val');
  els.trailInput = document.getElementById('trail');
  els.trailVal   = document.getElementById('trail-val');
  els.timeInput  = document.getElementById('time-scale');
  els.timeVal    = document.getElementById('time-scale-val');
  els.pauseBtn   = document.getElementById('pause-btn');
  els.editBtn    = document.getElementById('edit-btn');
  els.clearBtn   = document.getElementById('clear-btn');
  els.selHint    = document.getElementById('selection-hint');
  els.entityCount= document.getElementById('entity-count');
  els.stage      = document.getElementById('stage');
  els.typeBtns   = document.querySelectorAll('[data-type]');
  els.chargeBtns = document.querySelectorAll('[data-charge]');
  // Selection-only controls (shown when an entity is selected in edit mode):
  els.selSection = document.getElementById('selection-controls');
  els.pinBtn     = document.getElementById('pin-btn');
  els.deleteBtn  = document.getElementById('delete-entity-btn');
  els.pendingPinBtn = document.getElementById('pending-pin-btn');
  els.boundaryBtn   = document.getElementById('boundary-btn');
  // V9.1 field visualization toggle.
  els.fieldBtn      = document.getElementById('field-btn');
  // Background theme + trail width + advanced tuning panel.
  els.bgThemeBtn      = document.getElementById('bg-theme-btn');
  els.dumpStateBtn    = document.getElementById('dump-state-btn');
  els.trailWidthInput = document.getElementById('trail-width');
  els.trailWidthVal   = document.getElementById('trail-width-val');
  els.tuneResetBtn    = document.getElementById('tune-reset-btn');
  // For the bulk-reset button, list each advanced-tuning slider with its
  // input/valEl/decimals/stateKey so resetTuning() can iterate uniformly.
  els.tuneInputs = [
    { id: 'tune-G',           stateKey: 'G',                        decimals: 0 },
    { id: 'tune-restitution', stateKey: 'elasticRestitution',       decimals: 2 },
    { id: 'tune-K',           stateKey: 'launchSpeedK',             decimals: 2 },
    { id: 'tune-absorb',      stateKey: 'absorptionDuration',       decimals: 1 },
    { id: 'tune-eps',         stateKey: 'epsilon',                  decimals: 1 },
    { id: 'tune-predict',     stateKey: 'predictHorizon',           decimals: 1 },
    { id: 'tune-overlap',     stateKey: 'overlapEscalateThreshold', decimals: 0 },
    { id: 'tune-stiffness',   stateKey: 'contactStiffness',         decimals: 0 },
  ].map(spec => ({
    ...spec,
    input: document.getElementById(spec.id),
    valEl: document.getElementById(`${spec.id}-val`),
  }));
}

// ─── Segment selectors ────────────────────────────────────────────
function bindTypeSegment() {
  els.typeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.dataset.type;
      setActiveSegment(els.typeBtns, btn);
      applyAttribute('type', type);
    });
  });
}

function bindChargeSegment() {
  els.chargeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const charge = parseInt(btn.dataset.charge, 10);
      setActiveSegment(els.chargeBtns, btn);
      applyAttribute('charge', charge);
    });
  });
}

function setActiveSegment(group, activeBtn) {
  group.forEach(b => {
    const isActive = b === activeBtn;
    b.classList.toggle('active', isActive);
    b.setAttribute('aria-checked', String(isActive));
  });
}

// ─── Range sliders ────────────────────────────────────────────────
function bindRangeSlider(id, onChange, decimals = 0) {
  const input = document.getElementById(id);
  const valEl = document.getElementById(`${id}-val`);
  // Bind BOTH `input` (continuous during drag) and `change` (fires on
  // release). Defensive: some browsers / CSS combos suppress `input` events
  // for range sliders mid-drag; the `change` fallback guarantees the
  // displayed value catches up by release at the latest.
  const handler = () => {
    const raw = parseFloat(input.value);
    onChange(raw);
    valEl.textContent = formatVal(id, raw, decimals);
  };
  input.addEventListener('input', handler);
  input.addEventListener('change', handler);
}

function formatVal(id, raw, decimals) {
  if (id === 'time-scale') return `${raw.toFixed(decimals)}x`;
  if (id === 'radius') return `${raw.toFixed(decimals)}×`;
  // V8.1: trail slider is now a "lifetime in seconds" (slider / 50).
  if (id === 'trail') return `${(raw / 50).toFixed(1)} 秒`;
  // Tunable sliders with appropriate suffixes.
  if (id === 'trail-width')   return `${raw.toFixed(1)} px`;
  if (id === 'tune-absorb')   return `${raw.toFixed(1)} 秒`;
  if (id === 'tune-eps')      return `${raw.toFixed(1)} px`;
  if (id === 'tune-predict')  return `${raw.toFixed(1)} 秒`;
  if (id === 'view-tilt')     return `${Math.round(raw)}°`;
  if (id === 'membrane-opacity') return `${Math.round(raw * 100)}%`;
  return decimals > 0 ? raw.toFixed(decimals) : String(Math.round(raw));
}

// ─── Apply an attribute to selected entity OR pending template ────
// The `radius` slider value is a *ratio* (slider 1.0 = base 10 px). We
// convert ratio → effective px on the way in so internal entity.radius
// remains in physics-space units everywhere else.
function applyAttribute(key, value) {
  const stored = key === 'radius' ? value * state.radiusBase : value;
  const sel = state.selectedId !== null ? findEntityById(state.selectedId) : null;
  if (sel) {
    sel[key] = stored;
    refreshEntityColor(sel);
  } else {
    state.pending[key] = stored;
  }
}

// ─── Time scale + pause ──────────────────────────────────────────
function applyTimeScale(value) {
  state.timeScale = value;
  if (value > 0) state.prevTimeScale = value;
  updatePauseButtonLabel();
  updateHeaderTime();
}

// Top-of-panel time-rate badge — always shows the actual *effective* real-time
// multiplier, which respects edit-mode override. The slider ratio shown is the
// constant override when in edit mode, otherwise the user's slider value.
function updateHeaderTime() {
  if (!els.headerTime) return;
  const ratio = state.isEditMode ? EDIT_MODE_TIME_RATIO : state.timeScale;
  const effective = ratio * BASE_TIME_SCALE;
  const editTag = state.isEditMode ? ' · 编辑慢动作' : '';
  els.headerTime.textContent = `流速 ${ratio.toFixed(2)}× · 真实 ${effective.toFixed(2)}×${editTag}`;
}

function togglePause() {
  if (state.timeScale === 0) {
    // Resume to prev non-zero rate, falling back to default.
    state.timeScale = state.prevTimeScale > 0 ? state.prevTimeScale : DEFAULTS.timeScale;
  } else {
    state.prevTimeScale = state.timeScale;
    state.timeScale = 0;
  }
  els.timeInput.value = String(state.timeScale);
  els.timeVal.textContent = `${state.timeScale.toFixed(2)}x`;
  updatePauseButtonLabel();
  updateHeaderTime();
}

function updatePauseButtonLabel() {
  els.pauseBtn.textContent = state.timeScale === 0 ? '继续' : '暂停';
  els.pauseBtn.classList.toggle('active', state.timeScale === 0);
}

// ─── Edit mode toggle ────────────────────────────────────────────
// Edit mode is now independent of the time slider — the slider keeps the
// user's chosen rate; main.js overrides the *effective* rate to a constant
// EDIT_MODE_TIME_RATIO while editing. So toggling edit mode no longer
// manipulates state.timeScale or the slider DOM at all.
function toggleEditMode() {
  state.isEditMode = !state.isEditMode;
  if (!state.isEditMode) {
    // Clear selection when leaving edit mode — nothing to edit anymore.
    state.selectedId = null;
  }
  els.editBtn.classList.toggle('active', state.isEditMode);
  els.stage.classList.toggle('edit-mode', state.isEditMode);
  updateModeHint();
  updateHeaderTime();        // effective rate badge changes when edit toggles
  syncFromSelection();
}

// ─── Selection ↔ slider sync ─────────────────────────────────────
// Called by main.js after input.js mutates state.selectedId, and also by
// setupCanvas on resize. The setupCanvas call can fire before bindUI
// has cached element refs — guard so we don't crash on undefined DOM.
export function syncFromSelection() {
  if (!els.massInput) return;
  const sel = state.selectedId !== null ? findEntityById(state.selectedId) : null;
  const src = sel ?? state.pending;

  // Type
  els.typeBtns.forEach(b => {
    const active = b.dataset.type === src.type;
    b.classList.toggle('active', active);
    b.setAttribute('aria-checked', String(active));
  });
  // Charge
  els.chargeBtns.forEach(b => {
    const active = parseInt(b.dataset.charge, 10) === src.charge;
    b.classList.toggle('active', active);
    b.setAttribute('aria-checked', String(active));
  });
  // Sliders. radius is stored in entity as px; the slider displays a ratio.
  els.massInput.value = String(src.mass);
  els.massVal.textContent = String(Math.round(src.mass));
  const radiusRatio = src.radius / state.radiusBase;
  els.radiusInput.value = String(radiusRatio);
  els.radiusVal.textContent = `${radiusRatio.toFixed(2)}×`;

  // Selection hint banner + selection-only controls panel.
  if (sel) {
    els.selHint.hidden = false;
    const pinTag = sel.pinned ? ' · 📌已固定' : '';
    els.selHint.textContent = `已选中 #${sel.id} · ${sel.type === 'black_hole' ? '黑洞' : '行星'}${pinTag}`;
    els.selSection.hidden = !state.isEditMode;
    els.pinBtn.textContent = sel.pinned ? '解除固定' : '固定';
    els.pinBtn.classList.toggle('active', sel.pinned);
  } else {
    els.selHint.hidden = true;
    els.selSection.hidden = true;
  }
  updateModeHint();
}

function updateModeHint() {
  if (state.isEditMode) {
    els.modeHint.textContent = '编辑模式 · 点击实体调整属性 · 时间已减慢';
  } else {
    els.modeHint.textContent = '放置模式 · 按住拖拽放置星体';
  }
}

// Footer entity counter — called per frame by main.js.
export function updateEntityCount() {
  els.entityCount.textContent = `${state.entities.length} 个实体`;
}

// ─── Background theme + advanced-tuning handlers ─────────────────
const BG_DARK  = '#0a0a0f';
const BG_LIGHT = '#ececf0';

// Cycle canvas background between dark and light. Panel theme unchanged.
function toggleBgTheme() {
  if (!els.bgThemeBtn) return;
  state.bgColor = state.bgColor === BG_LIGHT ? BG_DARK : BG_LIGHT;
  refreshBgThemeBtn();
}

function refreshBgThemeBtn() {
  if (!els.bgThemeBtn) return;
  const isLight = state.bgColor === BG_LIGHT;
  els.bgThemeBtn.textContent = isLight ? '深色背景' : '浅色背景';
  els.bgThemeBtn.classList.toggle('active', isLight);
}

// Reset all 7 advanced-tuning sliders to DEFAULTS_TUNING.
function resetTuning() {
  for (const spec of els.tuneInputs) {
    const def = DEFAULTS_TUNING[spec.stateKey];
    state[spec.stateKey] = def;
    if (spec.input) spec.input.value = String(def);
    if (spec.valEl) spec.valEl.textContent = formatVal(spec.id, def, spec.decimals);
  }
}

// ─── Selection-only handlers ─────────────────────────────────────
// Toggle pinned on the selected entity. Visual + selection-hint refresh
// happens via syncFromSelection.
function togglePinSelected() {
  const sel = state.selectedId !== null ? findEntityById(state.selectedId) : null;
  if (!sel) return;
  sel.pinned = !sel.pinned;
  if (sel.pinned) {
    // Halt any residual motion immediately — feels weird otherwise.
    sel.vx = 0; sel.vy = 0; sel.ax = 0; sel.ay = 0;
  }
  syncFromSelection();
}

// Remove the currently-selected entity and clear selection. Distinct from
// the "clear all" sandbox button which wipes everything.
function deleteSelectedEntity() {
  if (state.selectedId === null) return;
  removeEntityById(state.selectedId);
  state.selectedId = null;
  syncFromSelection();
}

// ─── Persistent placement toggles ────────────────────────────────
// Flip `state.pending.pinned`. Newly placed bodies will spawn pinned.
function togglePendingPin() {
  state.pending.pinned = !state.pending.pinned;
  refreshPendingPinBtn();
}

function refreshPendingPinBtn() {
  if (!els.pendingPinBtn) return;
  els.pendingPinBtn.classList.toggle('active', state.pending.pinned);
  els.pendingPinBtn.textContent = state.pending.pinned
    ? '📌 创建为固定（开）'
    : '创建为固定';
}

// Toggle wrap-around vs destroy boundary mode.
function toggleBoundaryMode() {
  state.boundaryMode = state.boundaryMode === 'wrap' ? 'destroy' : 'wrap';
  refreshBoundaryBtn();
}

function refreshBoundaryBtn() {
  if (!els.boundaryBtn) return;
  const wrap = state.boundaryMode === 'wrap';
  els.boundaryBtn.classList.toggle('active', wrap);
  els.boundaryBtn.textContent = wrap ? '循环边界（开）' : '循环边界';
}

// V9.1: toggle the gravitational field overlay (equipotential contours +
// synchronized pulsing streamlines). When OFF the renderer skips drawField
// entirely, so the GPU cost is zero — gated check happens in main.js
// frame loop.
function toggleFieldMode() {
  state.showField = !state.showField;
  refreshFieldBtn();
}

function refreshFieldBtn() {
  if (!els.fieldBtn) return;
  els.fieldBtn.classList.toggle('active', state.showField);
  els.fieldBtn.textContent = state.showField ? '隐藏场' : '显示场';
}

function refreshStateDumpBtn() {
  if (!els.dumpStateBtn) return;
  const on = !!state.stateDumpEnabled;
  els.dumpStateBtn.classList.toggle('active', on);
  els.dumpStateBtn.textContent = on
    ? '状态录制：开 (按 D 截图)'
    : '状态录制：关 (点击开启)';
}
