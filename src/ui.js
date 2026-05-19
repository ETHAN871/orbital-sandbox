// ui.js — DOM binding for sliders, buttons, and the segmented type/charge selectors.
//
// Slider semantics depend on whether an entity is selected (edit mode):
//   - No selection → adjust `state.pending` (template for next placement).
//   - Selection    → adjust the selected entity in place; visuals update next frame.
//
// `syncFromSelection()` is called by main.js whenever selection changes, so the
// slider thumbs reflect the chosen entity's current values.

import {
  state, DEFAULTS, RADIUS_BASE, BASE_TIME_SCALE,
  clearEntities, findEntityById, removeEntityById,
} from './state.js';
import { refreshEntityColor } from './entities.js';

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
    syncFromSelection();
  });
  els.editBtn.addEventListener('click', toggleEditMode);
  els.pinBtn.addEventListener('click', togglePinSelected);
  els.deleteBtn.addEventListener('click', deleteSelectedEntity);
  els.pendingPinBtn.addEventListener('click', togglePendingPin);
  els.boundaryBtn.addEventListener('click', toggleBoundaryMode);

  // Initial sync — populate slider-side labels from state defaults so the HTML
  // hardcodes don't silently diverge if DEFAULTS change.
  els.timeVal.textContent = formatVal('time-scale', state.timeScale, 2);
  els.radiusVal.textContent = formatVal('radius', state.pending.radius / RADIUS_BASE, 2);
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
  els.timeVal    = document.getElementById('time-val');
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
  // V4 additions: persistent toggles in the placement panel.
  els.pendingPinBtn = document.getElementById('pending-pin-btn');
  els.boundaryBtn   = document.getElementById('boundary-btn');
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
  input.addEventListener('input', () => {
    const raw = parseFloat(input.value);
    onChange(raw);
    valEl.textContent = formatVal(id, raw, decimals);
  });
}

function formatVal(id, raw, decimals) {
  if (id === 'time-scale') return `${raw.toFixed(decimals)}x`;
  if (id === 'radius') return `${raw.toFixed(decimals)}×`;
  return decimals > 0 ? raw.toFixed(decimals) : String(Math.round(raw));
}

// ─── Apply an attribute to selected entity OR pending template ────
// The `radius` slider value is a *ratio* (slider 1.0 = base 10 px). We
// convert ratio → effective px on the way in so internal entity.radius
// remains in physics-space units everywhere else.
function applyAttribute(key, value) {
  const stored = key === 'radius' ? value * RADIUS_BASE : value;
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

// Top-of-panel time-rate badge — kept in lock-step with the slider value.
// Shows both the user-facing ratio AND the resulting real-time multiplier so
// the user can see "1.00× (= 2× real)" at a glance.
function updateHeaderTime() {
  if (!els.headerTime) return;
  const ratio = state.timeScale;
  const effective = ratio * BASE_TIME_SCALE;
  els.headerTime.textContent = `流速 ${ratio.toFixed(2)}× · 真实 ${effective.toFixed(2)}×`;
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
// Entering / exiting edit mode forces a particular timeScale, but we must
// NOT let those forced changes pollute `state.prevTimeScale` — that field is
// the pause-button's "what to resume to" value. We save/restore it around
// every `applyTimeScale` call so a later resume picks the user-chosen rate,
// not the 0.2x slow-down used during edit.
function toggleEditMode() {
  state.isEditMode = !state.isEditMode;
  if (state.isEditMode) {
    state.prevTimeScaleBeforeEdit = state.timeScale;
    const userPrev = state.prevTimeScale;
    applyTimeScale(DEFAULTS.editTimeScale);
    state.prevTimeScale = userPrev;
    els.timeInput.value = String(DEFAULTS.editTimeScale);
    els.timeVal.textContent = `${DEFAULTS.editTimeScale.toFixed(2)}x`;
  } else {
    const restore = state.prevTimeScaleBeforeEdit ?? DEFAULTS.timeScale;
    const userPrev = state.prevTimeScale;
    applyTimeScale(restore);
    // Only restore the pre-edit `prevTimeScale` if exiting back into a paused
    // state — otherwise the just-applied non-zero restore is exactly what we
    // want as the new "resume" value.
    if (restore === 0) state.prevTimeScale = userPrev;
    els.timeInput.value = String(restore);
    els.timeVal.textContent = `${restore.toFixed(2)}x`;
    state.prevTimeScaleBeforeEdit = null;
    state.selectedId = null;
  }
  els.editBtn.classList.toggle('active', state.isEditMode);
  els.stage.classList.toggle('edit-mode', state.isEditMode);
  updateModeHint();
  syncFromSelection();
}

// ─── Selection ↔ slider sync ─────────────────────────────────────
// Called by main.js after input.js mutates state.selectedId.
export function syncFromSelection() {
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
  const radiusRatio = src.radius / RADIUS_BASE;
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

// ─── Persistent placement toggles (V4) ───────────────────────────
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
