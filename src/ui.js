// ui.js — DOM binding for sliders, buttons, and the segmented type/charge selectors.
//
// Slider semantics depend on whether an entity is selected (edit mode):
//   - No selection → adjust `state.pending` (template for next placement).
//   - Selection    → adjust the selected entity in place; visuals update next frame.
//
// `syncFromSelection()` is called by main.js whenever selection changes, so the
// slider thumbs reflect the chosen entity's current values.

import { state, DEFAULTS, clearEntities, findEntityById } from './state.js';
import { refreshEntityColor } from './entities.js';

const els = {};

export function bindUI() {
  cacheElements();
  bindTypeSegment();
  bindChargeSegment();
  bindRangeSlider('mass', val => applyAttribute('mass', val));
  bindRangeSlider('radius', val => applyAttribute('radius', val));
  bindRangeSlider('trail', val => { state.trailLength = val; });
  bindRangeSlider('time-scale', val => applyTimeScale(val), 2);
  els.pauseBtn.addEventListener('click', togglePause);
  els.clearBtn.addEventListener('click', () => {
    clearEntities();
    syncFromSelection();
  });
  els.editBtn.addEventListener('click', toggleEditMode);

  // Initial sync.
  syncFromSelection();
  updateModeHint();
}

function cacheElements() {
  els.modeHint   = document.getElementById('mode-hint');
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
  return decimals > 0 ? raw.toFixed(decimals) : String(Math.round(raw));
}

// ─── Apply an attribute to selected entity OR pending template ────
function applyAttribute(key, value) {
  const sel = state.selectedId !== null ? findEntityById(state.selectedId) : null;
  if (sel) {
    sel[key] = value;
    refreshEntityColor(sel);
  } else {
    state.pending[key] = value;
  }
}

// ─── Time scale + pause ──────────────────────────────────────────
function applyTimeScale(value) {
  state.timeScale = value;
  if (value > 0) state.prevTimeScale = value;
  updatePauseButtonLabel();
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
  // Sliders
  els.massInput.value = String(src.mass);
  els.massVal.textContent = String(Math.round(src.mass));
  els.radiusInput.value = String(src.radius);
  els.radiusVal.textContent = String(Math.round(src.radius));

  // Selection hint banner
  if (sel) {
    els.selHint.hidden = false;
    els.selHint.textContent = `已选中 #${sel.id} · ${sel.type === 'black_hole' ? '黑洞' : '行星'}`;
  } else {
    els.selHint.hidden = true;
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
