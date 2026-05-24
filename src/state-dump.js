// state-dump.js — runtime snapshot recorder for diagnosing live bugs.
//
// Records a rolling ring of the last RING_SIZE frames. When the user
// presses D (or clicks the "状态快照" UI button), POSTs the buffer to
// the dev server's /dump endpoint, which persists it to disk so the
// dev (or Claude reading via the Read tool) can analyze the precise
// numeric trajectory of every entity around the moment a bug surfaced.
//
// The ring buffer captures three seconds of context (at 60 Hz) so the
// dev can see the lead-up + the bug moment + the immediate aftermath.
//
// Engine-agnostic: this module knows nothing about Rapier or planck.
// The caller passes in a `backendSnapshot` per frame which the
// backend constructs in its own snapshot() method (contact pairs,
// sleeping state, internal solver counters, etc.).

const RING_SIZE = 180;
const ring = new Array(RING_SIZE);
let writeIdx = 0;
let totalRecorded = 0;

let _state = null;
let _backendRef = null;     // backend wrapper from physics-backend.js (must have snapshot() method)
let _getTunables = null;    // () → object snapshot of all state tunables

export function installRecorder(state, backend, getTunables) {
  _state = state;
  _backendRef = backend;
  _getTunables = getTunables;
}

export function recordFrame(frameMeta) {
  if (!_state) return;
  const backendSnapshot = (_backendRef && typeof _backendRef.snapshot === 'function')
    ? _backendRef.snapshot()
    : null;
  const sleepingMap = backendSnapshot?.bodyStates || {};
  const entities = _state.entities;
  const out = new Array(entities.length);
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    const bs = sleepingMap[e.id];
    out[i] = {
      id: e.id,
      type: e.type,
      x: +e.x.toFixed(3),
      y: +e.y.toFixed(3),
      vx: +(e.vx || 0).toFixed(3),
      vy: +(e.vy || 0).toFixed(3),
      mass: e.mass,
      radius: +(e.radius || 0).toFixed(3),
      charge: e.charge,
      pinned: !!e.pinned,
      absorbing: e.absorbing !== null,
      sleeping: bs ? !!bs.sleeping : null,
      ccd:      bs ? !!bs.ccd      : null,
    };
  }
  ring[writeIdx] = {
    frame: totalRecorded,
    wallTimeMs: +performance.now().toFixed(1),
    substepsRun: frameMeta?.substepsRun ?? 0,
    wrappedEntityIds: frameMeta?.wrappedEntityIds || [],
    entities: out,
    contacts: backendSnapshot?.contacts || [],
    solverIters: backendSnapshot?.solverIters || null,
  };
  writeIdx = (writeIdx + 1) % RING_SIZE;
  totalRecorded++;
}

export async function dumpToServer(label = 'manual') {
  if (!_state) return { ok: false, reason: 'recorder not installed' };
  const ringOrdered = [];
  // Read out the ring in chronological order (oldest → newest).
  const oldest = totalRecorded < RING_SIZE
    ? 0
    : writeIdx;
  for (let i = 0; i < Math.min(RING_SIZE, totalRecorded); i++) {
    const slot = ring[(oldest + i) % RING_SIZE];
    if (slot) ringOrdered.push(slot);
  }
  const payload = {
    label,
    timestamp: new Date().toISOString(),
    viewport: { ..._state.viewport },
    boundaryMode: _state.boundaryMode,
    backendName: _state.backendName,
    entityCount: _state.entities.length,
    tunables: _getTunables ? _getTunables() : null,
    ringSize: RING_SIZE,
    ringFrames: ringOrdered.length,
    ringBuffer: ringOrdered,
  };
  try {
    const res = await fetch('/dump', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      console.error('[state-dump] POST failed:', res.status, res.statusText);
      flashHint('Dump failed: ' + res.status, '#f44');
      return { ok: false, reason: 'http ' + res.status };
    }
    const result = await res.json();
    console.info('[state-dump] saved →', result.path, '(' + result.size + ' B,', ringOrdered.length, 'frames)');
    flashHint('Saved: ' + result.path, '#4f4');
    return result;
  } catch (e) {
    console.error('[state-dump] POST error:', e);
    flashHint('Dump error', '#f44');
    return { ok: false, reason: String(e) };
  }
}

export function installKeyHandler() {
  window.addEventListener('keydown', (ev) => {
    if (ev.key !== 'D' && ev.key !== 'd') return;
    const t = ev.target;
    if (t && t.matches && t.matches('input, textarea, select')) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    ev.preventDefault();
    dumpToServer('keyboard D');
  });
}

// ── Visual feedback ─────────────────────────────────────────────────
// Bottom-left toast that fades. Confirms the dump fired and shows the
// saved path so the dev doesn't have to open devtools to check.

let _toastEl = null;
function flashHint(text, color = '#4f4') {
  if (!_toastEl) {
    _toastEl = document.createElement('div');
    Object.assign(_toastEl.style, {
      position: 'fixed',
      bottom: '8px',
      left: '8px',
      padding: '6px 10px',
      background: 'rgba(0,0,0,0.85)',
      color: '#fff',
      border: '1px solid #444',
      borderRadius: '4px',
      font: '12px monospace',
      zIndex: '10000',
      pointerEvents: 'none',
      transition: 'opacity 0.4s',
      maxWidth: '60vw',
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
    });
    document.body.appendChild(_toastEl);
  }
  _toastEl.textContent = text;
  _toastEl.style.color = color;
  _toastEl.style.opacity = '1';
  clearTimeout(_toastEl._fadeT);
  _toastEl._fadeT = setTimeout(() => {
    if (_toastEl) _toastEl.style.opacity = '0';
  }, 1500);
}

// Persistent "press D" hint near the FPS widget. Drawn once at startup.
let _persistentHintEl = null;
export function installPersistentHint() {
  if (_persistentHintEl) return;
  _persistentHintEl = document.createElement('div');
  Object.assign(_persistentHintEl.style, {
    position: 'fixed',
    bottom: '8px',
    right: '96px',  // left of FPS widget
    padding: '4px 8px',
    background: 'rgba(0,0,0,0.7)',
    color: '#9c9',
    border: '1px solid #2a2',
    borderRadius: '3px',
    font: '11px monospace',
    zIndex: '9999',
    pointerEvents: 'none',
  });
  _persistentHintEl.textContent = '按 D 抓取状态';
  document.body.appendChild(_persistentHintEl);
}
