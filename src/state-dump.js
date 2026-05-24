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
let _getTunables = null;    // () → object snapshot of all state tunables

export function installRecorder(state, getTunables) {
  _state = state;
  _getTunables = getTunables;
}

// Substep trace — called from inside backend.step(), once per substep,
// AFTER pull-back to entities. The trace argument carries the substep's
// pre-state (before gravity), gravity vectors per entity, post-state
// (after world.step), contact details (normal/depth from the engine),
// and any wrap-rebuild events. We log raw inputs and outputs only —
// derived quantities (kinetic energy, momentum, Δv_solver, tangential
// decomposition, etc.) are computed offline from these by the analysis
// pass, per "VSCode-debug-trace" diagnostic philosophy.
//
// Schema per ring slot:
//   {
//     substep:       monotonic substep index
//     wallTimeMs:    DOMHighResTimestamp at record moment
//     dt:            integrator timestep this substep used (SIM_DT)
//     solverIters:   { velocity, pgs }    — backend's current iter counts
//     pre:           [ { id, x, y, vx, vy, sleeping } ]
//     gravity:       [ { id, ax, ay } ]   — accel from computeGravity
//     post:          [ { id, x, y, vx, vy, sleeping } ]
//     contacts:      [ { aId, bId, nx, ny, depth } ]
//     wrappedEntityIds: ids destroyed+recreated by wrap-boundary
//   }
export function recordSubstep(trace) {
  if (!_state) return;
  ring[writeIdx] = {
    substep: totalRecorded,
    wallTimeMs: +performance.now().toFixed(2),
    dt: trace.dt,
    solverIters: trace.solverIters || null,
    pre: trace.pre || [],
    gravity: trace.gravity || [],
    post: trace.post || [],
    contacts: trace.contacts || [],
    wrappedEntityIds: trace.wrappedEntityIds || [],
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
  // Static identification of every entity that touched the ring window
  // (id → mass / radius / charge / type / pinned). Pulled OUT of the
  // per-substep slots so they don't repeat 180 × N times.
  const idMeta = {};
  for (const slot of ringOrdered) {
    for (const e of (slot.pre || [])) {
      if (!idMeta[e.id]) {
        const ent = _state.entities.find(x => x.id === e.id);
        if (ent) idMeta[e.id] = {
          type: ent.type,
          mass: ent.mass,
          radius: +ent.radius.toFixed(3),
          charge: ent.charge,
          pinned: !!ent.pinned,
        };
      }
    }
  }
  const payload = {
    label,
    timestamp: new Date().toISOString(),
    schemaVersion: 2,
    viewport: { ..._state.viewport },
    boundaryMode: _state.boundaryMode,
    backendName: _state.backendName,
    entityCount: _state.entities.length,
    tunables: _getTunables ? _getTunables() : null,
    idMeta,
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
