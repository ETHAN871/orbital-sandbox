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

// Ring length = 360 substeps ≈ 6 s at 60 Hz substep cadence. Doubled from
// 180 so contact moments that fall just outside a 3-second window still
// get captured — the bug-of-interest (tangential v accumulation during a
// brief contact) was being missed when the dump fired a beat late.
const RING_SIZE = 360;
const ring = new Array(RING_SIZE);
let writeIdx = 0;
let totalRecorded = 0;

let _state = null;
let _getTunables = null;    // () → object snapshot of all state tunables

// Per-id metadata cache. recordSubstep populates this while the entity
// is still alive in state.entities. dumpToServer reads from here when
// state.entities.find() returns null (entity already destroyed by the
// time the user pressed D), so dumps that span "place → contact →
// clear → place again" still describe every id in the ring window.
//
// Cache is not trimmed: at ~5 fields per id and monotonically increasing
// ids, growth is ≈ 50 B per entity ever placed — negligible.
const _idMetaCache = new Map();

export function installRecorder(state, getTunables) {
  _state = state;
  _getTunables = getTunables;
}

// Substep trace — called from inside backend.step(), once per substep,
// AFTER pull-back to entities. The trace argument carries the substep's
// pre-state (before gravity), gravity vectors per entity, post-state
// (after world.step), contact details (normal/depth from the engine),
// in-step contact events drained from the engine's EventQueue, and any
// wrap-rebuild events. We log raw inputs and outputs only — derived
// quantities (kinetic energy, momentum, Δv_solver, tangential
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
//     gravity:       [ { id, ax, ay, forceApplied } ]
//                       forceApplied=true  → addForce was called this substep
//                       forceApplied=false → skipped for one of these reasons:
//                                            (absorbing | pinned | no-body |
//                                             sleeping+tiny-impulse).
//                                            Note that ax/ay are still the
//                                            computed gravity vector — the
//                                            value is just not delivered to
//                                            the body.
//     post:          [ { id, x, y, vx, vy, sleeping } ]
//     contacts:      [ { aId, bId, nx, ny, depth } ]  post-step query
//     contactEventsTruncated: bool — true if more than 512 events fired
//                       in this substep and the tail was dropped. The
//                       cap exists to prevent unbounded GC pressure
//                       from a pathological dense cluster. Offline
//                       analysis should flag these substeps explicitly.
//     contactEvents: [ { aId, bId, started, aVx, aVy, bVx, bVy,
//                         nx, ny, depth } ]
//                       drained from Rapier EventQueue — fires for each
//                       contact start/end *inside* world.step, so brief
//                       impacts that resolve mid-step are still captured
//                       even when the post-step query sees nothing.
//                       aVx/aVy/bVx/bVy are read at drain time, AFTER
//                       world.step has fully returned — every event in
//                       this slot's contactEvents shares the same final
//                       post-step velocity for its body. They are NOT
//                       per-event velocities at the moment the event
//                       fired. Cross-reference with adjacent slots'
//                       pre[]/post[] entries by id for finer timing.
//                       nx/ny/depth are the contact normal (aId → bId
//                       orientation, world frame because balls have
//                       lockRotations) and Rapier's solver-target depth
//                       at drain time. NULL on end events (manifold
//                       has already dissolved by drain time) — use the
//                       contacts[] entry from one substep earlier for
//                       the normal that was active just before the end.
//                       In wrap mode, IDs can be NEGATIVE: aId = -(N+1)
//                       means the contact involved a ghost body that
//                       mirrors real entity N (a copy placed at the
//                       wrap-offset position so cross-edge contacts are
//                       detected). To recover the real entity, take
//                       (-aId - 1). Both positive and negative IDs
//                       refer to the same underlying matter.
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
    contactEvents: trace.contactEvents || [],
    contactEventsTruncated: !!trace.contactEventsTruncated,
    wrappedEntityIds: trace.wrappedEntityIds || [],
  };
  writeIdx = (writeIdx + 1) % RING_SIZE;
  totalRecorded++;

  // Cache metadata for every id we saw this substep while the entity
  // is still alive — dumpToServer can then describe destroyed ids too.
  // We REFRESH the entry every substep (no early-exit) so mutable
  // fields like `pinned` (toggleable from the UI) reflect the latest
  // live state. The cost is one Map.set per entity per substep, which
  // is negligible.
  for (const e of (trace.pre || [])) {
    const ent = _state.entities.find(x => x.id === e.id);
    if (!ent) continue;
    _idMetaCache.set(e.id, {
      type:   ent.type,
      mass:   ent.mass,
      radius: +ent.radius.toFixed(3),
      charge: ent.charge,
      pinned: !!ent.pinned,
    });
  }
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
  // per-substep slots so they don't repeat RING_SIZE × N times. Sources
  // in priority order:
  //   1. live entity in _state.entities — always preferred when available.
  //   2. _idMetaCache fallback for entities destroyed before dump time.
  //      The cache is refreshed every substep while the entity is live
  //      (see recordSubstep), so the cached value is the last-known
  //      live state — accurate even for fields the UI can mutate
  //      mid-life (currently just `pinned`).
  const idMeta = {};
  for (const slot of ringOrdered) {
    for (const e of (slot.pre || [])) {
      if (idMeta[e.id]) continue;
      const ent = _state.entities.find(x => x.id === e.id);
      if (ent) {
        idMeta[e.id] = {
          type: ent.type,
          mass: ent.mass,
          radius: +ent.radius.toFixed(3),
          charge: ent.charge,
          pinned: !!ent.pinned,
        };
      } else if (_idMetaCache.has(e.id)) {
        idMeta[e.id] = _idMetaCache.get(e.id);
      }
    }
  }
  const payload = {
    label,
    timestamp: new Date().toISOString(),
    // v3 (2026-05-24): slot.gravity[i] grew a `forceApplied` flag, slot
    // gained `contactEvents[]` from Rapier's EventQueue, ring length
    // doubled to 360 substeps. Reader code should fall back to absent
    // fields when reading older v2 dumps.
    schemaVersion: 3,
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
