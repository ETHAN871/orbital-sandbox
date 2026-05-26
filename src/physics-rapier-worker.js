// physics-rapier-worker.js — Web Worker entry for off-main-thread physics.
//
// Loaded by physics-backend.js via:
//   new Worker('/src/physics-rapier-worker.js', { type: 'module' })
//
// CONSTRAINT: workers do not inherit the document's import map. The bare
// specifier "@dimforge/rapier2d-compat" used in physics-rapier.js will not
// resolve in this context — instead we import Rapier via the full CDN URL
// (matches the entry in index.html's import map) and inject it into
// physics-rapier.js via setRapier() before invoking makeRapierBackend().
//
// MESSAGE PROTOCOL (v1 — minimum-viable round trip):
//   main → worker:
//     { type: 'init',  entities: EntitySnapshot[] }
//     { type: 'step',  entityData: Float32Array (transfer),
//                      entityIds:  Int32Array  (transfer),
//                      stateParams: { G, epsilon, contactStiffness, ... },
//                      delta: { spawned: [...], deleted: [...], mutated: [...] },
//                      dt, viewport, boundaryMode, isLastSubstep }
//     { type: 'destroy' }
//   worker → main:
//     { type: 'ready' }
//     { type: 'stepDone', entityData (transfer), entityIds (transfer) }
//     { type: 'error',    message }
//
// Stride-4 Float32Array layout: [x0, y0, vx0, vy0,  x1, y1, vx1, vy1,  ...]
// The parallel Int32Array holds entity IDs at the matching index.

import RAPIER from 'https://esm.sh/@dimforge/rapier2d-compat@0.19.3';
import { setRapier, makeRapierBackend } from './physics-rapier.js';
import { state } from './state.js';

let backend = null;
let workerEntities = [];   // mirror of main's state.entities (worker-local)
let idToEntity = new Map();

// Initialize Rapier once on first message (the WASM fetch + .init() is async).
let initPromise = null;
async function ensureBackend() {
  if (backend) return backend;
  if (!initPromise) {
    initPromise = (async () => {
      // RAPIER.init() loads + instantiates the WASM module. ~100-160 ms.
      await RAPIER.init();
      setRapier(RAPIER);
      backend = makeRapierBackend();
      // The backend mutates module-global `state` from state.js for tunables,
      // so we use that same module here. The worker's state object is a
      // SEPARATE INSTANCE from the main thread's (modules are reloaded per
      // worker), so we refresh tunables every step via stateParams.
      state.viewport = { width: 800, height: 600 };
      state.boundaryMode = 'destroy';
      return backend;
    })();
  }
  await initPromise;
  return backend;
}

function applyStateParams(params) {
  if (!params) return;
  for (const key of Object.keys(params)) {
    if (key === 'viewport') {
      state.viewport.width = params.viewport.width;
      state.viewport.height = params.viewport.height;
    } else {
      state[key] = params[key];
    }
  }
}

function packEntityData(entities) {
  const N = entities.length;
  const data = new Float32Array(N * 4);
  const ids = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    const e = entities[i];
    data[i * 4    ] = e.x;
    data[i * 4 + 1] = e.y;
    data[i * 4 + 2] = e.vx;
    data[i * 4 + 3] = e.vy;
    ids[i] = e.id;
  }
  return { data, ids };
}

function applyDelta(delta) {
  if (!delta) return;
  // Spawned: push to workerEntities + index. The backend's syncWorldToEntities
  // in prepareFrame will detect them as new entries and call createBodyForEntity.
  if (delta.spawned && delta.spawned.length) {
    for (const snap of delta.spawned) {
      const e = { trail: [], absorbing: false, ...snap };
      workerEntities.push(e);
      idToEntity.set(e.id, e);
    }
  }
  // Deleted: remove from workerEntities. syncWorldToEntities will see the
  // entity vanish and destroy the Rapier body via the orphan-cleanup pass.
  if (delta.deleted && delta.deleted.length) {
    const deletedSet = new Set(delta.deleted);
    for (let i = workerEntities.length - 1; i >= 0; i--) {
      if (deletedSet.has(workerEntities[i].id)) {
        idToEntity.delete(workerEntities[i].id);
        workerEntities.splice(i, 1);
      }
    }
  }
  // Mutated: patch fields in place. syncWorldToEntities watches for mass/
  // radius/type/pinned drift via its baked-marker check and rebuilds bodies.
  // We destructure `id` out of the patch before assigning so future code
  // can't accidentally rebind it (would corrupt idToEntity). Defensive —
  // code-review HIGH 2026-05-26.
  if (delta.mutated && delta.mutated.length) {
    for (const patch of delta.mutated) {
      const e = idToEntity.get(patch.id);
      if (!e) continue;
      const { id: _ignoredId, ...rest } = patch;
      Object.assign(e, rest);
    }
  }
}

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      await ensureBackend();
      workerEntities.length = 0;
      idToEntity.clear();
      for (const snap of (msg.entities || [])) {
        const e = { trail: [], absorbing: false, ...snap };
        workerEntities.push(e);
        idToEntity.set(e.id, e);
      }
      state.entities = workerEntities;
      await backend.init(workerEntities);
      self.postMessage({ type: 'ready' });
      return;
    }

    if (msg.type === 'step') {
      await ensureBackend();
      applyStateParams(msg.stateParams);
      applyDelta(msg.delta);

      backend.prepareFrame(workerEntities);
      await backend.step(workerEntities, msg.dt, state.viewport, state.boundaryMode, msg.isLastSubstep);

      const { data, ids } = packEntityData(workerEntities);
      self.postMessage(
        { type: 'stepDone', entityData: data, entityIds: ids },
        [data.buffer, ids.buffer]
      );
      return;
    }

    if (msg.type === 'destroy') {
      if (backend) {
        try { backend.destroy(); } catch (e) { /* swallow */ }
      }
      backend = null;
      initPromise = null;
      workerEntities = [];
      idToEntity.clear();
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
});
