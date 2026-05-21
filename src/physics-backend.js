// physics-backend.js — CPU / WebGPU backend abstraction.
//
// Phase 1 of the WebGPU acceleration project. See docs/webgpu-blueprint.md §6.2
// for the interface contract and §5 for the dispatch model.
//
// Architecture decisions (architect transcript, this branch):
//   - Fork A: A2.α — pipelined per-substep dispatch with optional
//     `injectedAccels` 3rd arg to `stepPBD` (NO staleness; only fp32-vs-fp64
//     divergence vs the CPU oracle).
//   - Fork B: B1 — no shadow buffer in Phase 1. CPU is authoritative for
//     entity.{x,y,vx,vy}; GPU only reads positions+meta and writes accels.
//     On device.lost, swap to CPU; CPU continues with its existing state.
//
// Public factory `createBackend()` returns a wrapper that hides the swap:
// the caller (main.js) holds one reference and calls .step()/.prepareFrame()
// regardless of which backend is active. On `device.lost`, the wrapper
// transparently swaps the active backend to CPU; no caller-side coordination.

import { state } from './state.js';
import {
  prepareFrame as cpuPrepareFrame,
  stepPBD,
  updateAbsorptions,
  applyBoundary,
} from './physics.js';
import { detectBackend, loadKernel, isVerbose } from './gpu-init.js';
import { createGravityGPU } from './physics-gpu-gravity.js';

// ── CPU backend ────────────────────────────────────────────────────
// Thin pass-through to existing physics.js. Bit-identical to the main
// branch (F1 acceptance). No GPU resources, no async work.

function makeCpuBackend() {
  return {
    name: 'cpu',
    async init() { /* no-op */ },
    prepareFrame(entities) {
      cpuPrepareFrame(entities);
    },
    async step(entities, dt, viewport, boundaryMode) {
      stepPBD(entities, dt);   // no injectedAccels → CPU O(N²) / BH path runs
      updateAbsorptions(entities, dt);
      applyBoundary(entities, viewport, boundaryMode);
    },
    onEntityMetaMaybeChanged() { /* no-op */ },
    destroy() { /* no-op */ },
  };
}

// ── GPU backend ────────────────────────────────────────────────────
// Owns the K1 gravity_accel pipeline (physics-gpu-gravity.js). Per-substep:
//   1. await pendingReadback   → accels from the dispatch submitted at end of
//                                 prior substep (or priming on first call /
//                                 after a buffer reallocation).
//   2. stepPBD(entities, dt, accels) → CPU integrator/collisions/solver with
//                                       the injected accels.
//   3. updateAbsorptions + applyBoundary.
//   4. uploadPositions+Meta+Params; record + submit K1 dispatch into
//      `stagingBufs[dispatchIdx % 2]`; pendingReadback = mapAsync of that slot.
//
// Meta is re-uploaded every substep so that FLAG_ABSORBING flips from
// beginAbsorption (inside stepPBD → handleCollisions) and updateAbsorptions
// reach the GPU before the next K1 dispatch (architect risk #2). 80 KB at
// N=5000 — sub-ms writeBuffer.

async function makeGpuBackend(device, wgslSource, onLost) {
  const gravity = await createGravityGPU(device, wgslSource);
  const verbose = isVerbose();

  let dispatchIdx = 0;
  let pendingReadback = null;
  let teardown = false;

  device.lost.then(info => {
    if (teardown) return;
    teardown = true;
    pendingReadback = null;
    try { gravity.destroy(); } catch {}
    if (verbose) console.warn('[physics-backend] device lost:', info.reason, info.message);
    onLost(info);
  });

  function readStateParams() {
    const wrap = state.boundaryMode === 'wrap';
    const W = wrap ? state.viewport.width  : 0;
    const H = wrap ? state.viewport.height : 0;
    return { G: state.G, epsilon: state.epsilon, W, H };
  }

  async function submitDispatch(entities) {
    const N = entities.length;
    if (N === 0) {
      pendingReadback = null;
      return;
    }
    gravity.uploadPositions(entities, N);
    gravity.uploadMetaAll(entities, N);
    const { G, epsilon, W, H } = readStateParams();
    gravity.uploadParams(N, G, epsilon, W, H);

    const stagingIdx = dispatchIdx % 2;
    dispatchIdx++;
    const enc = device.createCommandEncoder({ label: 'K1 frame encoder' });
    gravity.recordDispatch(enc, N, stagingIdx);
    device.queue.submit([enc.finish()]);
    pendingReadback = gravity.readbackStaging(stagingIdx, N).catch(err => {
      if (teardown) return null;
      console.warn('[physics-backend] readback rejected:', err);
      return null;
    });
  }

  function ensureCapacity(entities) {
    const N = entities.length;
    const realloc = gravity.growIfNeeded(N);
    if (realloc) pendingReadback = null;
    return realloc;
  }

  return {
    name: 'webgpu',

    async init(entities) {
      ensureCapacity(entities);
      await submitDispatch(entities);
    },

    prepareFrame(_entities) {
      // No BH tree build (GPU runs exact O(N²)); no spatial hash either —
      // that's still per-substep inside handleCollisions on the CPU side.
    },

    async step(entities, dt, viewport, boundaryMode) {
      if (teardown) return;
      ensureCapacity(entities);
      if (pendingReadback === null) {
        await submitDispatch(entities);
      }
      const accelsBuf = pendingReadback;
      pendingReadback = null;
      const accels = await accelsBuf;
      if (teardown || accels == null) return;

      stepPBD(entities, dt, accels);
      updateAbsorptions(entities, dt);
      applyBoundary(entities, viewport, boundaryMode);

      ensureCapacity(entities);
      await submitDispatch(entities);
    },

    onEntityMetaMaybeChanged() { /* meta re-uploaded each submitDispatch */ },

    destroy() {
      teardown = true;
      pendingReadback = null;
      try { gravity.destroy(); } catch {}
    },
  };
}

// ── Wrapper ───────────────────────────────────────────────────────
// Single facade exposed to main.js. Hides the swap on device.lost.

export async function createBackend() {
  let active = makeCpuBackend();
  let initialized = false;

  function swapToCpu(reason) {
    active = makeCpuBackend();
    state.backendName = active.name;
    if (isVerbose()) console.info('[physics-backend] swapped to CPU:', reason);
  }

  async function tryInitGpu(entities) {
    const detection = await detectBackend();
    if (detection.backend !== 'webgpu') {
      if (isVerbose()) console.info('[physics-backend] using CPU:', detection.reason);
      return false;
    }
    let wgslSource;
    try {
      wgslSource = await loadKernel('./kernels/gravity_accel.wgsl');
    } catch (e) {
      if (isVerbose()) console.warn('[physics-backend] loadKernel failed:', e);
      detection.device.destroy();
      return false;
    }
    try {
      active = await makeGpuBackend(
        detection.device,
        wgslSource,
        info => swapToCpu(`device.lost: ${info.reason || 'unknown'}`),
      );
      await active.init(entities);
    } catch (e) {
      console.warn('[physics-backend] GPU backend init failed; falling back to CPU:', e);
      try { detection.device.destroy(); } catch {}
      active = makeCpuBackend();
      return false;
    }
    return true;
  }

  return {
    get name() { return active.name; },

    async init(entities) {
      if (initialized) return;
      initialized = true;
      const onGpu = await tryInitGpu(entities);
      state.backendName = active.name;
      if (isVerbose()) console.info('[physics-backend] active:', active.name, '(gpu init:', onGpu, ')');
    },

    prepareFrame(entities) {
      active.prepareFrame(entities);
    },

    async step(entities, dt, viewport, boundaryMode) {
      await active.step(entities, dt, viewport, boundaryMode);
    },

    onEntityMetaMaybeChanged() {
      active.onEntityMetaMaybeChanged();
    },

    destroy() {
      active.destroy();
    },
  };
}
