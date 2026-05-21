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
import { createK2GPU } from './physics-gpu-k2.js';
import { createBroadphaseGPU } from './physics-gpu-broadphase.js';

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
    async step(entities, dt, viewport, boundaryMode, _isLastSubstep) {
      stepPBD(entities, dt);   // no options → CPU O(N²) / BH path runs unchanged
      updateAbsorptions(entities, dt);
      applyBoundary(entities, viewport, boundaryMode);
    },
    onEntityMetaMaybeChanged() { /* no-op */ },
    destroy() { /* no-op */ },
  };
}

// ── GPU backend ────────────────────────────────────────────────────
// Phase 2a: K1 (gravity_accel) + K2 (kick_predict) chained in one encoder.
// K1 writes accels; K2 reads them + writes outPositions + velocities (in
// place). CPU reads back positions / velocities / nanCounter each substep,
// applies them to entity objects, then runs collision + solver + boundary
// on CPU via stepPBD(..., { skipKickPredict: true }). Steps A and I still
// run CPU-side (per blueprint §12 G12 + physics.js header comment).
//
// Buffer ownership: gravity wrapper owns positions / velocities / metas /
// accels (now with COPY_SRC for K2 readback). K2 wrapper owns outPositions /
// pseudoVels / nanCheckBuf and binds the gravity buffers by reference via
// getters (architect M1 resolution). On gravity realloc, K2's bind group
// must be rebuilt via k2.onGravityRealloc().
//
// nanCheckBuf reset is per-frame (in prepareFrame), NOT per-substep, so a
// NaN trip in any substep within the frame stays visible to the CPU drain
// at the next substep boundary — triggers swapToCpu('nan-propagation').

async function makeGpuBackend(device, wgslSources, onLost) {
  const gravity = await createGravityGPU(device, wgslSources.k1);
  const k2 = await createK2GPU(device, wgslSources.k2, gravity);
  const broadphase = await createBroadphaseGPU(device, {
    k3:  wgslSources.k3,
    k3b: wgslSources.k3b,
    k3c: wgslSources.k3c,
  }, gravity, k2);
  const verbose = isVerbose();

  let dispatchIdx = 0;
  let pendingReadback = null;     // resolves to { positions, velocities, nanCount }
  let teardown = false;

  device.lost.then(info => {
    if (teardown) return;
    teardown = true;
    pendingReadback = null;
    try { broadphase.destroy(); } catch {}
    try { k2.destroy(); } catch {}
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

  async function submitDispatch(entities, dt) {
    const N = entities.length;
    if (N === 0) {
      pendingReadback = null;
      return;
    }
    gravity.uploadPositions(entities, N);
    gravity.uploadVelocities(entities, N);    // K2 reads velocities (kick)
    gravity.uploadMetaAll(entities, N);
    const { G, epsilon, W, H } = readStateParams();
    gravity.uploadParams(N, G, epsilon, W, H);
    k2.uploadParams(N, dt);
    // Broadphase grid params recompute per substep — depends on maxR + viewport.
    // viewport state is global; uploadParams uses state.viewport directly.
    broadphase.uploadParams(entities, N, state.viewport.width, state.viewport.height);

    const stagingIdx = dispatchIdx % 2;
    dispatchIdx++;
    const enc = device.createCommandEncoder({ label: 'K1+K2+K3 frame encoder' });
    gravity.recordDispatch(enc, N, stagingIdx);     // K1 → accels
    k2.recordDispatch(enc, N, stagingIdx);          // K2 → outPositions / velocities / nanCount
    broadphase.recordDispatch(enc, N);              // K3 / K3b / K3c → cellCounts / cellOffsets / cellContents (dark-launch)
    device.queue.submit([enc.finish()]);
    pendingReadback = (async () => {
      const [positions, velocities, nanCount] = await Promise.all([
        k2.readbackPositions(stagingIdx, N),
        k2.readbackVelocities(stagingIdx, N),
        k2.readbackNanCounter(),
      ]);
      return { positions, velocities, nanCount };
    })().catch(err => {
      if (teardown) return null;
      if (verbose) console.warn('[physics-backend] readback rejected:', err);
      return null;
    });
  }

  function ensureCapacity(entities) {
    const N = entities.length;
    const gravityRealloc = gravity.growIfNeeded(N);
    if (gravityRealloc) { k2.onGravityRealloc(); broadphase.onGravityRealloc(); }
    const k2Realloc = k2.growIfNeeded(N);
    if (k2Realloc) broadphase.onK2Realloc();
    const bpRealloc = broadphase.growIfNeeded(N);
    if (gravityRealloc || k2Realloc || bpRealloc) pendingReadback = null;
    return gravityRealloc || k2Realloc || bpRealloc;
  }

  function applyK2OutputToEntities(entities, N, positions, velocities) {
    for (let i = 0; i < N; i++) {
      const e = entities[i];
      e.x  = positions[i * 2];
      e.y  = positions[i * 2 + 1];
      e.vx = velocities[i * 2];
      e.vy = velocities[i * 2 + 1];
    }
  }

  return {
    name: 'webgpu',

    async init(entities) {
      ensureCapacity(entities);
      k2.resetNanCounter();
      await submitDispatch(entities, 1 / 60);
    },

    prepareFrame(_entities) {
      // No BH tree build (GPU runs exact O(N²)); no spatial hash either —
      // that's still per-substep inside handleCollisions on the CPU side.
      // nanCheckBuf reset is per-frame (not per-substep) so a NaN trip in
      // any substep stays visible to this frame's CPU drain — see blueprint
      // §3 NaN guards paragraph.
      k2.resetNanCounter();
    },

    async step(entities, dt, viewport, boundaryMode, _isLastSubstep) {
      // _isLastSubstep is the G12 5th-arg seam for sub-phase 2e+ shadow
      // buffer activation. In 2a, K7 isn't yet wired into the substep loop
      // and the shadow buffer isn't activated; this arg is currently a no-op.
      if (teardown) return;
      ensureCapacity(entities);
      if (pendingReadback === null) {
        await submitDispatch(entities, dt);
      }
      const readbackPromise = pendingReadback;
      pendingReadback = null;
      const readback = await readbackPromise;
      if (teardown || readback == null) return;

      // NaN propagation check (blueprint §3 / §6.3). If K2 (or K7 once
      // wired) tripped the guard, GPU positions are corrupted. Fall back
      // to CPU with the entity state we had BEFORE applying the bad
      // readback — i.e., do not call applyK2OutputToEntities.
      if (readback.nanCount > 0) {
        if (verbose) console.warn('[physics-backend] NaN detected on GPU; swapping to CPU');
        teardown = true;
        try { broadphase.destroy(); } catch {}
        try { k2.destroy(); } catch {}
        try { gravity.destroy(); } catch {}
        onLost({ reason: 'nan-propagation', message: 'GPU produced NaN; restored CPU state' });
        return;
      }

      const N = entities.length;
      applyK2OutputToEntities(entities, N, readback.positions, readback.velocities);

      // K2 already did steps C (kick) and D (predict). CPU still runs:
      //   A (pseudoVel reset of entity._pvx — separate from GPU pseudoVels)
      //   E (handleCollisions), F (vel solver), F' (warm-start rebuild),
      //   G (position solver), I (pinned hard-reset of entity.vx/vy).
      stepPBD(entities, dt, { skipKickPredict: true });
      updateAbsorptions(entities, dt);
      applyBoundary(entities, viewport, boundaryMode);

      ensureCapacity(entities);
      await submitDispatch(entities, dt);
    },

    onEntityMetaMaybeChanged() { /* meta re-uploaded each submitDispatch */ },

    destroy() {
      teardown = true;
      pendingReadback = null;
      try { broadphase.destroy(); } catch {}
      try { k2.destroy(); } catch {}
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
    let wgslSources;
    try {
      // Phase 2a-2b: K1 (gravity) + K2 (kick_predict) + K3/K3b/K3c (broadphase).
      const [k1, k2, k3, k3b, k3c] = await Promise.all([
        loadKernel('./kernels/gravity_accel.wgsl'),
        loadKernel('./kernels/kick_predict.wgsl'),
        loadKernel('./kernels/broadphase_count.wgsl'),
        loadKernel('./kernels/broadphase_prefix_sum.wgsl'),
        loadKernel('./kernels/broadphase_scatter.wgsl'),
      ]);
      wgslSources = { k1, k2, k3, k3b, k3c };
    } catch (e) {
      if (isVerbose()) console.warn('[physics-backend] loadKernel failed:', e);
      detection.device.destroy();
      return false;
    }
    try {
      active = await makeGpuBackend(
        detection.device,
        wgslSources,
        info => swapToCpu(`${info.reason || 'unknown'}: ${info.message || ''}`),
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

    async step(entities, dt, viewport, boundaryMode, isLastSubstep) {
      await active.step(entities, dt, viewport, boundaryMode, isLastSubstep);
    },

    onEntityMetaMaybeChanged() {
      active.onEntityMetaMaybeChanged();
    },

    destroy() {
      active.destroy();
    },
  };
}
