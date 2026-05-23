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
import { createK4GPU } from './physics-gpu-k4.js';
import { createK5GPU, computeVelIter } from './physics-gpu-k5.js';
import { createK6GPU } from './physics-gpu-k6.js';
import { createK8GPU } from './physics-gpu-k8.js';

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
  const k4 = await createK4GPU(device, wgslSources.k4, gravity, broadphase, wgslSources.k4post);
  const k8 = await createK8GPU(device, wgslSources.k8, gravity, k4);
  const k5 = await createK5GPU(device, { k5a: wgslSources.k5a, k5: wgslSources.k5 }, gravity, k4, k8);
  const k6 = await createK6GPU(device, wgslSources.k6, gravity, k2, k4);
  const verbose = isVerbose();
  let lastGridMeta = null;
  let prevContactCount = 32;   // blueprint §3.2 cold-start seed → 8 iters initially

  let dispatchIdx = 0;
  let pendingReadback = null;     // resolves to { positions, velocities, nanCount }
  let teardown = false;

  device.lost.then(info => {
    if (teardown) return;
    teardown = true;
    pendingReadback = null;
    try { k8.destroy(); } catch {}
    try { k6.destroy(); } catch {}
    try { k5.destroy(); } catch {}
    try { k4.destroy(); } catch {}
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
    lastGridMeta = broadphase.uploadParams(entities, N, state.viewport.width, state.viewport.height);
    k4.uploadParams(N, lastGridMeta, dt, state.boundaryMode === 'wrap');
    // K5 iter count is 1-substep-stale per blueprint G4. prevContactCount
    // seeds to 32 (→8 iters) for the very first substep.
    const iterCount = computeVelIter(prevContactCount);
    k5.uploadParams(N, prevContactCount, dt, state.G, state.elasticRestitution);
    k6.uploadParams(N, prevContactCount, dt);
    k8.uploadParams(prevContactCount);

    const stagingIdx = dispatchIdx % 2;
    dispatchIdx++;
    const enc = device.createCommandEncoder({ label: 'K1-K8 frame encoder' });
    gravity.recordDispatch(enc, N, stagingIdx);
    k2.recordDispatch(enc, N, stagingIdx);
    broadphase.recordDispatch(enc, N);
    k4.recordDispatch(enc, N);
    k5.recordDispatch(enc, N, prevContactCount, iterCount);   // dark-launch
    k6.recordDispatch(enc, N, prevContactCount);              // dark-launch
    k8.recordDispatch(enc, prevContactCount);                 // dark-launch — fills pairImpulseTable for next substep
    device.queue.submit([enc.finish()]);
    pendingReadback = (async () => {
      // Phase 2g: also drain K5/K6 solver outputs + K4 contactCount.
      const [positions, velocities, nanCount, gpuVel, gpuPV, k4Read] = await Promise.all([
        k2.readbackPositions(stagingIdx, N),
        k2.readbackVelocities(stagingIdx, N),
        k2.readbackNanCounter(),
        k5.readbackVelocities(N),
        k6.readbackPseudoVels(N),
        k4.readback(),
      ]);
      // dispatchedN lets the apply path detect "entities.length changed
      // between submit and apply" (e.g. input.js push during the awaited
      // readback, or a prior-substep splice from applyBoundary destroy /
      // updateAbsorptions). Without it, applyK2OutputToEntities would walk
      // a NEW entities array with an OLD-sized Float32Array → out-of-range
      // reads return undefined → entity.x becomes NaN → renderer drops it.
      return { positions, velocities, nanCount, gpuVel, gpuPV, k4Read, dispatchedN: N };
    })().catch(err => {
      if (teardown) return null;
      if (verbose) console.warn('[physics-backend] readback rejected:', err);
      return null;
    });
  }

  function ensureCapacity(entities) {
    const N = entities.length;
    const gravityRealloc = gravity.growIfNeeded(N);
    if (gravityRealloc) { k2.onGravityRealloc(); broadphase.onGravityRealloc(); k4.onGravityRealloc(); k5.onGravityRealloc(); k6.onGravityRealloc(); k8.onGravityRealloc(); }
    const k2Realloc = k2.growIfNeeded(N);
    if (k2Realloc) { broadphase.onK2Realloc(); k6.onK2Realloc(); }
    const bpRealloc = broadphase.growIfNeeded(N);
    if (bpRealloc) k4.onBroadphaseRealloc();
    const k4Realloc = k4.growIfNeeded(N);
    if (k4Realloc) { k5.onK4Realloc(); k6.onK4Realloc(); k8.onK4Realloc(); }
    // K8 grows BEFORE K5 so K5's bind group references fresh K8 buffers.
    const k8Realloc = k8.growIfNeeded(N);
    const k5Realloc = k5.growIfNeeded(N);
    if (k8Realloc && !k5Realloc) k5.onK8Realloc();   // K5a binds K8 table; rebuild if K5 didn't already
    const k6Realloc = k6.growIfNeeded(N);
    if (gravityRealloc || k2Realloc || bpRealloc || k4Realloc || k5Realloc || k6Realloc || k8Realloc) pendingReadback = null;
    return gravityRealloc || k2Realloc || bpRealloc || k4Realloc || k5Realloc || k6Realloc || k8Realloc;
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
        try { k8.destroy(); } catch {}
        try { k6.destroy(); } catch {}
        try { k5.destroy(); } catch {}
        try { k4.destroy(); } catch {}
        try { broadphase.destroy(); } catch {}
        try { k2.destroy(); } catch {}
        try { gravity.destroy(); } catch {}
        onLost({ reason: 'nan-propagation', message: 'GPU produced NaN; restored CPU state' });
        return;
      }

      // 2026-05-21 regression fix: entity count can change between the GPU
      // submit and the awaited readback (input.js pointerup pushes a new
      // entity in the microtask between submit and the next step's await,
      // or a prior substep's applyBoundary destroy / updateAbsorptions
      // spliced). Applying an old-N readback against the new entities array
      // walks a Float32Array out of range → undefined → NaN positions →
      // renderer hides the new entity. Symptom user reported: "放置后被
      // 立刻清除掉". Fall back to a full CPU step this substep; next
      // submitDispatch picks up the new N. One frame of CPU physics is
      // a tolerable transient compared to losing the placed entity.
      if (readback.dispatchedN !== entities.length) {
        if (verbose) console.info('[physics-backend] entity count change ('
          + readback.dispatchedN + ' → ' + entities.length + '); CPU step this substep');
        cpuPrepareFrame(entities);
        stepPBD(entities, dt);   // no options → full CPU pipeline
        updateAbsorptions(entities, dt);
        applyBoundary(entities, viewport, boundaryMode);
        ensureCapacity(entities);
        await submitDispatch(entities, dt);
        return;
      }

      const N = entities.length;
      // Phase 2g flip: apply K2 positions (needed by CPU handleCollisions
      // broadphase + CPU absorption detection); override velocities with
      // K5's converged output; pre-load _pvx/_pvy with K6's converged
      // pseudo-velocities so stepPBD's integrate-only path consumes them.
      applyK2OutputToEntities(entities, N, readback.positions, readback.velocities);
      const haveGpuVel = readback.gpuVel && readback.gpuVel.length >= N * 2;
      const haveGpuPV  = readback.gpuPV  && readback.gpuPV.length  >= N * 2;
      if (haveGpuVel) {
        for (let i = 0; i < N; i++) {
          entities[i].vx = readback.gpuVel[i * 2];
          entities[i].vy = readback.gpuVel[i * 2 + 1];
        }
      }
      if (haveGpuPV) {
        for (let i = 0; i < N; i++) {
          entities[i]._pvx = readback.gpuPV[i * 2];
          entities[i]._pvy = readback.gpuPV[i * 2 + 1];
        }
      }
      // stepPBD runs:
      //   A (skipped via skipPseudoVelReset — GPU values preserved)
      //   B+C+D (skipped via skipKickPredict — K2 did them)
      //   E (handleCollisions runs — drives beginAbsorption on BH-pair
      //     contacts; CPU is the absorption authority per blueprint
      //     §11.9 deferral until K4 atomicOr lands)
      //   F+F' (skipped via skipVelocitySolver — K5 + K8 did them)
      //   G (skipped iter; _integratePseudoVelsOnly applies GPU _pv to x/y)
      //   I (pinned hard-reset runs)
      stepPBD(entities, dt, {
        skipKickPredict:        true,
        skipVelocitySolver:     haveGpuVel,
        skipPositionSolverIter: haveGpuPV,
        skipPseudoVelReset:     haveGpuPV,
      });
      updateAbsorptions(entities, dt);
      applyBoundary(entities, viewport, boundaryMode);

      // Update prevContactCount for next substep's K5 iter ramp.
      if (readback.k4Read && readback.k4Read.contactCount !== undefined) {
        prevContactCount = readback.k4Read.contactCount;
      }

      ensureCapacity(entities);
      await submitDispatch(entities, dt);
    },

    onEntityMetaMaybeChanged() { /* meta re-uploaded each submitDispatch */ },

    destroy() {
      teardown = true;
      pendingReadback = null;
      try { k8.destroy(); } catch {}
      try { k6.destroy(); } catch {}
      try { k5.destroy(); } catch {}
      try { k4.destroy(); } catch {}
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
      // Phase 2a-2f: + K8 rebuild_warm_start.
      const [k1, k2, k3, k3b, k3c, k4, k4post, k5a, k5, k6, k8] = await Promise.all([
        loadKernel('./kernels/gravity_accel.wgsl'),
        loadKernel('./kernels/kick_predict.wgsl'),
        loadKernel('./kernels/broadphase_count.wgsl'),
        loadKernel('./kernels/broadphase_prefix_sum.wgsl'),
        loadKernel('./kernels/broadphase_scatter.wgsl'),
        loadKernel('./kernels/contact_detect.wgsl'),
        loadKernel('./kernels/compute_dispatch_args.wgsl'),
        loadKernel('./kernels/warm_start_calibrate.wgsl'),
        loadKernel('./kernels/velocity_solver.wgsl'),
        loadKernel('./kernels/position_solver.wgsl'),
        loadKernel('./kernels/rebuild_warm_start.wgsl'),
      ]);
      wgslSources = { k1, k2, k3, k3b, k3c, k4, k4post, k5a, k5, k6, k8 };
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
