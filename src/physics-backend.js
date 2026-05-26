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
// Legacy K1-K8 WebGPU pipeline imports moved INSIDE makeGpuBackend()
// (2026-05-25 cleanup). When Rapier is the active backend (default),
// these modules never load — saves ~5-10 ms startup parse time and
// the 7 JS-file fetches the user's browser would otherwise do for
// dead code. The legacy-WebGPU path (?engine=webgpu, no Rapier) pays
// a one-time dynamic-import cost during makeGpuBackend.

// Diagnostic URL param parse: ?solver=simple switches stepPBD's contact
// solver to a 1-iteration direct-math version. See state.js for rationale.
(function parseSimpleSolverFlag() {
  try {
    const sp = new URLSearchParams(window.location.search);
    if (sp.get('solver') === 'simple') state.simpleSolver = true;
  } catch {}
})();

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
  // Lazy-load the K1-K8 modules. Parallel fetch via Promise.all so the
  // round-trip cost is one RTT not seven. Falls into the closures below
  // via destructuring (computeVelIter is in K5's module).
  const [
    { createGravityGPU },
    { createK2GPU },
    { createBroadphaseGPU },
    { createK4GPU },
    k5Module,
    { createK6GPU },
    { createK8GPU },
  ] = await Promise.all([
    import('./physics-gpu-gravity.js'),
    import('./physics-gpu-k2.js'),
    import('./physics-gpu-broadphase.js'),
    import('./physics-gpu-k4.js'),
    import('./physics-gpu-k5.js'),
    import('./physics-gpu-k6.js'),
    import('./physics-gpu-k8.js'),
  ]);
  const { createK5GPU, computeVelIter } = k5Module;

  const gravity = await createGravityGPU(device, wgslSources.k1);
  const k2 = await createK2GPU(device, wgslSources.k2, gravity);
  const broadphase = await createBroadphaseGPU(device, {
    k3:  wgslSources.k3,
    k3b: wgslSources.k3b,
    k3c: wgslSources.k3c,
  }, gravity, k2);
  const k4 = await createK4GPU(device, wgslSources.k4, gravity, broadphase, wgslSources.k4post, k2);
  const k8 = await createK8GPU(device, wgslSources.k8, gravity, k4);
  const k5 = await createK5GPU(device, { k5a: wgslSources.k5a, k5: wgslSources.k5 }, gravity, k4, k8);
  const k6 = await createK6GPU(device, wgslSources.k6, gravity, k2, k4);
  const verbose = isVerbose();
  let lastGridMeta = null;
  let prevContactCount = 32;   // blueprint §3.2 cold-start seed → 8 iters initially

  let dispatchIdx = 0;
  let pendingReadback = null;     // resolves to { positions, velocities, nanCount }
  let teardown = false;
  let _readbackWarnedOnce = false;   // throttle for the readback-rejection log

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
      // Log non-teardown readback failures, but only ONCE per session to
      // avoid console spam under transient WebGPU stalls (tab backgrounding,
      // GPU scheduler hiccups). Returning null causes step() to silently
      // skip the substep — the warning is critical for diagnosing freezes
      // but doesn't need repeating every frame. The verbose gate ALSO logs
      // every occurrence for the deep-diagnostic path.
      // Code-review HIGH 2026-05-25 (silent-failure-hunter + code-reviewer).
      if (!_readbackWarnedOnce) {
        _readbackWarnedOnce = true;
        console.warn('[physics-backend] readback rejected — substep skipped, entities will not move this frame (further occurrences suppressed; pass ?backend=verbose for full log):', err);
      } else if (verbose) {
        console.warn('[physics-backend] readback rejected (repeat):', err);
      }
      return null;
    });
  }

  function ensureCapacity(entities) {
    const N = entities.length;
    const gravityRealloc = gravity.growIfNeeded(N);
    if (gravityRealloc) { k2.onGravityRealloc(); broadphase.onGravityRealloc(); k4.onGravityRealloc(); k5.onGravityRealloc(); k6.onGravityRealloc(); k8.onGravityRealloc(); }
    const k2Realloc = k2.growIfNeeded(N);
    if (k2Realloc) { broadphase.onK2Realloc(); k4.onK2Realloc(); k6.onK2Realloc(); }
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
      // Diagnostic: in simple-solver mode, IGNORE K5/K6 readback and let CPU
      // stepPBD run its own contact solvers (which internally branch to the
      // simple version when state.simpleSolver is set). This isolates the
      // bug to the iterative/warm-start machinery vs upstream K1/K2/K4.
      const useGpuSolvers = !state.simpleSolver;
      const haveGpuVel = useGpuSolvers && readback.gpuVel && readback.gpuVel.length >= N * 2;
      const haveGpuPV  = useGpuSolvers && readback.gpuPV  && readback.gpuPV.length  >= N * 2;
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
      // Engine selection:
      //   default OR ?engine=rapier → Rapier2D backend (physics-rapier.js)
      //                   ↑ confirmed working in real Chrome 2026-05-24.
      //                   Wrap-bug fix (destroy+recreate on teleport)
      //                   built in.
      //   ?engine=planck → planck.js backend (physics-planck.js).
      //                   Kept as fallback / regression compare.
      //   ?engine=cpu    → legacy hand-ported Box2D-style PBD (physics.js)
      //   ?engine=webgpu → legacy K1-K8 GPU pipeline (physics-gpu-*.js)
      // Note: prior preview-MCP debugging session falsely concluded
      // Rapier was broken — that was a preview-only crash, not a
      // Rapier bug. Real Chrome runs Rapier at 240fps stable.
      const params = new URLSearchParams(window.location.search);
      const engineParam = params.get('engine');
      const wantRapierWorker = engineParam === 'rapier-worker';
      // When rapier-worker is requested but init fails, fall back to in-thread
      // Rapier instead of the CPU path. Otherwise wantRapier is set by the
      // explicit `rapier` flag or by the default (no engine param) path.
      let wantRapier = engineParam === 'rapier' ||
        (engineParam === null && params.get('backend') !== 'force-cpu');
      const wantPlanck = engineParam === 'planck';
      let usingExternalEngine = false;
      if (wantRapierWorker) {
        try {
          active = makeRapierWorkerBackend();
          await active.init(entities);
          usingExternalEngine = true;
        } catch (e) {
          console.warn('[physics-backend] rapier-worker init failed; falling back to in-thread Rapier:', e);
          wantRapier = true;   // promote to in-thread fallback
        }
      }
      if (!usingExternalEngine && wantRapier) {
        try {
          // Import the Rapier module via bare specifier (resolved by index.html
          // import map) and inject it into physics-rapier.js. The same module
          // can also be loaded by physics-rapier-worker.js with the full CDN
          // URL — both paths funnel through setRapier() before init().
          const RAPIER = (await import('@dimforge/rapier2d-compat')).default;
          const { makeRapierBackend, setRapier } = await import('./physics-rapier.js');
          setRapier(RAPIER);
          active = makeRapierBackend();
          await active.init(entities);
          usingExternalEngine = true;
        } catch (e) {
          console.warn('[physics-backend] rapier init failed; falling back:', e);
        }
      } else if (wantPlanck) {
        try {
          const { makePlanckBackend } = await import('./physics-planck.js');
          active = makePlanckBackend();
          await active.init(entities);
          usingExternalEngine = true;
        } catch (e) {
          console.warn('[physics-backend] planck init failed; falling back:', e);
        }
      }
      const onGpu = usingExternalEngine ? false : await tryInitGpu(entities);
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

// ── Rapier worker backend ──────────────────────────────────────────
// Proxies the in-thread Rapier API across a postMessage round-trip to a
// Web Worker. The worker owns the Rapier instance and runs step() off the
// main thread; main packs entity state into a transferable Float32Array
// each substep and waits for the worker's response.
//
// Activated via ?engine=rapier-worker. Falls back to in-thread Rapier on
// any init failure (no Worker support, module load error, etc.).
//
// v1 limitations (deferred to follow-up iterations):
//   - State-dump tracing happens worker-side only; main thread receives
//     no trace data. The 状态录制 UI button is a no-op when worker active.
//   - perf-monitor's per-phase ms breakdown collapses into a single
//     "workerRoundTrip" phase from main's perspective. Per-phase data
//     still exists worker-side but isn't surfaced yet.
//   - GPU gravity is bypassed inside the worker (navigator.gpu is rarely
//     available in workers; the worker uses CPU gravity exclusively).
//   - Spawn/delete/mutate detection from input.js is best-effort via a
//     prepareFrame-time diff; rapid spawn bursts during a single RAF
//     might land in the next-next substep.

function makeRapierWorkerBackend() {
  let worker = null;
  let readyPromise = null;
  let inFlight = null;        // Promise resolving on next stepDone / error
  let resolveInFlight = null;
  let rejectInFlight  = null;
  // v4 pipeline: holds the previous step's in-flight promise + the sentIds
  // associated with it, so the NEXT step() call can await + apply it.
  let prevInFlight = null;
  // `dead` short-circuits step() after destroy() or a fatal worker error,
  // so the substep loop fails fast instead of hanging on a postMessage to
  // a terminated worker. Code-review HIGH 2026-05-26.
  let dead = false;
  // Snapshot of last-known entity ids — used to diff against state.entities
  // each prepareFrame to compute the delta the worker needs.
  const knownIds = new Set();
  // Per-id snapshot of mutable meta fields — used to detect slider-driven
  // changes (mass / radius / type / pinned / charge) that need worker rebuild.
  const lastMeta = new Map();

  // Helpers ──────────────────────────────────────────────────────────
  function snapshotEntity(e) {
    return {
      id: e.id, type: e.type,
      x: e.x, y: e.y, vx: e.vx, vy: e.vy,
      mass: e.mass, radius: e.radius, charge: e.charge,
      pinned: !!e.pinned,
      color: e.color,
    };
  }
  function recordMeta(e) {
    lastMeta.set(e.id, {
      mass: e.mass, radius: e.radius, type: e.type,
      pinned: !!e.pinned, charge: e.charge,
    });
  }
  function metaChanged(e) {
    const last = lastMeta.get(e.id);
    if (!last) return false;
    return last.mass !== e.mass || last.radius !== e.radius ||
           last.type !== e.type || last.pinned !== !!e.pinned ||
           last.charge !== e.charge;
  }

  function computeDelta(entities) {
    const seen = new Set();
    const spawned = [];
    const mutated = [];
    for (const e of entities) {
      seen.add(e.id);
      if (!knownIds.has(e.id)) {
        spawned.push(snapshotEntity(e));
        knownIds.add(e.id);
        recordMeta(e);
      } else if (metaChanged(e)) {
        mutated.push({
          id: e.id,
          mass: e.mass, radius: e.radius, type: e.type,
          pinned: !!e.pinned, charge: e.charge,
        });
        recordMeta(e);
      }
    }
    const deleted = [];
    for (const id of knownIds) {
      if (!seen.has(id)) {
        deleted.push(id);
        knownIds.delete(id);
        lastMeta.delete(id);
      }
    }
    return (spawned.length || deleted.length || mutated.length)
      ? { spawned, deleted, mutated }
      : null;
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

  function applyStepDone(msg, entities, sentIds) {
    const { entityData, entityIds } = msg;
    // Phase 1: update positions of returned entities.
    const idMap = new Map();
    for (let i = 0; i < entities.length; i++) idMap.set(entities[i].id, entities[i]);
    for (let i = 0; i < entityIds.length; i++) {
      const e = idMap.get(entityIds[i]);
      if (!e || e.absorbing) continue;  // don't overwrite mid-absorbing animation
      e.x  = entityData[i * 4    ];
      e.y  = entityData[i * 4 + 1];
      e.vx = entityData[i * 4 + 2];
      e.vy = entityData[i * 4 + 3];
    }
    // Phase 2: reconcile absorptions / boundary-destroys. Anything that was
    // sent TO the worker but didn't come back was removed worker-side. We
    // splice it from main's state.entities to keep the renderer in sync.
    // Anything in main's entities NOT in sentIds is "newer than this step"
    // (e.g. spawned by input.js mid-flight) and is left alone.
    //
    // EXCEPTION: entities main has flagged `absorbing` (mid-animation) must
    // NOT be spliced even if the worker removed them. main's updateAbsorptions
    // owns the animation lifecycle and will splice them when the animation
    // completes. Splicing here would cause a body to vanish mid-animation
    // if it also triggered a worker-side absorption simultaneously.
    // Code-review HIGH 2026-05-26.
    //
    // RACE NOTE (v4 pipeline): the position write-back in Phase 1 above
    // could theoretically clobber a "user mid-drag" position. Today this
    // app does not support dragging EXISTING entities — state.drag is the
    // placement-preview/slingshot which spawns a new body only on
    // pointerup (see input.js). No race exists. If drag-existing is ever
    // added, gate Phase 1 write-back with a `!e._userDragging` check.
    if (sentIds && sentIds.length > 0) {
      const returnedSet = new Set();
      for (let i = 0; i < entityIds.length; i++) returnedSet.add(entityIds[i]);
      const sentSet = new Set();
      for (let i = 0; i < sentIds.length; i++) sentSet.add(sentIds[i]);
      for (let i = entities.length - 1; i >= 0; i--) {
        const e = entities[i];
        if (e.absorbing) continue;          // animation owns this entity
        if (sentSet.has(e.id) && !returnedSet.has(e.id)) {
          // Worker removed this entity (absorption / boundary destroy).
          knownIds.delete(e.id);
          lastMeta.delete(e.id);
          entities.splice(i, 1);
        }
      }
    }
  }

  function currentStateParams() {
    return {
      G: state.G,
      epsilon: state.epsilon,
      contactStiffness: state.contactStiffness,
      elasticRestitution: state.elasticRestitution,
      launchSpeedK: state.launchSpeedK,
      absorptionDuration: state.absorptionDuration,
      overlapEscalateThreshold: state.overlapEscalateThreshold,
      overlapCooldownFrames: state.overlapCooldownFrames,
      overlapBulletThreshold: state.overlapBulletThreshold,
      stateDumpEnabled: false,   // v1: never enable worker-side trace
      viewport: { width: state.viewport.width, height: state.viewport.height },
      boundaryMode: state.boundaryMode,
    };
  }

  return {
    name: 'rapier-worker',

    async init(entities) {
      // Construct the Worker. Using a relative URL with `type: 'module'`.
      // If the browser doesn't support module workers OR the URL fails
      // to resolve, this throws and the caller falls back to in-thread.
      worker = new Worker(new URL('./physics-rapier-worker.js', import.meta.url), { type: 'module' });

      // Single message handler — multiplexes ready / stepDone / error.
      worker.addEventListener('message', (ev) => {
        const m = ev.data;
        if (m.type === 'stepDone') {
          if (resolveInFlight) {
            const r = resolveInFlight; resolveInFlight = null; rejectInFlight = null;
            r(m);
          }
        } else if (m.type === 'error') {
          console.warn('[physics-backend] worker error:', m.message);
          if (rejectInFlight) {
            const r = rejectInFlight; resolveInFlight = null; rejectInFlight = null;
            r(new Error(m.message));
          }
        }
      });
      worker.addEventListener('error', (ev) => {
        console.warn('[physics-backend] worker fatal — marking backend dead:', ev.message || ev);
        dead = true;
        if (rejectInFlight) {
          const r = rejectInFlight; resolveInFlight = null; rejectInFlight = null;
          r(new Error(ev.message || 'worker error'));
        }
      });

      // Seed worker entity mirror.
      for (const e of entities) {
        knownIds.add(e.id);
        recordMeta(e);
      }

      // Send init + await ready.
      readyPromise = new Promise((resolve, reject) => {
        const handler = (ev) => {
          if (ev.data && ev.data.type === 'ready') {
            worker.removeEventListener('message', handler);
            resolve();
          } else if (ev.data && ev.data.type === 'error') {
            worker.removeEventListener('message', handler);
            reject(new Error(ev.data.message));
          }
        };
        worker.addEventListener('message', handler);
      });
      worker.postMessage({
        type: 'init',
        entities: entities.map(snapshotEntity),
      });
      await readyPromise;
      state.backendName = 'rapier-worker';
    },

    prepareFrame(entities) {
      // No-op on main thread — the worker runs prepareFrame internally
      // when it receives the step message. Delta computation happens in
      // step() since it needs to ride along on the step payload anyway.
    },

    async step(entities, dt, viewport, boundaryMode, isLastSubstep) {
      if (dead) throw new Error('[physics-backend] worker is dead; cannot step');
      // v4 PIPELINE: 1-deep. step() awaits the PREVIOUS in-flight result
      // (applying it to entities), then fires the CURRENT step async without
      // awaiting. Main never blocks on its own step — it blocks only on the
      // PREVIOUS step's worker compute, which has been overlapping with
      // main's render of the prior frame.
      //
      // Trade-off: rendered entity state is from physics frame N-1, not N.
      // At 60 FPS the lag is 16.7 ms — visually imperceptible for orbital
      // motion (velocity-extrapolation could close it further, deferred).
      //
      // On the very first step() call, prevInFlight is null and we skip
      // the await — the rendered frame uses pre-init entity state.
      if (prevInFlight) {
        const { promise: pendingPromise, sentIds: pendingSentIds } = prevInFlight;
        prevInFlight = null;
        const result = await pendingPromise;
        applyStepDone(result, entities, pendingSentIds);
      }

      if (dead) throw new Error('[physics-backend] worker died between pipeline drain and resend');

      const delta = computeDelta(entities);
      const { data, ids } = packEntityData(entities);
      const sentIds = Array.from(ids);

      const newPromise = new Promise((resolve, reject) => {
        resolveInFlight = resolve;
        rejectInFlight = reject;
      });

      worker.postMessage(
        {
          type: 'step',
          entityData: data,
          entityIds: ids,
          stateParams: currentStateParams(),
          delta,
          dt,
          viewport: { width: viewport.width, height: viewport.height },
          boundaryMode,
          isLastSubstep,
        },
        [data.buffer, ids.buffer]
      );

      prevInFlight = { promise: newPromise, sentIds };
      // step() returns resolved — main moves on to render. The result of
      // THIS step is awaited on the NEXT step() call.
    },

    onEntityMetaMaybeChanged() {
      // No-op — worker detects mutations each step via the delta diff.
    },

    destroy() {
      dead = true;
      // Reject any in-flight step() promise BEFORE terminating the worker.
      // Otherwise the await in step() hangs forever and the substep loop
      // freezes silently. Code-review HIGH 2026-05-26.
      if (rejectInFlight) {
        const r = rejectInFlight;
        resolveInFlight = null;
        rejectInFlight = null;
        try { r(new Error('worker destroyed before step completed')); } catch {}
      }
      prevInFlight = null;   // v4: drop the pending pipeline slot
      if (worker) {
        try { worker.postMessage({ type: 'destroy' }); } catch {}
        try { worker.terminate(); } catch {}
      }
      worker = null;
      knownIds.clear();
      lastMeta.clear();
    },
  };
}
