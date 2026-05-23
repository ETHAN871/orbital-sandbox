// physics-planck.js — Stage 1 planck.js backend (no GPU).
//
// Replaces our hand-ported K1-K8 Box2D-style machinery with calls into
// planck (a Box2D 2.x JS port loaded via import map from esm.sh). planck
// owns: integration, collision detection, contact solver, CCD/TOI (for
// bodies flagged bullet=true), warm-start, broadphase.
//
// WE still own: charge-asymmetric Plummer-softened gravity (Newton's 3rd
// is intentionally broken — see CLAUDE.md), absorption animation lifecycle
// (entity.absorbing object), applyBoundary destroy/wrap mode, entity-array
// lifecycle (input.js push, splice).
//
// Stage 1 → Stage 2 contract: `computeGravity(entities)` is the SINGLE
// async hook that Stage 2 will replace with a GPU dispatch into
// physics-gpu-gravity.js (K1). All other code in this module stays
// untouched between stages.

import * as pl from 'planck';
import { state } from './state.js';
import { updateAbsorptions, applyBoundary } from './physics.js';
import { detectBackend, loadKernel, isVerbose } from './gpu-init.js';
import { createGravityGPU } from './physics-gpu-gravity.js';

let world = null;
let bodyById = new Map();   // entity.id (u32) → planck Body

// ── Stage 2: GPU K1 gravity acceleration ─────────────────────────────
// At backend init we attempt to bring up a WebGPU device + K1 pipeline.
// If successful, computeGravity() dispatches the existing K1 kernel
// (physics-gpu-gravity.js + gravity_accel.wgsl) for N ≥ GPU_THRESHOLD
// and reads back the Float32Array of accels. Below threshold the CPU
// O(N²) loop is faster because mapAsync round-trip dominates.
//
// Stage 1 → Stage 2 contract: this is the ONLY hook that changed. The
// rest of physics-planck.js — body sync, applyForceToCenter, world.step,
// absorption pipeline — is unchanged.
const GPU_THRESHOLD = 200;
let gpuDevice = null;
let gpuGravityHandle = null;

async function tryInitGpuGravity() {
  try {
    const detection = await detectBackend();
    if (detection.backend !== 'webgpu') {
      if (isVerbose()) console.info('[physics-planck] no WebGPU, gravity stays on CPU:', detection.reason);
      return false;
    }
    const wgslSource = await loadKernel('./kernels/gravity_accel.wgsl');
    gpuDevice = detection.device;
    gpuGravityHandle = await createGravityGPU(gpuDevice, wgslSource);
    if (isVerbose()) console.info('[physics-planck] GPU K1 gravity ready');
    return true;
  } catch (e) {
    if (isVerbose()) console.warn('[physics-planck] GPU init failed; staying on CPU gravity:', e);
    return false;
  }
}

async function computeGravity(entities) {
  const n = entities.length;
  const out = new Float32Array(n * 2);
  if (n < 2) return out;
  // Below threshold the CPU loop wins (mapAsync overhead dominates GPU
  // dispatch for tiny N). Above threshold the GPU O(N²) tile is faster.
  if (gpuGravityHandle && n >= GPU_THRESHOLD) {
    try {
      return await computeGravityGPU(entities, n);
    } catch (e) {
      if (isVerbose()) console.warn('[physics-planck] GPU gravity dispatch failed; CPU fallback:', e);
      // Fall through to CPU.
    }
  }
  return computeGravityCPU(entities, n, out);
}

async function computeGravityGPU(entities, n) {
  gpuGravityHandle.growIfNeeded(n);
  const wrap = state.boundaryMode === 'wrap';
  gpuGravityHandle.uploadPositions(entities, n);
  gpuGravityHandle.uploadMetaAll(entities, n);
  gpuGravityHandle.uploadParams(
    n, state.G, state.epsilon,
    wrap ? state.viewport.width  : 0,
    wrap ? state.viewport.height : 0,
  );
  const enc = gpuDevice.createCommandEncoder({ label: 'planck stage2 gravity' });
  gpuGravityHandle.recordDispatch(enc, n, 0);
  gpuDevice.queue.submit([enc.finish()]);
  return await gpuGravityHandle.readbackStaging(0, n);
}

function computeGravityCPU(entities, n, out) {
  const G = state.G;
  const eps = state.epsilon;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width  : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;
  for (let i = 0; i < n; i++) {
    const a = entities[i];
    if (a.absorbing) continue;
    for (let j = i + 1; j < n; j++) {
      const b = entities[j];
      if (b.absorbing) continue;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
      const r2Raw = dx * dx + dy * dy;
      const minR = Math.max(a.radius + b.radius, eps);
      const r2 = r2Raw + minR * minR;
      const r = Math.sqrt(r2);
      const nx = dx / r;
      const ny = dy / r;
      if (b.charge !== 0) {
        const mag = b.charge * G * b.mass / r2;
        out[i * 2]     += mag * nx;
        out[i * 2 + 1] += mag * ny;
      }
      if (a.charge !== 0) {
        const mag = a.charge * G * a.mass / r2;
        out[j * 2]     += mag * -nx;
        out[j * 2 + 1] += mag * -ny;
      }
    }
  }
  return out;
}

// ── Body lifecycle ───────────────────────────────────────────────────

function createBodyForEntity(e) {
  const body = world.createBody({
    type:           e.pinned ? 'kinematic' : 'dynamic',
    position:       pl.Vec2(e.x, e.y),
    linearVelocity: pl.Vec2(e.vx || 0, e.vy || 0),
    bullet:         true,    // CCD/TOI — defends against dense-cluster tunneling
    fixedRotation:  true,    // we don't model rotational dynamics
    allowSleep:     false,   // gravity is always non-zero, sleeping breaks our model
    userData:       e.id,
  });
  const r = Math.max(e.radius, 0.01);
  const density = e.mass / (Math.PI * r * r);
  body.createFixture({
    shape:       pl.Circle(r),
    density,
    friction:    0,
    restitution: state.elasticRestitution,
    // Black holes are sensors: detect overlap (so detectAndStartBHAbsorptions
    // can fire) but apply NO contact impulse. Otherwise planck's solver
    // would push planets back out of the BH before our absorption animation
    // can capture them — bug observed: B6 had only 4/8 planets consumed.
    isSensor: e.type === 'black_hole',
  });
  return body;
}

function destroyBody(body) {
  try { world.destroyBody(body); } catch {}
}

function syncWorldToEntities(entities) {
  const seenIds = new Set();
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    // Absorbing entities are mid-animation: prey radius shrinks visually,
    // updateAbsorptions interpolates x/y toward predator. They MUST NOT
    // remain as physics bodies — otherwise other bodies (including the BH
    // predator) would bounce off them at stale planck positions, blocking
    // the animation completion. Destroy on absorption start; entity stays
    // alive in the entities[] array until updateAbsorptions splices it.
    if (e.absorbing) {
      const b = bodyById.get(e.id);
      if (b) { destroyBody(b); bodyById.delete(e.id); }
      continue;
    }
    seenIds.add(e.id);
    let b = bodyById.get(e.id);
    if (!b) {
      b = createBodyForEntity(e);
      bodyById.set(e.id, b);
    } else {
      const wantKinematic = !!e.pinned;
      const isKinematic = b.getType() === 'kinematic';
      if (wantKinematic !== isKinematic) {
        b.setType(wantKinematic ? 'kinematic' : 'dynamic');
        if (wantKinematic) b.setLinearVelocity(pl.Vec2(0, 0));
      }
    }
  }
  for (const [id, b] of bodyById) {
    if (!seenIds.has(id)) { destroyBody(b); bodyById.delete(id); }
  }
}

function pushEntityStateToBodies(entities) {
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const b = bodyById.get(e.id);
    if (!b) continue;
    const pos = b.getPosition();
    if (pos.x !== e.x || pos.y !== e.y) b.setPosition(pl.Vec2(e.x, e.y));
    const vel = b.getLinearVelocity();
    if (vel.x !== e.vx || vel.y !== e.vy) b.setLinearVelocity(pl.Vec2(e.vx || 0, e.vy || 0));
  }
}

function pullBodyStateToEntities(entities) {
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const b = bodyById.get(e.id);
    if (!b) continue;
    const pos = b.getPosition();
    const vel = b.getLinearVelocity();
    e.x  = pos.x;
    e.y  = pos.y;
    e.vx = vel.x;
    e.vy = vel.y;
  }
}

// ── BH absorption (CPU-owned, planck doesn't model it) ──────────────

function detectAndStartBHAbsorptions(entities) {
  const n = entities.length;
  for (let i = 0; i < n; i++) {
    const a = entities[i];
    if (a.absorbing) continue;
    for (let j = i + 1; j < n; j++) {
      const b = entities[j];
      if (b.absorbing) continue;
      const aBH = a.type === 'black_hole';
      const bBH = b.type === 'black_hole';
      if (!aBH && !bBH) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const r2 = dx * dx + dy * dy;
      const rSum = a.radius + b.radius;
      if (r2 >= rSum * rSum) continue;
      let prey;
      if (aBH && !bBH) prey = b;
      else if (!aBH && bBH) prey = a;
      else if (a.mass < b.mass) prey = a;
      else if (b.mass < a.mass) prey = b;
      else continue;
      const predator = (prey === a) ? b : a;
      prey.absorbing = {
        t: 0,
        duration: state.absorptionDuration,
        predator,
        startX: prey.x,
        startY: prey.y,
        startRadius: prey.radius,
      };
    }
  }
}

// ── Backend factory ──────────────────────────────────────────────────

export function makePlanckBackend() {
  return {
    name: 'planck',

    async init(entities) {
      world = pl.World({
        gravity:    pl.Vec2(0, 0),  // we apply our own pairwise gravity via applyForceToCenter
        allowSleep: false,
      });
      bodyById = new Map();
      // Stage 2: try to bring up GPU K1 gravity. Non-fatal if it fails —
      // computeGravity() falls back to the CPU loop transparently.
      await tryInitGpuGravity();
      syncWorldToEntities(entities);
    },

    prepareFrame(entities) {
      syncWorldToEntities(entities);
    },

    async step(entities, dt, viewport, boundaryMode, _isLastSubstep) {
      syncWorldToEntities(entities);
      pushEntityStateToBodies(entities);

      const accels = await computeGravity(entities);
      for (let i = 0; i < entities.length; i++) {
        const e = entities[i];
        if (e.absorbing || e.pinned) continue;
        const b = bodyById.get(e.id);
        if (!b) continue;
        b.applyForceToCenter(pl.Vec2(e.mass * accels[i * 2], e.mass * accels[i * 2 + 1]));
      }

      world.step(dt);

      pullBodyStateToEntities(entities);

      detectAndStartBHAbsorptions(entities);
      updateAbsorptions(entities, dt);
      applyBoundary(entities, viewport, boundaryMode);
      syncWorldToEntities(entities);
    },

    onEntityMetaMaybeChanged() {
      // Stage 1: TODO support live radius/mass edits by re-creating
      // fixture for affected body. For now sliders only act on newly-
      // placed entities (matches existing behavior in CPU backend).
    },

    destroy() {
      for (const [, b] of bodyById) { try { world.destroyBody(b); } catch {} }
      bodyById.clear();
      world = null;
      if (gpuGravityHandle) { try { gpuGravityHandle.destroy(); } catch {} gpuGravityHandle = null; }
      if (gpuDevice)        { try { gpuDevice.destroy();        } catch {} gpuDevice        = null; }
    },
  };
}
