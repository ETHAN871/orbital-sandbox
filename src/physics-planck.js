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
import { AdaptiveOverlapManager } from './physics-planck-overlap.js';

// ── Unit-scale Settings overrides ─────────────────────────────────────
// planck's defaults are tuned for "1 unit = 1 meter, body radii ~1m,
// velocities ≤ 2 m/s". We feed pixel coords directly so every constant
// below was silently mis-scaled by ~1-100×, with consequences ranging
// from imperceptible (linearSlop) to catastrophic (maxTranslation silently
// clamped a 258 px/s body to 114 px/s within 1 sec; live-tested 2026-05-23).
//
// These values are derived from our typical scene geometry (radii 8-30 px,
// viewport ~1000-2000 px, intended velocity range up to ~1000 px/s) and
// are NOT user-tunable — changing them re-introduces the mechanism each
// was set to suppress. See the 2026-05-23 over-penetration triage doc.
//
//   linearSlop=1.0          — allowed residual overlap. Doubled from 0.5
//                             (2026-05-23 pm) so micro-penetration inside
//                             gravitationally-loaded aggregates is in the
//                             solver's dead zone — eliminates the perpetual
//                             jitter that prevents clusters from coming to
//                             rest. Approximately 6% of smallest body
//                             radius — visually undetectable.
//   maxLinearCorrection=8.0 — per-contact per-iter correction cap (5.7× headroom
//                             over measured worst-case 1.4 px/substep penetration
//                             in the m=1000,r=0.4 dense cluster bug case)
//   maxTranslation=50.0     — per-substep displacement cap (= 3000 px/s velocity
//                             ceiling, 3× over our stated 1000 px/s use range)
//   velocityThreshold=0.5   — below this px/s, restitution is suppressed (i.e.
//                             contacts treated as resting). 0.5 is comfortably
//                             above float noise but low enough that grazing
//                             contacts still get elastic treatment.
//   baumgarte=0.1           — Position-correction rate. Halved from default
//                             0.2 to soften the corrective impulse on every
//                             solver iteration, removing high-frequency
//                             ringing in resting contacts.
//   linearSleepTolerance=1.0 — Speed threshold below which a body is eligible
//                              for sleep. Default 0.01 is sub-pixel/sec —
//                              unreachable in our scene where solver residuals
//                              produce ~0.1-1 px/s noise. 1.0 px/s allows
//                              gravitationally-pinned aggregates to truly
//                              freeze; planck's per-body sleep flag is
//                              respected by world.step.
const PLANCK_SETTINGS_OVERRIDE = Object.freeze({
  linearSlop:           1.0,
  maxLinearCorrection:  8.0,
  maxTranslation:       50.0,
  velocityThreshold:    0.5,
  baumgarte:            0.1,
  linearSleepTolerance: 1.0,
});

// Restored at destroy() so a future planck re-init starts from a clean baseline
// (idempotency safety for tests / device-lost reload scenarios).
const PLANCK_SETTINGS_DEFAULT = Object.freeze({
  linearSlop:           0.005,
  maxLinearCorrection:  0.2,
  maxTranslation:       2.0,
  velocityThreshold:    1.0,
  baumgarte:            0.2,
  linearSleepTolerance: 0.01,
});

let world = null;
let bodyById = new Map();   // entity.id (u32) → planck Body
let overlapMgr = null;       // AdaptiveOverlapManager — built in init()

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
  const initSpeed = Math.hypot(e.vx || 0, e.vy || 0);
  const wantsBullet = initSpeed > state.overlapBulletThreshold;
  const body = world.createBody({
    type:           e.pinned ? 'kinematic' : 'dynamic',
    position:       pl.Vec2(e.x, e.y),
    linearVelocity: pl.Vec2(e.vx || 0, e.vy || 0),
    bullet:         wantsBullet,
    fixedRotation:  true,    // we don't model rotational dynamics
    allowSleep:     true,    // see World.allowSleep comment in init()
    userData:       { id: e.id, isBullet: wantsBullet },
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
  const bulletThr = state.overlapBulletThreshold;
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const b = bodyById.get(e.id);
    if (!b) continue;
    const pos = b.getPosition();
    if (pos.x !== e.x || pos.y !== e.y) b.setPosition(pl.Vec2(e.x, e.y));
    const vel = b.getLinearVelocity();
    if (vel.x !== e.vx || vel.y !== e.vy) b.setLinearVelocity(pl.Vec2(e.vx || 0, e.vy || 0));
    const ud = b.getUserData();
    const speed = Math.hypot(e.vx || 0, e.vy || 0);
    const wantBullet = speed > bulletThr;
    if (ud && ud.isBullet !== wantBullet) {
      b.setBullet(wantBullet);
      ud.isBullet = wantBullet;
    }
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
      pl.Settings.linearSlop           = PLANCK_SETTINGS_OVERRIDE.linearSlop;
      pl.Settings.maxLinearCorrection  = PLANCK_SETTINGS_OVERRIDE.maxLinearCorrection;
      pl.Settings.maxTranslation       = PLANCK_SETTINGS_OVERRIDE.maxTranslation;
      pl.Settings.velocityThreshold    = PLANCK_SETTINGS_OVERRIDE.velocityThreshold;
      pl.Settings.baumgarte            = PLANCK_SETTINGS_OVERRIDE.baumgarte;
      pl.Settings.linearSleepTolerance = PLANCK_SETTINGS_OVERRIDE.linearSleepTolerance;

      world = pl.World({
        gravity:    pl.Vec2(0, 0),  // we apply our own pairwise gravity via applyForceToCenter
        // 2026-05-23 pm: allowSleep flipped to true. Previously false on the
        // assumption "gravity is always non-zero", which is wrong for bodies
        // in a stable cluster where gravity is balanced by contact normal
        // force. Stable aggregates can now genuinely freeze (and skip solver
        // work). Gravity application loop in step() reads body.isAwake() to
        // avoid waking sleeping bodies every substep with redundant forces.
        allowSleep: true,
      });
      bodyById = new Map();
      overlapMgr = new AdaptiveOverlapManager(state);
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
      // 2026-05-23: User clarified — sleep is NOT a force-rejection mode.
      // A sleeping body still receives forces; it just stays asleep when
      // the applied force is too weak to push its velocity above the
      // sleep tolerance in one substep. This preserves the "no resistance"
      // contract: any meaningful external force wakes the body and
      // computes normally; only negligible forces are ignored (and they'd
      // be discarded by Box2D's applyForceToCenter on a sleeping body
      // anyway, so the gate just lets us skip the call cleanly).
      //
      // Threshold derivation: |Δv| over one substep = |accel| × dt. The
      // body sleeps when |v| < linearSleepTolerance. If the applied accel
      // would produce |Δv| above tolerance, that's a non-negligible force.
      const sleepTol = pl.Settings.linearSleepTolerance;
      for (let i = 0; i < entities.length; i++) {
        const e = entities[i];
        if (e.absorbing || e.pinned) continue;
        const b = bodyById.get(e.id);
        if (!b) continue;
        const ax = accels[i * 2];
        const ay = accels[i * 2 + 1];
        if (!b.isAwake()) {
          const dv = Math.hypot(ax, ay) * dt;
          if (dv <= sleepTol) continue;
        }
        b.applyForceToCenter(pl.Vec2(e.mass * ax, e.mass * ay));
      }

      const [velIter, posIter] = overlapMgr.decideIterations();
      world.step(dt, velIter, posIter);
      overlapMgr.recordPostStep(world);

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
      if (overlapMgr) { overlapMgr.reset(); overlapMgr = null; }
      if (gpuGravityHandle) { try { gpuGravityHandle.destroy(); } catch {} gpuGravityHandle = null; }
      if (gpuDevice)        { try { gpuDevice.destroy();        } catch {} gpuDevice        = null; }
      pl.Settings.linearSlop           = PLANCK_SETTINGS_DEFAULT.linearSlop;
      pl.Settings.maxLinearCorrection  = PLANCK_SETTINGS_DEFAULT.maxLinearCorrection;
      pl.Settings.maxTranslation       = PLANCK_SETTINGS_DEFAULT.maxTranslation;
      pl.Settings.velocityThreshold    = PLANCK_SETTINGS_DEFAULT.velocityThreshold;
      pl.Settings.baumgarte            = PLANCK_SETTINGS_DEFAULT.baumgarte;
      pl.Settings.linearSleepTolerance = PLANCK_SETTINGS_DEFAULT.linearSleepTolerance;
    },
  };
}
