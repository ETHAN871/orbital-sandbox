// physics-rapier.js — Default physics backend (Rapier2D via WASM).
//
// Replaces planck.js. Rapier is Rust → WASM, ~5-10× faster than planck
// at typical scene sizes, and exposes `lengthUnit` directly on the world
// so the unit-scale wrangling we did for planck (Settings.linearSlop=0.5,
// Settings.maxTranslation=50, etc.) becomes a single-knob configuration
// here.
//
// Closure-interface contract — identical to physics-planck.js:
//   init(entities)               build world + bodies
//   prepareFrame(entities)       per-frame sync (currently same as
//                                inside step; kept for parity)
//   step(entities, dt,           one substep
//        viewport,
//        boundaryMode,
//        isLastSubstep)
//   onEntityMetaMaybeChanged()   slider edits (no-op for now, matches
//                                planck behavior)
//   destroy()                    teardown
//
// What WE still own:
//   - Charge-asymmetric Plummer-softened gravity (computeGravity here)
//   - BH absorption animation lifecycle (entity.absorbing object)
//   - applyBoundary destroy/wrap handling
//   - Entity-array lifecycle
//
// Wrap-boundary handling — IMPORTANT:
//   Rapier (like Box2D / planck) uses a Euclidean broadphase (BVH-based)
//   that knows nothing about wrap. Pre-Rapier we hit a bug where a body
//   wrapping from x=W+ε to x=ε would instantly appear next to a cluster
//   at x=2, planck would register a fake high-velocity contact and apply
//   a catastrophic impulse → oscillation.
//
//   Rapier fix in step(): after applyBoundary teleports a body across the
//   wrap edge, we DESTROY the Rapier rigid body and RECREATE it at the
//   wrapped position. This atomically clears all contact pairs that
//   reference the pre-wrap collider, so the new body starts with zero
//   contact history. Re-creation cost ~µs per wrap event; only a tiny
//   fraction of bodies wrap per frame in typical scenes.
//
// Unit scale:
//   world.lengthUnit = 30 (1 meter ≈ 30 px). All Rapier tolerances
//   (allowedLinearError, contact_erp, sleep thresholds, etc.) are
//   normalized by lengthUnit internally, so this single setting brings
//   everything into pixel-sane defaults.
//
// Sleep:
//   Bodies have setCanSleep(true). Mirror of physics-planck.js logic:
//   the gravity-apply loop checks isSleeping() and computes
//   |accel|*dt — if that would push velocity above sleep tolerance,
//   the force is applied (Rapier's addForce wakes by default).
//   Otherwise the body stays asleep.

import RAPIER from '@dimforge/rapier2d-compat';
import { state } from './state.js';
import { updateAbsorptions, applyBoundary } from './physics.js';
import { detectBackend, loadKernel, isVerbose } from './gpu-init.js';
import { createGravityGPU } from './physics-gpu-gravity.js';
import { AdaptiveOverlapManager } from './physics-planck-overlap.js';
import { recordSubstep } from './state-dump.js';

// ── Unit-scale + solver knobs ─────────────────────────────────────────
// lengthUnit converts pixel coordinates to physics-engine "meters" so
// Rapier's internal tolerances (slop, sleep thresholds, etc.) land at
// values appropriate for our scene (radii 8-30 px, viewport ~2000 px,
// velocities ≤ 1000 px/s). 30 px ≈ 1 m is the standard game ratio.
const LENGTH_UNIT = 30;

// Velocity solver iterations. Default 4 too few for dense clusters; 8
// matches planck's heavy-mode start point.
const SOLVER_ITERATIONS_BASE = 8;
const SOLVER_ITERATIONS_MAX  = 24;

// NGS position solver iterations. Default is 1 (!) — wildly too few for
// our dense aggregate cases. 4 is a balanced baseline.
const PGS_ITERATIONS_BASE = 4;
const PGS_ITERATIONS_MAX  = 12;

// Sleep tolerance (Rapier exposes this via integration params; in px/s
// after lengthUnit conversion). Body sleeps when |v| stays below this
// for a few substeps. Mirrors planck.linearSleepTolerance=1.0 px/s.
const SLEEP_LINEAR_THRESHOLD_PX_PER_S = 1.0;

let RAPIER_READY = false;
let world = null;
let bodyById = new Map();    // entity.id (u32) → Rapier RigidBody
let colliderById = new Map(); // entity.id → Rapier Collider
let overlapMgr = null;

// Reusable Set for dedup'ing contact pairs in step()'s touching-count
// loop. Module-scope to avoid per-step allocation.
const _contactCounterSeen = new Set();

// ── Stage 2 GPU K1 (mirror of planck path) ──────────────────────────
const GPU_THRESHOLD = 200;
let gpuDevice = null;
let gpuGravityHandle = null;

async function tryInitGpuGravity() {
  try {
    const detection = await detectBackend();
    if (detection.backend !== 'webgpu') {
      if (isVerbose()) console.info('[physics-rapier] no WebGPU, gravity stays on CPU:', detection.reason);
      return false;
    }
    const wgslSource = await loadKernel('./kernels/gravity_accel.wgsl');
    gpuDevice = detection.device;
    gpuGravityHandle = await createGravityGPU(gpuDevice, wgslSource);
    if (isVerbose()) console.info('[physics-rapier] GPU K1 gravity ready');
    return true;
  } catch (e) {
    if (isVerbose()) console.warn('[physics-rapier] GPU init failed; staying on CPU gravity:', e);
    return false;
  }
}

async function computeGravity(entities) {
  const n = entities.length;
  const out = new Float32Array(n * 2);
  if (n < 2) return out;
  if (gpuGravityHandle && n >= GPU_THRESHOLD) {
    try {
      return await computeGravityGPU(entities, n);
    } catch (e) {
      if (isVerbose()) console.warn('[physics-rapier] GPU gravity dispatch failed; CPU fallback:', e);
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
  const enc = gpuDevice.createCommandEncoder({ label: 'rapier stage2 gravity' });
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
  const wantsCcd = initSpeed > state.overlapBulletThreshold;

  let desc;
  if (e.pinned) {
    desc = RAPIER.RigidBodyDesc.kinematicPositionBased();
  } else {
    desc = RAPIER.RigidBodyDesc.dynamic();
  }
  // Decomposed (not chained) — some rapier2d-compat 0.19.x builder methods
  // either don't return `this` or have surprising shapes. Calling each in
  // statement form makes chain-break failure modes impossible.
  desc.setTranslation(e.x, e.y);
  desc.setLinvel(e.vx || 0, e.vy || 0);
  desc.setCanSleep(true);
  desc.setCcdEnabled(wantsCcd);
  desc.lockRotations();

  const body = world.createRigidBody(desc);
  // userData is a post-construction plain JS property on RigidBody in
  // rapier2d-compat (NOT a builder method on the desc). Storing the
  // entity id here lets future code identify which body belongs to which
  // entity in callbacks (currently we use bodyById Map; userData is kept
  // for parity with planck path and future contact-event use).
  body.userData = e.id;

  const r = Math.max(e.radius, 0.01);
  const density = e.mass / (Math.PI * r * r);
  let cDesc = RAPIER.ColliderDesc.ball(r)
    .setDensity(density)
    .setFriction(0)
    .setRestitution(state.elasticRestitution);
  // Black holes are sensors — they detect overlap (so the absorption
  // lifecycle in detectAndStartBHAbsorptions can fire) but apply no
  // impulse to penetrating planets.
  if (e.type === 'black_hole') cDesc = cDesc.setSensor(true);

  const collider = world.createCollider(cDesc, body);

  // We store isCcd as a JS-side mirror to skip redundant Rapier toggles.
  body._isCcdEnabled = wantsCcd;

  bodyById.set(e.id, body);
  colliderById.set(e.id, collider);
  return body;
}

function destroyBody(id) {
  const b = bodyById.get(id);
  if (b) {
    try { world.removeRigidBody(b); } catch {}
    bodyById.delete(id);
  }
  colliderById.delete(id);
}

function syncWorldToEntities(entities) {
  const seenIds = new Set();
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) {
      // Same reasoning as planck: absorbing entities must be removed from
      // Rapier so they don't physically block the predator from reaching
      // them. Entity stays in state.entities until updateAbsorptions splices.
      if (bodyById.has(e.id)) destroyBody(e.id);
      continue;
    }
    seenIds.add(e.id);
    let b = bodyById.get(e.id);
    if (!b) {
      createBodyForEntity(e);
    } else {
      const wantKinematic = !!e.pinned;
      const isKinematic = b.bodyType() === RAPIER.RigidBodyType.KinematicPositionBased;
      if (wantKinematic !== isKinematic) {
        b.setBodyType(
          wantKinematic ? RAPIER.RigidBodyType.KinematicPositionBased
                        : RAPIER.RigidBodyType.Dynamic,
          true /* wake */,
        );
        if (wantKinematic) b.setLinvel({ x: 0, y: 0 }, true);
      }
    }
  }
  for (const id of [...bodyById.keys()]) {
    if (!seenIds.has(id)) destroyBody(id);
  }
}

function pushEntityStateToBodies(entities) {
  const bulletThr = state.overlapBulletThreshold;
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const b = bodyById.get(e.id);
    if (!b) continue;
    const pos = b.translation();
    if (pos.x !== e.x || pos.y !== e.y) b.setTranslation({ x: e.x, y: e.y }, true);
    const vel = b.linvel();
    if (vel.x !== e.vx || vel.y !== e.vy) b.setLinvel({ x: e.vx || 0, y: e.vy || 0 }, true);
    const speed = Math.hypot(e.vx || 0, e.vy || 0);
    const wantCcd = speed > bulletThr;
    if (b._isCcdEnabled !== wantCcd) {
      // enableCcd is a method on RigidBody in rapier2d-compat
      // (not setCcdEnabled — that's only on the desc).
      if (typeof b.enableCcd === 'function') b.enableCcd(wantCcd);
      b._isCcdEnabled = wantCcd;
    }
  }
}

function pullBodyStateToEntities(entities) {
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const b = bodyById.get(e.id);
    if (!b) continue;
    const pos = b.translation();
    const vel = b.linvel();
    e.x  = pos.x;
    e.y  = pos.y;
    e.vx = vel.x;
    e.vy = vel.y;
  }
}

// ── BH absorption (mirror of planck path; Rapier doesn't model it) ──

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

// ── Wrap-aware boundary handling ────────────────────────────────────
// Box2D-style broadphases (planck's b2DynamicTree, Rapier's BVH) are
// Euclidean — they have no concept of a wrap boundary. The naive
// applyBoundary teleport from physics.js sets entity.x atomically, then
// syncWorldToEntities pushes the new position into the engine body.
// The engine sees a body suddenly appear next to whatever's on the
// other side and registers it as a new contact pair with stale contact
// state. With planck this produced the dense-edge oscillation bug.
//
// Fix: after wrap-mode applyBoundary runs, identify each entity whose
// position was teleported this substep, then destroy + recreate its
// Rapier body. Re-creation has no contact history, so the engine
// starts the next step with a fresh broadphase entry — no phantom
// solver impulse from a stale pre-wrap contact.

function applyBoundaryAndRebuildOnWrap(entities, viewport, boundaryMode) {
  if (boundaryMode !== 'wrap') {
    applyBoundary(entities, viewport, boundaryMode);
    return [];
  }
  const W = viewport.width, H = viewport.height;
  if (W <= 0 || H <= 0) return [];
  const wrappedIds = [];
  for (const e of entities) {
    if (e.absorbing) continue;
    let wrapped = false;
    if (e.x < 0)      { e.x += W; wrapped = true; }
    else if (e.x > W) { e.x -= W; wrapped = true; }
    if (e.y < 0)      { e.y += H; wrapped = true; }
    else if (e.y > H) { e.y -= H; wrapped = true; }
    if (wrapped) wrappedIds.push(e.id);
  }
  // Atomic rebuild: destroy old body+collider, recreate at wrapped position.
  // This severs all contact pairs from the pre-wrap location so the next
  // step's broadphase + narrowphase compute fresh state for this body.
  for (const id of wrappedIds) {
    const e = entities.find(en => en.id === id);
    if (!e) continue;
    destroyBody(id);
    createBodyForEntity(e);
  }
  return wrappedIds;
}

// ── Backend factory ──────────────────────────────────────────────────

export function makeRapierBackend() {
  return {
    name: 'rapier',

    async init(entities) {
      if (!RAPIER_READY) {
        await RAPIER.init();
        RAPIER_READY = true;
      }
      world = new RAPIER.World({ x: 0, y: 0 });
      world.lengthUnit = LENGTH_UNIT;
      world.numSolverIterations = SOLVER_ITERATIONS_BASE;
      world.numInternalPgsIterations = PGS_ITERATIONS_BASE;
      world.timestep = 1 / 60;  // matches SIM_DT in state.js

      bodyById = new Map();
      colliderById = new Map();
      overlapMgr = new AdaptiveOverlapManager(state);

      await tryInitGpuGravity();

      // Build initial bodies from any entities that exist at startup
      // (drag-place can spawn before init completes).
      syncWorldToEntities(entities);
    },

    prepareFrame(entities) {
      syncWorldToEntities(entities);
    },

    async step(entities, dt, viewport, boundaryMode, _isLastSubstep) {
      syncWorldToEntities(entities);
      pushEntityStateToBodies(entities);

      // ── state-dump trace capture: pre-state ──────────────────────
      // Records every entity's exact position/velocity/sleep flag at
      // the start of this substep, BEFORE gravity is applied. Combined
      // with the gravity vectors below and the post-state after
      // world.step, an offline analyzer can compute Δv_solver per
      // entity and decompose it onto the contact normal to detect
      // tangential-impulse leaks.
      const pre = entities.map(e => {
        const b = bodyById.get(e.id);
        return {
          id: e.id,
          x:  +e.x.toFixed(3),
          y:  +e.y.toFixed(3),
          vx: +(e.vx || 0).toFixed(3),
          vy: +(e.vy || 0).toFixed(3),
          sleeping: b ? !!b.isSleeping() : null,
        };
      });

      const accels = await computeGravity(entities);

      // Per-entity gravity vector this substep (before addForce).
      const gravity = entities.map((e, i) => ({
        id: e.id,
        ax: +accels[i * 2].toFixed(6),
        ay: +accels[i * 2 + 1].toFixed(6),
      }));
      const sleepTol = SLEEP_LINEAR_THRESHOLD_PX_PER_S;
      for (let i = 0; i < entities.length; i++) {
        const e = entities[i];
        if (e.absorbing || e.pinned) continue;
        const b = bodyById.get(e.id);
        if (!b) continue;
        const ax = accels[i * 2];
        const ay = accels[i * 2 + 1];
        if (b.isSleeping()) {
          const dv = Math.hypot(ax, ay) * dt;
          if (dv <= sleepTol) continue;
          // Force is big enough to wake — Rapier wakes on addForce default,
          // but to be explicit:
          b.wakeUp();
        }
        b.addForce({ x: e.mass * ax, y: e.mass * ay }, true);
      }

      // Adaptive iteration scaling — match the planck pattern. Rapier
      // applies iterations via the world's numSolverIterations /
      // numInternalPgsIterations properties (read at world.step() call).
      const [velIter, posIter] = overlapMgr.decideIterations();
      // The AdaptiveOverlapManager constants are calibrated for planck's
      // 8/3 baseline; Rapier's units happen to be similar enough that
      // the same numbers work as upper bounds (clamped at MAX).
      world.numSolverIterations = Math.min(SOLVER_ITERATIONS_MAX, velIter);
      world.numInternalPgsIterations = Math.min(PGS_ITERATIONS_MAX, posIter);

      world.step();

      // Single contact-iteration pass: counts touching pairs for the
      // adaptive overlap manager AND extracts manifold details (normal,
      // depth) for the state-dump trace. Dedup'd by sorted handle pair.
      const contactsTrace = [];
      const seenPairs = _contactCounterSeen;
      seenPairs.clear();
      world.forEachCollider(c => {
        const aHandle = c.handle;
        const aBody = c.parent();
        const aId = aBody ? aBody.userData : null;
        world.contactPairsWith(c, (other) => {
          const bHandle = other.handle;
          const key = aHandle < bHandle
            ? aHandle * 0x100000000 + bHandle
            : bHandle * 0x100000000 + aHandle;
          if (seenPairs.has(key)) return;
          seenPairs.add(key);
          const bBody = other.parent();
          const bId = bBody ? bBody.userData : null;
          let nx = null, ny = null, depth = null;
          try {
            const manifold = world.contactPair(c, other);
            if (manifold && manifold.numContacts && manifold.numContacts() > 0) {
              const n1 = manifold.localNormal1 ? manifold.localNormal1() : null;
              if (n1) { nx = +n1.x.toFixed(4); ny = +n1.y.toFixed(4); }
              if (manifold.solverContactDepth) {
                depth = +manifold.solverContactDepth(0).toFixed(4);
              }
            }
          } catch {}
          contactsTrace.push({ aId, bId, nx, ny, depth });
        });
      });
      overlapMgr.recordPostStep(contactsTrace.length);

      pullBodyStateToEntities(entities);

      detectAndStartBHAbsorptions(entities);
      updateAbsorptions(entities, dt);
      const wrappedIds = applyBoundaryAndRebuildOnWrap(entities, viewport, boundaryMode);
      syncWorldToEntities(entities);

      // ── state-dump trace: post-state ─────────────────────────────
      // After pull-back + absorption + boundary + sync. This is the
      // "final" entity state for the substep. The recorder's offline
      // pass takes (pre.v, gravity.a, post.v) and computes
      //   Δv_solver = post.v - pre.v - gravity.a * dt
      // which isolates Rapier's solver contribution per substep.
      const post = entities.map(e => {
        const b = bodyById.get(e.id);
        return {
          id: e.id,
          x:  +e.x.toFixed(3),
          y:  +e.y.toFixed(3),
          vx: +(e.vx || 0).toFixed(3),
          vy: +(e.vy || 0).toFixed(3),
          sleeping: b ? !!b.isSleeping() : null,
        };
      });
      recordSubstep({
        dt,
        solverIters: {
          velocity: world.numSolverIterations,
          pgs: world.numInternalPgsIterations,
        },
        pre,
        gravity,
        post,
        contacts: contactsTrace,
        wrappedEntityIds: wrappedIds,
      });
    },

    onEntityMetaMaybeChanged() {
      // TODO support live radius/mass edits (matches planck stage 1).
    },

    // Engine-side snapshot for the state-dump module. Returns:
    //   bodyStates: { [entityId]: { sleeping, ccd } }
    //   contacts:   [{ a: entityIdA, b: entityIdB }, ...]
    //   solverIters: { velocity, pgs }
    // contactPairsWith fires once per (collider, pair) so we dedup via
    // a sorted pair-key. Entity ids come from body.userData (set in
    // createBodyForEntity).
    snapshot() {
      if (!world) return null;
      const bodyStates = {};
      for (const [id, b] of bodyById) {
        bodyStates[id] = {
          sleeping: b.isSleeping(),
          ccd: !!b._isCcdEnabled,
        };
      }
      const contacts = [];
      const seen = new Set();
      world.forEachCollider(c => {
        const aHandle = c.handle;
        const aBody = c.parent();
        const aId = aBody ? aBody.userData : null;
        world.contactPairsWith(c, (other) => {
          const bHandle = other.handle;
          const key = aHandle < bHandle
            ? aHandle * 0x100000000 + bHandle
            : bHandle * 0x100000000 + aHandle;
          if (seen.has(key)) return;
          seen.add(key);
          const bBody = other.parent();
          const bId = bBody ? bBody.userData : null;
          contacts.push({ a: aId, b: bId });
        });
      });
      return {
        bodyStates,
        contacts,
        solverIters: {
          velocity: world.numSolverIterations,
          pgs: world.numInternalPgsIterations,
        },
      };
    },

    destroy() {
      for (const id of [...bodyById.keys()]) destroyBody(id);
      bodyById.clear();
      colliderById.clear();
      if (overlapMgr) { overlapMgr.reset(); overlapMgr = null; }
      if (world) { try { world.free(); } catch {} world = null; }
      if (gpuGravityHandle) { try { gpuGravityHandle.destroy(); } catch {} gpuGravityHandle = null; }
      if (gpuDevice)        { try { gpuDevice.destroy();        } catch {} gpuDevice        = null; }
    },
  };
}

