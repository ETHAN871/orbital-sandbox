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

// TGS-Soft contact-spring natural frequency. Rapier 0.19 default
// ω₀ ≈ 377 rad/s = 2π × 60 Hz makes position-correction near-
// instantaneous → "spawn explosion" when a body is created inside
// another (worst-case 2-radius overlap → solver injects
// v_corrective ≈ ω₀ × penetration / (1 + ω₀·dt) ≈ 3000 px/s for
// r=30, ω₀=377). Setting ω₀ = 6 reduces v_corrective to ~330 px/s
// for the same overlap — non-explosive; resolves in ~6 substeps
// when combined with the conditional spawn-damping burst below.
// Trade-off: pile-up settling becomes slightly springy (a few ms
// extra to fully settle after a disturbance). Runtime fast-impact
// bounces remain sharp because TGS-Soft applies restitution from
// incoming relative velocity, not from spring deformation.
//
// NB: rapier2d-compat 0.19 exposes only a setter for this field
// (no getter). `contactDampingRatio` and `maxCorrectiveVelocity`
// are NOT in the 0.19 JS bindings, so this single knob is our
// only world-level lever for softening contacts.
const CONTACT_NATURAL_FREQUENCY = 6.0;

// Normalized slop threshold. Default ~0.001 (= 0.03 px at
// lengthUnit=30). Widening to 0.005 (~0.15 px) cuts final-
// convergence oscillation on near-resolved contacts without
// visible geometry impact.
const NORMALIZED_ALLOWED_LINEAR_ERROR = 0.005;

// Conditional spawn-damping: applied to a newly-created body ONLY
// if Rapier's intersectionsWithShape query reports it overlaps an
// existing collider at the spawn moment. Goal per user spec: bodies
// arrive at the no-overlap boundary with relative velocity ≈ 0
// (not "fly out at residual speed").
//
// Approach: critical damping during the burst + active termination
// the moment overlap clears.
//   1. d = 2·ω₀ ≈ 12 (critical damping at ω₀ = 6). Critically
//      damped springs decelerate monotonically toward equilibrium
//      without overshoot. Rapier 0.12+ applies linearDamping as
//      `v *= max(0, 1 − d·dt)` per substep (linear, clamped). At
//      d=12, dt=1/60: factor = 0.80/sub — velocity decays smoothly.
//   2. Each substep we re-query intersectionsWithShape at the
//      body's CURRENT Rapier position. The instant the query
//      reports no-overlap, we call `setLinvel(0, 0)` on the body,
//      restore linearDamping=0, and clear the counter. This is
//      the only place in the codebase where setLinvel is called
//      on an existing dynamic body — a clean transition point
//      that ends the spawn-resolution phase (blueprint §3 P2
//      exception M7).
//   3. SPAWN_DAMPING_SUBSTEPS is now a SAFETY CAP (≤ 30 substeps
//      ≈ 0.25 s real-time). The normal path terminates earlier
//      via overlap-detection. If the cap is hit (e.g., body
//      completely engulfed → can't separate), we release damping
//      and accept current state — DOES NOT snap velocity (the
//      cap is reserved for pathological cases).
//
// Non-overlapping spawns (drag-place into empty space, slingshot
// launch) skip this entirely — no damping, no termination, full
// launch velocity preserved.
//
// Rebuild paths (wrap-teleport, meta-drift) also skip this — see
// the isRebuild=true callers (code-review F1).
const SPAWN_DAMPING_VALUE    = 12.0;
const SPAWN_DAMPING_SUBSTEPS = 30;

let RAPIER_READY = false;
let world = null;
let eventQueue = null;       // RAPIER.EventQueue — collects contact start/stop
                             // events that fire INSIDE world.step. Drained
                             // each substep, then passed into the state-dump
                             // trace so the analyzer can see brief impacts
                             // even when the post-step contactPairsWith
                             // query sees nothing (because the contact has
                             // already resolved by the time we query).
let bodyById = new Map();    // entity.id (u32) → Rapier RigidBody
let colliderById = new Map(); // entity.id → Rapier Collider
let overlapMgr = null;

// ── Wrap-boundary ghost bodies ──────────────────────────────────────
// Rapier's broadphase is Euclidean — it has no notion of toroidal
// topology. To make bodies near opposite edges "see" each other for
// contact, we maintain mirror copies (ghosts) placed at the 8 possible
// wrap offsets. Each ghost is a dynamic body whose position/velocity
// are slaved to its real body at the start of each substep. Other
// bodies can collide with ghosts; the contact impulse the ghost
// receives is forwarded to the real body via applyImpulse after the
// step (see step()'s ghost-delta phase).
//
// Map: realEntityId → Map<edgeSig, ghostBody>
// edgeSig is one of: 'L', 'R', 'T', 'B', 'LT', 'LB', 'RT', 'RB'.
const ghostsByRealId = new Map();

// Collision-group encoding: Rapier packs membership and filter masks
// into a single u32 as `(membership << 16) | filter`.
// REAL: belongs to group 1, collides with groups 1 and 2 (REAL and GHOST).
// GHOST: belongs to group 2, collides only with group 1 (REAL — never
//        with other GHOSTs, which would double-count cross-wrap contacts).
const COLLISION_GROUP_REAL  = (0x0001 << 16) | 0x0003;
const COLLISION_GROUP_GHOST = (0x0002 << 16) | 0x0001;

// Floor for the dynamic ghost-creation margin. The effective margin
// used per-substep is `max(GHOST_MARGIN_PX_FLOOR, maxRadiusInScene + 30)`
// so even with very-large-radius bodies (UI radius slider up to 10×
// base), any cross-wrap contact pair always has both ghosts spawned
// before they could physically contact. See _effectiveGhostMargin.
const GHOST_MARGIN_PX_FLOOR = 120;

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

// `isRebuild = true` is set by callers that destroy + recreate an
// existing entity's body (wrap-teleport, meta-drift slider edit). On
// the rebuild path we MUST NOT apply the spawn-damping burst because:
//   - wrap-rebuild: the body has been continuously simulated and is
//     just being teleported; treating each wrap as a "spawn" would
//     decay its velocity 34 %/substep × 4 every time it crossed an
//     edge — destroying orbits in wrap mode (see code-review F1).
//   - meta-drift rebuild: the user is interactively editing radius/
//     mass via the slider; injecting damping mid-edit would feel
//     unpredictable. The softer global contact spring still resolves
//     any new overlap gently.
// Initial spawn (user click → input.js → state.entities.push) is
// the only path that calls this with isRebuild = false (the default).
function createBodyForEntity(e, isRebuild = false) {
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
    .setRestitution(state.elasticRestitution)
    // ActiveEvents.COLLISION_EVENTS opts this collider into emitting
    // contact-started / contact-stopped notifications into the world's
    // EventQueue. Default is "no events" — we need this on every collider
    // (sensors included) for the state-dump trace to see brief impacts.
    .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)
    // Real body — collides with both other REAL bodies and with GHOSTs
    // (which represent real bodies on the other side of a wrap edge).
    .setCollisionGroups(COLLISION_GROUP_REAL);
  // Black holes are sensors — they detect overlap (so the absorption
  // lifecycle in detectAndStartBHAbsorptions can fire) but apply no
  // impulse to penetrating planets. Sensor "intersection" start/stop also
  // fires through drainCollisionEvents when COLLISION_EVENTS is enabled.
  if (e.type === 'black_hole') cDesc = cDesc.setSensor(true);

  const collider = world.createCollider(cDesc, body);

  // We store isCcd as a JS-side mirror to skip redundant Rapier toggles.
  body._isCcdEnabled = wantsCcd;

  // Stamp the entity's "physics-baked" meta. syncWorldToEntities checks
  // these every substep — if e.mass / e.radius / e.type drift (because
  // the user moved the mass or radius slider while the entity is selected,
  // or toggled planet ↔ black_hole), the body MUST be destroyed and
  // recreated (Rapier has no setMass / setRadius / setSensor API safe to
  // call mid-life). Stamping here keeps the bake-vs-live comparison cheap.
  e._bakedMass   = e.mass;
  e._bakedRadius = e.radius;
  e._bakedType   = e.type;

  bodyById.set(e.id, body);
  colliderById.set(e.id, collider);

  // Conditional spawn-damping: only if this is an INITIAL spawn
  // (not a rebuild) AND the body actually overlaps an existing
  // collider at creation time, apply a 4-substep linear-damping
  // burst to absorb the contact-spring's separation energy.
  // Non-overlapping spawns (empty space / slingshot) and rebuilds
  // (wrap-teleport, meta-drift) get nothing here. The overlap
  // check uses Rapier's native intersectionsWithShape query (no
  // JS-side geometry iteration). Black-hole sensors are excluded
  // (they don't physically collide; absorption is JS-driven).
  if (!isRebuild &&
      e.type !== 'black_hole' &&
      _spawnOverlapsExisting(e, collider.handle)) {
    body.setLinearDamping(SPAWN_DAMPING_VALUE);
    e._spawnDampingSubstepsLeft = SPAWN_DAMPING_SUBSTEPS;
  }

  return body;
}

// Asks Rapier's narrowphase: "given a ball of radius e.radius at
// (e.x, e.y), does it intersect any existing collider?" Returns true
// on first hit. Uses world.intersectionsWithShape — Rapier's spatial
// query, NOT a JS-side geometric iteration. The just-created
// collider for this entity is excluded via filterExcludeCollider so
// the query never matches the body against itself.
//
// Filter: COLLISION_GROUP_REAL = (0x0001 << 16) | 0x0003.
//   - Query treats itself as group 1 (REAL).
//   - Filter mask 0x0003 = groups 1+2 → matches both REAL bodies AND
//     GHOST wrap-mirror bodies.
// Hitting a GHOST IS a real overlap: the ghost represents matter on
// the other side of the wrap edge, and the spawn-damping should
// suppress the would-be explosion when a body spawns next to a
// wrap-mirrored body.
// Per-substep "is this spawn-damped body still overlapping anything?"
// query, used by the countdown loop to detect the clean-exit moment.
// Reads the body's CURRENT Rapier position (not the JS mirror, which
// is stale by one substep) and asks the narrowphase for any
// intersecting collider via intersectionsWithShape.
//
// On query failure (API mismatch / WASM hiccup) returns true (be
// conservative — keep damping rather than prematurely snap to v=0).
function _bodyStillOverlapping(body, e, ownColliderHandle) {
  if (!world) return false;
  const pos = body.translation();
  const r = Math.max(e.radius, 0.01);
  const shape = new RAPIER.Ball(r);
  let hit = false;
  try {
    world.intersectionsWithShape(
      { x: pos.x, y: pos.y },
      0,
      shape,
      (otherHandle) => { hit = true; return false; },  // first hit aborts
      undefined,
      COLLISION_GROUP_REAL,
      ownColliderHandle,
    );
  } catch {
    try {
      hit = false;
      world.intersectionsWithShape(
        { x: pos.x, y: pos.y },
        0,
        shape,
        (otherHandle) => {
          if (otherHandle !== ownColliderHandle) { hit = true; return false; }
          return true;
        },
      );
    } catch {
      return true;  // can't tell — keep damping rather than snap prematurely
    }
  }
  return hit;
}

function _spawnOverlapsExisting(e, ownColliderHandle) {
  if (!world) return false;
  const r = Math.max(e.radius, 0.01);
  const shape = new RAPIER.Ball(r);
  let hit = false;
  try {
    world.intersectionsWithShape(
      { x: e.x, y: e.y },
      0,                         // rotation
      shape,
      (otherHandle) => { hit = true; return false; },  // first hit aborts
      undefined,                 // filterFlags
      COLLISION_GROUP_REAL,      // membership=REAL → collides with REAL+GHOST;
                                 // but the filter is interpreted by Rapier as
                                 // "treat the query shape as a member of group
                                 // 1 (REAL) — i.e., test against group-1 colliders
                                 // and group-2 colliders that allow group 1 contact"
      ownColliderHandle,         // exclude self
    );
  } catch (err) {
    // Some 0.19 bundles have a slightly different parameter list. Fall
    // back to the minimal signature; downside is the self-exclusion
    // filter is dropped, so we have to manually re-check the hit handle.
    try {
      hit = false;
      world.intersectionsWithShape(
        { x: e.x, y: e.y },
        0,
        shape,
        (otherHandle) => {
          if (otherHandle !== ownColliderHandle) { hit = true; return false; }
          return true;
        },
      );
    } catch {
      // If the entire query API isn't reachable, fail safe by skipping
      // spawn-damping (body spawns normally — original explosion
      // behavior remains, but no further regression).
      return false;
    }
  }
  return hit;
}

function destroyBody(id) {
  // Destroy all ghosts of this real body first — they have no purpose
  // without the real body to mirror.
  destroyAllGhostsOfRealId(id);
  const b = bodyById.get(id);
  if (b) {
    try { world.removeRigidBody(b); } catch {}
    bodyById.delete(id);
  }
  colliderById.delete(id);
}

// ── Wrap-boundary ghost helpers ──────────────────────────────────────
// The 8 possible wrap offsets. Sig is a stable string key so we can
// reuse ghost bodies across substeps (creating + destroying every step
// would be expensive at scale).
const _GHOST_EDGE_SIGS = ['L', 'R', 'T', 'B', 'LT', 'LB', 'RT', 'RB'];

function _ghostOffsetFor(sig, W, H) {
  switch (sig) {
    case 'L':  return { dx:  W, dy:  0 };
    case 'R':  return { dx: -W, dy:  0 };
    case 'T':  return { dx:  0, dy:  H };
    case 'B':  return { dx:  0, dy: -H };
    case 'LT': return { dx:  W, dy:  H };
    case 'LB': return { dx:  W, dy: -H };
    case 'RT': return { dx: -W, dy:  H };
    case 'RB': return { dx: -W, dy: -H };
  }
  return { dx: 0, dy: 0 };
}

function _effectiveGhostMargin(entities) {
  // The margin must cover the OTHER body's radius too: for a cross-wrap
  // contact A↔B, distance through wrap = (A's distance from edge)
  // + (B's distance from opposite edge), which has to be ≤ A.r + B.r
  // for them to touch. So both bodies must be within max(other radius)
  // of their respective edges. We use max-radius-in-scene + 30 px
  // safety so the dynamic margin scales with the actual bodies.
  let maxR = 0;
  for (const e of entities) {
    if (e.absorbing) continue;
    if (e.radius > maxR) maxR = e.radius;
  }
  return Math.max(GHOST_MARGIN_PX_FLOOR, maxR + 30);
}

function _neededEdgeSigs(realPos, realRadius, viewport, margin) {
  const W = viewport.width, H = viewport.height;
  const reach = realRadius + margin;
  const nearL = realPos.x < reach;
  const nearR = realPos.x > W - reach;
  const nearT = realPos.y < reach;
  const nearB = realPos.y > H - reach;
  const sigs = [];
  if (nearL) sigs.push('L');
  if (nearR) sigs.push('R');
  if (nearT) sigs.push('T');
  if (nearB) sigs.push('B');
  if (nearL && nearT) sigs.push('LT');
  if (nearL && nearB) sigs.push('LB');
  if (nearR && nearT) sigs.push('RT');
  if (nearR && nearB) sigs.push('RB');
  return sigs;
}

function _createGhostBodyFor(e, sig, viewport) {
  const realBody = bodyById.get(e.id);
  if (!realBody) return null;
  const pos = realBody.translation();
  const vel = realBody.linvel();
  const { dx, dy } = _ghostOffsetFor(sig, viewport.width, viewport.height);

  // Dynamic — the ghost must be able to RECEIVE contact impulses so
  // we can forward the resulting velocity delta back to the real body.
  // Kinematic ghosts wouldn't pick up impulses and the forwarding
  // mechanism would degenerate.
  const desc = RAPIER.RigidBodyDesc.dynamic();
  desc.setTranslation(pos.x + dx, pos.y + dy);
  desc.setLinvel(vel.x, vel.y);
  // Ghosts must not sleep — their state is overwritten every substep
  // from the real body, so the "is this still moving" question is
  // meaningless for them. Disabling sleep avoids Rapier deciding to
  // skip integration for a ghost we're actively syncing.
  desc.setCanSleep(false);
  desc.lockRotations();
  const ghost = world.createRigidBody(desc);

  // userData encoding for the dump's contact-event trace:
  //   real body:  userData = e.id (non-negative integer)
  //   ghost body: userData = -(realId + 1)  (negative; subtract 1 to
  //                                          recover the realId)
  // This lets the analyzer tell them apart by sign and trace contacts
  // back to the real body that the ghost mirrors.
  ghost.userData = -(e.id + 1);
  ghost._realId = e.id;
  ghost._ghostSig = sig;

  // Same shape / mass / restitution as the real body (so contact
  // dynamics are physically equivalent). Friction stays 0 — wrap
  // semantics don't introduce friction. Sensor flag mirrors the
  // real collider so a black-hole ghost remains a sensor.
  const r = Math.max(e.radius, 0.01);
  const density = e.mass / (Math.PI * r * r);
  let cDesc = RAPIER.ColliderDesc.ball(r)
    .setDensity(density)
    .setFriction(0)
    .setRestitution(state.elasticRestitution)
    .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)
    // Ghost — collides ONLY with REAL bodies, never with other ghosts.
    // (Ghost-vs-ghost contact would be a topological artifact since both
    // mirror the same underlying matter via different offsets.)
    .setCollisionGroups(COLLISION_GROUP_GHOST);
  if (e.type === 'black_hole') cDesc = cDesc.setSensor(true);
  world.createCollider(cDesc, ghost);

  return ghost;
}

function destroyAllGhostsOfRealId(realId) {
  const myGhosts = ghostsByRealId.get(realId);
  if (!myGhosts) return;
  for (const ghost of myGhosts.values()) {
    try { world.removeRigidBody(ghost); } catch {}
  }
  myGhosts.clear();
  ghostsByRealId.delete(realId);
}

function destroyAllGhosts() {
  for (const realId of [...ghostsByRealId.keys()]) {
    destroyAllGhostsOfRealId(realId);
  }
}

// Per-substep ghost lifecycle. Run after syncWorldToEntities — the
// real bodies must already exist (created or updated for meta drift)
// before we mirror them. Also stashes each ghost's pre-step linvel on
// the ghost itself so the post-step impulse-forwarding phase can
// compute Δv = ghost.linvel_after - ghost._preStepVel and apply it
// to the real body.
function syncGhosts(entities, viewport, boundaryMode) {
  if (boundaryMode !== 'wrap') {
    destroyAllGhosts();
    return;
  }
  const W = viewport.width, H = viewport.height;
  if (W <= 0 || H <= 0) return;
  // Margin scales with the largest radius currently in play so even
  // big bodies have ghosts spawned before they could physically touch
  // a partner across the wrap edge.
  const margin = _effectiveGhostMargin(entities);
  // Pinned bodies (kinematic-position-based real bodies) still get
  // ghosts: even though applyImpulse is a no-op on kinematic bodies
  // (so the pinned body never moves from cross-wrap contacts, which is
  // correct), the GHOST being dynamic means OTHER bodies still bounce
  // off the ghost-of-pinned-wall through the edge. Dropping pinned
  // ghosts as an optimization would break cross-wrap contact with
  // pinned walls.
  for (const e of entities) {
    // Absorbing entities have no Rapier body — and shouldn't be mirrored.
    if (e.absorbing) {
      destroyAllGhostsOfRealId(e.id);
      continue;
    }
    const realBody = bodyById.get(e.id);
    if (!realBody) continue;
    const pos = realBody.translation();
    const vel = realBody.linvel();
    const needed = _neededEdgeSigs(pos, e.radius, viewport, margin);

    let myGhosts = ghostsByRealId.get(e.id);
    if (!myGhosts) {
      if (needed.length === 0) continue;  // common case: not near any edge
      myGhosts = new Map();
      ghostsByRealId.set(e.id, myGhosts);
    }
    const neededSet = new Set(needed);

    // Update or create ghosts for required offsets
    for (const sig of needed) {
      const off = _ghostOffsetFor(sig, W, H);
      let ghost = myGhosts.get(sig);
      if (!ghost) {
        ghost = _createGhostBodyFor(e, sig, viewport);
        if (ghost) myGhosts.set(sig, ghost);
      } else {
        // Slave the ghost's pose + velocity to the real body. P1/P2
        // exception M6: ghost bodies are broadphase proxies, not
        // simulation participants — their state IS by construction
        // the real body's state translated by the wrap offset.
        // Cost note: 2 WASM round-trips per ghost per substep. At
        // N=2000 with ~10% edge-adjacent bodies, ~400 round-trips per
        // substep. If this ever shows up in profiling, consider a
        // single batched setBothPositionAndVelocity call (Rapier
        // doesn't expose one today; would need to upstream).
        ghost.setTranslation({ x: pos.x + off.dx, y: pos.y + off.dy }, false);
        ghost.setLinvel({ x: vel.x, y: vel.y }, false);
      }
      // Stash pre-step linvel for the impulse-forwarding phase. Read
      // back via ghost._preStepVx / _preStepVy in step().
      if (ghost) {
        ghost._preStepVx = vel.x;
        ghost._preStepVy = vel.y;
      }
    }
    // Destroy ghosts no longer needed (real body drifted away from an edge)
    for (const [sig, ghost] of myGhosts) {
      if (!neededSet.has(sig)) {
        try { world.removeRigidBody(ghost); } catch {}
        myGhosts.delete(sig);
      }
    }
    if (myGhosts.size === 0) ghostsByRealId.delete(e.id);
  }
  // GC: any ghost whose real entity is gone or absorbing.
  const liveIds = new Set(entities.filter(e => !e.absorbing).map(e => e.id));
  for (const realId of [...ghostsByRealId.keys()]) {
    if (!liveIds.has(realId)) destroyAllGhostsOfRealId(realId);
  }
}

// After world.step, each ghost has a velocity delta = (linvel_after
// minus the linvel we set it to at sync time). That delta represents
// the contact impulses Rapier applied to the ghost during this step.
// Forwarding it to the real body as an applyImpulse(delta × mass)
// gives the real body the contact response it would have received if
// Rapier knew about toroidal topology natively. The delta excludes
// any gravity (we never apply gravity to ghosts) so it isolates the
// cross-wrap contact contribution cleanly.
function forwardGhostImpulses() {
  for (const myGhosts of ghostsByRealId.values()) {
    for (const ghost of myGhosts.values()) {
      const v = ghost.linvel();
      const dvx = v.x - (ghost._preStepVx || 0);
      const dvy = v.y - (ghost._preStepVy || 0);
      if (dvx === 0 && dvy === 0) continue;
      const realBody = bodyById.get(ghost._realId);
      if (!realBody) continue;
      const m = ghost.mass();
      // applyImpulse: Δv_real = impulse / mass_real. Ghost and real
      // share mass (same density × area), so passing (dv * m_ghost)
      // delivers exactly the velocity delta the ghost experienced.
      realBody.applyImpulse({ x: dvx * m, y: dvy * m }, true);
    }
  }
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
      // Detect "physics-baked" meta drift from a live UI slider edit:
      // mass / radius / type all require collider rebuild because Rapier
      // has no safe in-place setter for these. We pull current Rapier
      // pos/vel into the entity FIRST (so the rebuild inherits the
      // body's current trajectory, not the stale JS mirror), then
      // destroy + recreate. The new body picks up the changed meta in
      // createBodyForEntity. This is the canonical live-edit path —
      // it is the ONLY context where destroy+recreate is allowed for a
      // non-wrap, non-absorbing entity.
      if (e.mass   !== e._bakedMass ||
          e.radius !== e._bakedRadius ||
          e.type   !== e._bakedType) {
        const pos = b.translation();
        const vel = b.linvel();
        e.x  = pos.x;  e.y  = pos.y;
        e.vx = vel.x;  e.vy = vel.y;
        // Clear any in-flight spawn-damping burst — the body is being
        // rebuilt and the new body must start with a clean damping
        // state. createBodyForEntity is called with isRebuild=true so
        // it won't re-arm the burst even if the new (resized) body
        // overlaps a neighbor.
        if (e._spawnDampingSubstepsLeft !== undefined) {
          delete e._spawnDampingSubstepsLeft;
        }
        destroyBody(e.id);
        createBodyForEntity(e, /* isRebuild */ true);
        continue;  // skip the pin-toggle branch below — fresh body already has correct type
      }
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

  // Spawn-damping resolution loop. Bodies that spawned INTO existing
  // overlap carry e._spawnDampingSubstepsLeft. Each substep we:
  //   1. Query Rapier: is the body STILL overlapping anything? If
  //      not → CLEAN EXIT — setLinvel(0, 0), restore damping=0,
  //      clear counter. This is the v_rel = 0 guarantee per spec.
  //      (Blueprint §3 P2 exception M7: spawn-resolution finalize.)
  //   2. Still overlapping → tick the safety counter. At 0 we
  //      release damping but DO NOT touch velocity (cap is reserved
  //      for pathological "can't separate" cases; we accept whatever
  //      state Rapier produced).
  // Bodies spawned in empty space never had the field set and are
  // skipped via the === undefined fast path. Absorbing entities
  // skip early — their Rapier body was already destroyed.
  for (const e of entities) {
    if (e._spawnDampingSubstepsLeft === undefined) continue;
    if (e.absorbing) continue;
    const b = bodyById.get(e.id);
    if (!b) {
      delete e._spawnDampingSubstepsLeft;
      continue;
    }
    const myCollider = colliderById.get(e.id);
    const ownHandle = myCollider ? myCollider.handle : undefined;
    if (ownHandle !== undefined && !_bodyStillOverlapping(b, e, ownHandle)) {
      // No more overlap → finalize at v=0. Critical damping during
      // the burst should have already brought v close to 0, so this
      // is a small correction, not a jarring stop.
      b.setLinvel({ x: 0, y: 0 }, false);
      b.setLinearDamping(0);
      delete e._spawnDampingSubstepsLeft;
      continue;
    }
    // Still overlapping: tick safety counter.
    e._spawnDampingSubstepsLeft--;
    if (e._spawnDampingSubstepsLeft <= 0) {
      // Cap reached while still overlapping (rare — body engulfed
      // or solver can't find separation direction). Release damping,
      // accept current state — DO NOT setLinvel(0). The body
      // continues with whatever velocity Rapier produced.
      b.setLinearDamping(0);
      delete e._spawnDampingSubstepsLeft;
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
  // In wrap mode, use the minimum-image distance so a black hole near
  // one edge can absorb a planet that crossed the opposite edge. This
  // matches the gravity calculation (which also uses min-image) and
  // the ghost-body broadphase setup.
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
      const aBH = a.type === 'black_hole';
      const bBH = b.type === 'black_hole';
      if (!aBH && !bBH) continue;
      let dx = b.x - a.x, dy = b.y - a.y;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
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
  // Reading e.x/e.y here is safe: pullBodyStateToEntities ran immediately
  // before this in step(), so e.x/y equals body.translation() exactly.
  // Absorbing entities are skipped (their JS x/y is mid-animation, not
  // Rapier-authoritative — those bodies were already destroyed earlier).
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
  // Build an id→entity Map once instead of entities.find() per wrap (O(N)
  // per lookup → O(W·N) total; the Map makes it O(N + W)).
  if (wrappedIds.length > 0) {
    const idToEntity = new Map(entities.map(en => [en.id, en]));
    for (const id of wrappedIds) {
      const e = idToEntity.get(id);
      if (!e) continue;
      // Wrap-rebuild MUST pass isRebuild=true to suppress the spawn-
      // damping overlap check. Without this, a body orbiting through
      // the wrap edge would lose 34 %/substep × 4 of its velocity
      // every time it crossed — destroying any wrap-mode orbit
      // (code-review F1).
      destroyBody(id);
      createBodyForEntity(e, /* isRebuild */ true);
    }
  }
  // Note on the 1-substep ghost-gap window: destroyBody removed this
  // body's ghosts and createBodyForEntity does NOT make new ones. The
  // next substep's syncGhosts repopulates ghosts at the new (wrapped)
  // position. Worst case is a single substep (~17 ms wall at default
  // timeScale) where the wrapped body has no ghosts; given the dynamic
  // margin (≥ maxRadius + 30 px) and typical body speeds (≤ 17 px /
  // substep at v=1000 px/s), any cross-wrap contact that would have
  // happened during this window would already have spawned the partner
  // body's ghost on the OPPOSITE edge in the previous substep — so the
  // pair can still be detected when its ghost is re-created. No
  // observable miss in typical use; if this becomes an issue, fold
  // syncGhosts call into the bottom of applyBoundaryAndRebuildOnWrap.
  return wrappedIds;
}

// ── Manifold reader (rapier2d-compat 0.19 API) ──────────────────────
// `world.contactPair(c1, c2)` does NOT exist as a sync return-value API
// in 0.19 — calling it returns undefined and the manifold details were
// silently lost (which is why the earlier dumps had nx/ny/depth=null
// across the board). The real API is callback-style on NarrowPhase:
//
//   world.narrowPhase.contactPair(handleA, handleB, (manifold, flipped) => …)
//
// `manifold.localNormal1()` is the contact normal in collider1's local
// frame, pointing from collider1 → collider2. Our balls have rotations
// locked (createBodyForEntity → desc.lockRotations), so local == world.
//
// When `flipped` is true, the manifold treats handleB as its internal
// collider1, so localNormal1 is pointing from B → A. We negate so the
// returned (nx, ny) is always oriented from A → B regardless of how the
// pair was ordered internally.
//
// Returns { nx, ny, depth } with all three null when no live manifold
// exists for the pair (e.g. when called for an end-of-contact event
// after Rapier has dissolved the manifold).
function readManifold(narrowPhase, handleA, handleB) {
  let nx = null, ny = null, depth = null;
  try {
    narrowPhase.contactPair(handleA, handleB, (manifold, flipped) => {
      if (!manifold || manifold.numContacts() === 0) return;
      const n1 = manifold.localNormal1();
      // Defensive: Rapier can emit numContacts>0 with a degenerate
      // localNormal1=(0,0) when bodies are exactly coincident. A zero
      // vector silently passes the null-check downstream and the
      // offline analyzer would divide by |n|=0 when decomposing v
      // onto the normal. Leave the fields null in that case so the
      // truncation is explicit.
      const mag = Math.hypot(n1.x, n1.y);
      if (mag < 1e-9) return;
      const sign = flipped ? -1 : 1;
      nx = +(sign * n1.x).toFixed(4);
      ny = +(sign * n1.y).toFixed(4);
      depth = +manifold.solverContactDepth(0).toFixed(4);
    });
  } catch {}
  return { nx, ny, depth };
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

      // Softer TGS-Soft contact spring (vs Rapier default ~377 rad/s)
      // — this is what prevents the "spawn explosion" when a body
      // is created inside another. The conditional spawn-damping
      // burst in createBodyForEntity finishes the job for the
      // overlapping spawn case. See top-of-file constants block for
      // the physics derivation and trade-off analysis.
      //
      // rapier2d-compat 0.19 exposes only setters here (no getters);
      // we wrap in try/catch so an unknown-property runtime failure
      // doesn't bring down the entire init.
      try {
        world.integrationParameters.contact_natural_frequency = CONTACT_NATURAL_FREQUENCY;
      } catch (err) {
        if (isVerbose()) console.warn('[physics-rapier] contact_natural_frequency setter failed:', err);
      }
      try {
        world.integrationParameters.normalizedAllowedLinearError = NORMALIZED_ALLOWED_LINEAR_ERROR;
      } catch (err) {
        if (isVerbose()) console.warn('[physics-rapier] normalizedAllowedLinearError setter failed:', err);
      }

      // autoDrain=true → drainCollisionEvents() clears the queue after
      // delivering events. Without this we'd accumulate every event
      // forever.
      eventQueue = new RAPIER.EventQueue(true);

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
      // Rapier owns position + velocity. syncWorldToEntities handles
      // create / destroy / pin-toggle / meta-drift (mass / radius /
      // type changed via UI slider). There is NO push-JS-state-back
      // to Rapier — that would invalidate Rapier's warm-start and
      // contact persistence (see blueprint §3 P1/P2/P3).
      syncWorldToEntities(entities);

      // Wrap-boundary ghost lifecycle. Must run AFTER syncWorldToEntities
      // (so the real bodies it mirrors definitely exist) and BEFORE the
      // gravity impulse loop / world.step (so the ghosts are properly
      // staged for this substep's contact resolution). In non-wrap
      // modes this destroys any leftover ghosts and returns immediately.
      syncGhosts(entities, viewport, boundaryMode);

      // ── state-dump trace capture: pre-state ──────────────────────
      // Read directly from Rapier so the dump's "pre" snapshot is the
      // authoritative pre-step body state, not a JS mirror that lags by
      // one substep. The offline analyzer's
      //   Δv_solver = post.v - pre.v - gravity.a * dt
      // computation requires pre.v to be the velocity Rapier saw at the
      // start of this world.step (before addForce + step). That value
      // lives in body.linvel() at this point, never in e.vx.
      const pre = entities.map(e => {
        const b = bodyById.get(e.id);
        if (!b) {
          // No Rapier body — the entity is either absorbing (body was
          // destroyed earlier this substep by syncWorldToEntities) or in
          // some transient init state. Emit explicit nulls so the
          // offline analyzer does NOT treat the JS mirror as ground
          // truth for Rapier's pre-step state.
          return {
            id: e.id,
            x: null, y: null, vx: null, vy: null,
            sleeping: null,
          };
        }
        const pos = b.translation();
        const vel = b.linvel();
        return {
          id: e.id,
          x:  +pos.x.toFixed(3),
          y:  +pos.y.toFixed(3),
          vx: +vel.x.toFixed(3),
          vy: +vel.y.toFixed(3),
          sleeping: !!b.isSleeping(),
        };
      });

      const accels = await computeGravity(entities);

      // Per-entity gravity vector this substep. forceApplied starts false
      // and is flipped only when addForce actually runs — this lets the
      // offline analyzer distinguish "skipped" from "applied with
      // magnitude 0". Skip reasons match the state-dump.js docstring:
      //   absorbing | pinned | no-body | sleeping+tiny-impulse
      const gravity = entities.map((e, i) => ({
        id: e.id,
        ax: +accels[i * 2].toFixed(6),
        ay: +accels[i * 2 + 1].toFixed(6),
        forceApplied: false,
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
          // Force is big enough to wake — applyImpulse(..., wake=true)
          // wakes atomically inside Rapier. No need for a separate
          // wakeUp() WASM round-trip.
        }
        // Use applyImpulse, NOT addForce. Empirically rapier2d-compat
        // 0.19's `addForce` does NOT integrate over world.timestep the
        // way its docs claim — observed Δv per substep matches F/m
        // (impulse semantics) rather than F/m × dt (force semantics).
        // Compensating by passing impulse = mass × acceleration × dt
        // gives the correct Newtonian Δv = a × dt per substep. Using
        // applyImpulse (whose impulse → Δv = impulse/mass semantics
        // ARE deterministic per docs) is the cleanest way to express
        // this without depending on addForce's empirical behavior.
        b.applyImpulse(
          { x: e.mass * ax * dt, y: e.mass * ay * dt },
          true,
        );
        gravity[i].forceApplied = true;
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

      // Pass the eventQueue so Rapier fills it with start/stop events as
      // contacts form and break inside the solver substep. Without it,
      // the only way to see a contact is the post-step contactPairsWith
      // query — which misses brief impacts that have already resolved
      // by the time we look.
      world.step(eventQueue);

      // Forward cross-wrap contact impulses from ghosts to real bodies.
      // Must run BEFORE pullBodyStateToEntities and BEFORE the contact-
      // events drain (so the contact-events trace reflects the final
      // post-forwarding velocity for any subsequent analysis pass).
      // No-op in non-wrap modes since ghostsByRealId is empty.
      forwardGhostImpulses();

      // ── Drain contact events ─────────────────────────────────────
      // For each event we record: which entities, started-vs-ended, and
      // the body's velocity at drain time. Rapier does NOT expose the
      // velocity at the moment the event was emitted inside world.step —
      // every event in this slot shares the same post-step velocity for
      // its body. Use the adjacent slots' pre[]/post[] entries (by id)
      // to reconstruct what each body was doing immediately before /
      // after the contact.
      //
      // Body lookup can fail when a participating collider has been
      // destroyed (e.g. wrap-rebuild in the previous substep queued a
      // stop event that drains here); those entries get null ids and
      // zero velocities and are filterable downstream.
      //
      // CONTACT_EVENTS_CAP guards against a pathological dense cluster
      // generating thousands of events in one substep — an unbounded
      // push() loop would spike GC and distort the very timing data
      // we're trying to collect. 512 is ~8× the worst-case I expect
      // (32 contacting pairs × start+end).
      const CONTACT_EVENTS_CAP = 512;
      const contactEvents = [];
      let contactEventsTruncated = false;
      const eventNarrowPhase = world.narrowPhase;
      eventQueue.drainCollisionEvents((handle1, handle2, started) => {
        if (contactEvents.length >= CONTACT_EVENTS_CAP) {
          contactEventsTruncated = true;
          return;
        }
        const c1 = world.getCollider(handle1);
        const c2 = world.getCollider(handle2);
        const aBody = c1 ? c1.parent() : null;
        const bBody = c2 ? c2.parent() : null;
        const aId = aBody ? aBody.userData : null;
        const bId = bBody ? bBody.userData : null;
        let aVx = 0, aVy = 0, bVx = 0, bVy = 0;
        if (aBody) { const v = aBody.linvel(); aVx = v.x; aVy = v.y; }
        if (bBody) { const v = bBody.linvel(); bVx = v.x; bVy = v.y; }
        // Normal/depth: only meaningful for start events (the manifold
        // still exists). On end events the manifold has been dissolved
        // by Rapier and readManifold returns null fields, which is
        // exactly what we want recorded.
        const md = readManifold(eventNarrowPhase, handle1, handle2);
        contactEvents.push({
          aId,
          bId,
          started: !!started,
          aVx: +aVx.toFixed(3),
          aVy: +aVy.toFixed(3),
          bVx: +bVx.toFixed(3),
          bVy: +bVy.toFixed(3),
          nx: md.nx,
          ny: md.ny,
          depth: md.depth,
        });
      });

      // Single contact-iteration pass: counts truly-touching pairs (live
      // manifold with numContacts > 0) for the adaptive overlap manager
      // AND extracts manifold details (normal, depth) for the state-dump
      // trace. Dedup'd by sorted handle pair.
      //
      // CRITICAL: contactPairsWith enumerates broadphase-persistent pairs,
      // which persist for several substeps after physical separation. If
      // we count those as "touching", the adaptive iteration manager
      // escalates solver iters for pairs that no longer have contact
      // constraints — that escalation drives spurious impulse application
      // (the source of the post-contact tangential drift in dump A/B/C).
      // The filter `md.nx !== null` (i.e., readManifold confirmed
      // numContacts() > 0) drops the stale broadphase ghosts.
      const contactsTrace = [];
      const seenPairs = _contactCounterSeen;
      seenPairs.clear();
      const narrowPhase = world.narrowPhase;
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
          const md = readManifold(narrowPhase, aHandle, bHandle);
          if (md.nx === null) return;  // broadphase-persistent ghost; skip
          contactsTrace.push({ aId, bId, nx: md.nx, ny: md.ny, depth: md.depth });
        });
      });
      overlapMgr.recordPostStep(contactsTrace.length);

      pullBodyStateToEntities(entities);

      detectAndStartBHAbsorptions(entities);
      updateAbsorptions(entities, dt);
      const wrappedIds = applyBoundaryAndRebuildOnWrap(entities, viewport, boundaryMode);
      // Second syncWorldToEntities: handles bodies that became absorbing
      // in detectAndStartBHAbsorptions (destroys them) and re-verifies
      // bake-marker integrity for any wrap-rebuilt bodies (no-op because
      // createBodyForEntity already re-stamped the markers, so the
      // drift-detect branch sees equal values and skips).
      syncWorldToEntities(entities);

      // ── state-dump trace: post-state ─────────────────────────────
      // After pull-back + absorption + boundary + sync. This is the
      // "final" entity state for the substep. The recorder's offline
      // pass takes (pre.v, gravity.a, post.v) and computes
      //   Δv_solver = post.v - pre.v - gravity.a * dt
      // which isolates Rapier's solver contribution per substep.
      const post = entities.map(e => {
        const b = bodyById.get(e.id);
        // For live bodies prefer Rapier-direct reads (same rule as pre
        // snapshot — body state IS the source of truth). For destroyed
        // bodies (absorbing entity, or wrap-rebuilt then immediately
        // destroyed) fall back to the JS mirror set by
        // pullBodyStateToEntities + downstream JS bookkeeping; that
        // mirror IS what subsequent substeps + renderer will see.
        if (b) {
          const pos = b.translation();
          const vel = b.linvel();
          return {
            id: e.id,
            x:  +pos.x.toFixed(3),
            y:  +pos.y.toFixed(3),
            vx: +vel.x.toFixed(3),
            vy: +vel.y.toFixed(3),
            sleeping: !!b.isSleeping(),
          };
        }
        return {
          id: e.id,
          x:  +e.x.toFixed(3),
          y:  +e.y.toFixed(3),
          vx: +(e.vx || 0).toFixed(3),
          vy: +(e.vy || 0).toFixed(3),
          sleeping: null,
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
        contactEvents,
        contactEventsTruncated,
        wrappedEntityIds: wrappedIds,
      });
    },

    onEntityMetaMaybeChanged() {
      // No work needed: syncWorldToEntities polls each substep and
      // detects mass / radius / type drift via the baked markers
      // stamped in createBodyForEntity. UI code is free to mutate
      // e.mass / e.radius / e.type at any time; the rebuild happens
      // automatically on the next substep before world.step runs.
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
      const snapNarrowPhase = world.narrowPhase;
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
          // Same filter as the per-substep contactsTrace builder: drop
          // broadphase-persistent pairs that have no live manifold.
          // Without this, snapshot() consumers would see "ghost"
          // contacts that aren't physically touching.
          const md = readManifold(snapNarrowPhase, aHandle, bHandle);
          if (md.nx === null) return;
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
      destroyAllGhosts();
      for (const id of [...bodyById.keys()]) destroyBody(id);
      bodyById.clear();
      colliderById.clear();
      if (overlapMgr) { overlapMgr.reset(); overlapMgr = null; }
      if (eventQueue) { try { eventQueue.free(); } catch {} eventQueue = null; }
      if (world) { try { world.free(); } catch {} world = null; }
      if (gpuGravityHandle) { try { gpuGravityHandle.destroy(); } catch {} gpuGravityHandle = null; }
      if (gpuDevice)        { try { gpuDevice.destroy();        } catch {} gpuDevice        = null; }
    },
  };
}

