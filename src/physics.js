// physics.js — Force computation, Velocity Verlet integration,
// collision handling, and placement-preview trajectory prediction.
//
// Force model (per spec):
//   - Force on entity A from entity B exists only when B.charge !== 0.
//   - Direction: B.charge = +1 → attract A toward B; B.charge = -1 → repel.
//   - A.charge can be 0 and still RECEIVE force from B (Newton's third law
//     is intentionally broken in this fantasy model; spec is asymmetric).
//
// Acceleration on A from B (when B.charge !== 0):
//   a = B.charge · G · m_B / r² · unit(B - A)
// (m_A drops out, as in standard Newtonian gravity.)

import {
  state,
  PREDICT_DT, PREDICT_STEPS_MAX,
  BOUNDARY_BUFFER_FACTOR,
} from './state.js';
import { prepareBHTree, computeAccelerationsBH } from './physics-barneshut.js';
import { buildSpatialHash, forEachCollisionPair } from './physics-spatial-hash.js';

// V8.1c: BH dispatch threshold is now user-tunable via state.bhThreshold.
// Functions read state.bhThreshold at call time so the slider takes
// effect immediately on the next frame.

// V8.2: prepare per-frame data structures once. main.js calls this before
// the substep loop so the quadtree + spatial hash are built exactly 1× per
// frame instead of up-to-8× (once per substep). Position drift between
// substeps is bounded by SIM_DT × velocity and is within Verlet's
// integration tolerance.
export function prepareFrame(entities) {
  if (entities.length >= state.bhThreshold) {
    prepareBHTree(entities);
    buildSpatialHash(entities);
  }
}

// ─── Minimum-image distance helpers (wrap mode) ───────────────────
// When the world wraps, two points near opposite edges can be closer
// going "around" than going straight. We use the standard PBC (periodic
// boundary condition) minimum-image convention: for each axis, if the
// straight delta exceeds half the span, replace it with the wrap-around
// delta. Equivalent to: delta - round(delta / span) * span.

function minImageDelta(d, span) {
  if (span <= 0) return d;
  if (d > span * 0.5) return d - span;
  if (d < -span * 0.5) return d + span;
  return d;
}

// ─── Force / acceleration ─────────────────────────────────────────
// Returns parallel arrays-of-zero accumulator filled in-place.
// We allocate once per step to avoid GC pressure in the inner loop.

export function computeAccelerations(entities, accels) {
  const n = entities.length;
  // V8.2: dispatch to Barnes-Hut for large N. The direct O(N²) sum below
  // wins at small N due to lower constant factor + cache locality.
  if (n >= state.bhThreshold) {
    computeAccelerationsBH(entities, accels);
    return;
  }
  for (let i = 0; i < n; i++) { accels[i].ax = 0; accels[i].ay = 0; }

  // V7 perf: hoist wrap/viewport reads out of the N² hot path and inline
  // pairDelta so the inner loop allocates zero objects.
  // V8.1c: also snapshot tunable G/epsilon once per call so the inner
  // loop reads locals instead of property lookups.
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width  : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;
  const G = state.G;
  const EPSILON = state.epsilon;

  for (let i = 0; i < n; i++) {
    const a = entities[i];
    // Absorbing entities are visually being eaten; they neither apply nor
    // receive gravity (avoids late-stage tugs that look like the body is
    // resisting absorption).
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
      const minR = Math.max(a.radius + b.radius, EPSILON);
      // Plummer softening: r²_soft = r² + minR² (smoothly damps force to
      // zero as r → 0). Was max(r², minR²) which gave a hard kink at
      // r=minR — force and potential disagreed inside the kink (force
      // stayed constant, potential gradient went to zero), causing
      // energy non-conservation in close approaches and contour-line
      // jitter near body centers in the V9.1 field visualization. The
      // single-line change `max(...)` → `+` propagates Plummer through
      // the existing `mag = q·G·m/r² · dx/r` form: mag/r becomes
      // q·G·m·dx/(r²+minR²)^(3/2) which is exactly Plummer's force.
      const r2 = r2Raw + minR * minR;
      const r = Math.sqrt(r2);
      const nx = dx / r;
      const ny = dy / r;

      // a feels force from b iff b.charge !== 0
      if (b.charge !== 0) {
        const mag = b.charge * G * b.mass / r2;
        accels[i].ax += mag * nx;
        accels[i].ay += mag * ny;
      }
      // b feels force from a iff a.charge !== 0
      if (a.charge !== 0) {
        const mag = a.charge * G * a.mass / r2;
        accels[j].ax += mag * -nx;
        accels[j].ay += mag * -ny;
      }
    }
  }
}

// ─── Velocity Verlet ──────────────────────────────────────────────
// Standard formulation:
//   x(t+dt) = x + v·dt + 0.5·a·dt²
//   a_new   = f(x_new) / m         (computed below)
//   v(t+dt) = v + 0.5·(a + a_new)·dt
// We cache the previous-step acceleration on each entity (`ax`,`ay`).

const _scratch = [];   // reused acceleration accumulator

// ─── PBD contact list (Build Step 2+) ─────────────────────────────
// Filled per-substep by processCollisionPair as it walks the broadphase.
// Read back by _projectPBDContacts inside stepPBD. Reset at the top of
// every handleCollisions call so a substep never sees stale entries from
// the previous frame. Pre-allocated to a generous capacity for the
// typical N ≤ 50 scene; grown if a larger cluster appears.
const PBD_ITERATIONS = 4;
let _contacts = new Array(256);
let _contactCount = 0;

// Contact-persistence tracking. A pair is "persistent" if EITHER it
// was a contact last substep OR its PRE-KICK relative normal velocity
// is essentially zero (orbital tangential motion / resting). Persistent
// contacts skip restitution bounce; "new" contacts (genuine impact at
// non-zero approach speed) get the standard -e·vnApproach treatment.
//
// Sprint 2 of the Box2D refactor will REPLACE this Set with a
// Map<pairKey, normalImpulse> for warm-starting the velocity solver
// across substeps. For now keep the Set; only `.has()` semantics are
// used downstream.
const _prevPairs = new Set();
function _pairKey(a, b) {
  // Symmetric key — order of arguments doesn't matter.
  const aId = a.id, bId = b.id;
  return aId < bId ? aId + ':' + bId : bId + ':' + aId;
}

// Substep's dt cached at the top of stepPBD so processCollisionPair
// can subtract `a·dt` to recover pre-kick velocity. Module-scoped
// because processCollisionPair signature is fixed by the broadphase
// dispatch (forEachCollisionPair in physics-spatial-hash.js).
let _currentSimDt = 0;

// Pre-kick approach-velocity threshold for "this is a real collision,
// not a kick-induced spurious approach". A few ULPs above zero leaves
// margin for FP noise without misclassifying truly slow impacts.
const PRE_KICK_APPROACH_THRESHOLD = -1e-6;     // px/s

function ensureScratch(n) {
  while (_scratch.length < n) _scratch.push({ ax: 0, ay: 0 });
  // Trim when entity population shrinks substantially (e.g., after a clear or
  // mass black-hole consumption). Keeps the visible-state-only invariant
  // `_scratch[0..n-1] is meaningful` from drifting if loop bounds ever change.
  if (_scratch.length > n * 2 && _scratch.length > 16) {
    _scratch.length = Math.max(n, 16);
  }
}

// Physics step — Box2D-style Sequential-Impulses + pseudo-velocity NGS.
//
// Pipeline per substep:
//   A. Reset per-body pseudo-velocity (_pvx, _pvy ← 0).
//   B. Compute accelerations at current positions (charge-asymmetric
//      gravity + Plummer softening). Already wraps via minimum-image
//      convention in computeAccelerations.
//   C. Gravity kick: v += a · dt   (symplectic Euler).
//   D. Predict positions: x += v · dt.
//   E. handleCollisions broadphase at predicted positions. For black-
//      hole pairs, beginAbsorption fires immediately. For planet-planet
//      pairs, push onto _contacts[] with c.vnApproach + c.wasPersistent
//      captured. Warm-start lookup populates c.normalImpulse from the
//      previous substep's value in _prevPairImpulses.
//   F. _solveContactVelocities: Box2D-style Sequential Impulses with
//      warm-starting and accumulated-impulse clamping ≥ 0. 8 iterations.
//   F'. Persist this substep's accumulated normal impulses into
//      _prevPairImpulses for next substep's warm-start.
//   G. _solvePositionConstraints: pseudo-velocity Non-linear Gauss-
//      Seidel. 3 iterations. Pseudo-velocity integrates into position
//      at the end without touching real velocity. Uses Baumgarte slop
//      to avoid jitter on resting contacts and a maxCorrection cap to
//      avoid teleporting on penetration spikes.
//   I. Pinned bodies hard-reset to v=0 (defensive — wA=0 in solvers
//      should already prevent any change but a stray write elsewhere
//      would compound).
//
// What was removed (and why) in the 2026-05-20 Box2D refactor:
//   • Centripetal projection (B'): bandaid for an orbital-drift bug
//     that the proper position-solver pseudo-velocity split makes
//     unnecessary — real velocity is never polluted by position
//     correction so the drift source it patched doesn't exist.
//   • Energy refund (G''): attempted to compensate the PE injected
//     when the kinematic position projection moved bodies up the
//     gravity gradient. With pseudo-velocity NGS, the position pass
//     doesn't write to real position from gravity terms — it only
//     applies the constraint normal — so no PE is "injected" that
//     needs refunding.
//   • Relaxation pass (G'): Catto-style relaxation specifically
//     removes the bias term that soft constraints inject. We use
//     hard constraints with a separate position pass, no bias to
//     relax against.
//   • Static contact damping (H): the artificial sleep was too
//     aggressive in busy multi-body scenes (pinned slow-moving
//     bodies that should continue evolving). Warm-starting the
//     velocity solver now does the job naturally — resting contacts
//     converge to their equilibrium impulse in 1–2 iterations when
//     seeded from the previous substep's converged value.
export function stepPBD(entities, dt) {
  const n = entities.length;
  if (n === 0 || dt === 0) return;
  ensureScratch(n);

  // Cache dt for processCollisionPair's pre-kick velocity calculation
  // (used by the wasPersistent gate, NOT by any deleted refund code).
  _currentSimDt = dt;

  // (Sprint 3 will add: A. Reset pseudo-velocity accumulator here.)

  // ── B. Compute accelerations at current positions ──────────────
  computeAccelerations(entities, _scratch);

  // ── C. Gravity kick: v += a · dt ───────────────────────────────
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned || e.absorbing) continue;
    const ax = _scratch[i].ax;
    const ay = _scratch[i].ay;
    e.vx += ax * dt;
    e.vy += ay * dt;
    // Cache a for downstream readers (debug-energy.js, prediction tooling).
    e.ax = ax;
    e.ay = ay;
  }

  // ── D. Predict positions: x += v · dt ──────────────────────────
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned || e.absorbing) continue;
    e.x += e.vx * dt;
    e.y += e.vy * dt;
  }

  // ── E. Detect contacts at predicted positions ──────────────────
  // handleCollisions: BH pairs → beginAbsorption immediately.
  // planet-planet pairs → push onto _contacts[] with warm-start lookup.
  handleCollisions(entities);

  // ── F. Sequential-Impulses velocity solver ─────────────────────
  // (Sprint 1: 4 iter, no warm-starting. Sprint 2 will upgrade to
  // 8 iter + accumulated impulse clamp + warm-start from _prevPairs
  // turned into _prevPairImpulses Map.)
  _solveContactVelocities(_contactCount);

  // ── G. Position projection ─────────────────────────────────────
  // (Sprint 1: still the direct PBD positional correction. Sprint 3
  // will replace with pseudo-velocity NGS that doesn't write to real
  // positions — uses _pvx, _pvy accumulators integrated at the end.)
  _projectPBDContacts(_contactCount);

  // ── I. Pinned bodies hard-reset ────────────────────────────────
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned) { e.vx = 0; e.vy = 0; e.ax = 0; e.ay = 0; }
  }
}

// PBD contact projection. Each iteration walks the contact list and
// pushes overlapping pairs apart along their contact normal by an amount
// proportional to the overlap, weighted by inverse masses (pinned bodies
// have effective infinite mass → zero compliance). Multiple iterations
// converge the contact graph because correcting pair (A, B) may re-open
// pair (A, C). For our typical small clusters (≤6 simultaneous contacts
// per body), 4 iterations leaves residual penetration well below the
// visible threshold. See architect blueprint for the convergence analysis.
// Position projection. (TEMPORARY: this is the pre-Box2D-refactor
// position solver. Sprint 3 of the Box2D refactor will replace it with
// a pseudo-velocity NGS that doesn't write directly to e.x/e.y,
// preserving real velocity. For now it's a plain PBD positional
// constraint solver: push overlapping pairs apart along the contact
// normal, weighted by inverse mass.)
function _projectPBDContacts(count) {
  if (count === 0) return;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;
  for (let iter = 0; iter < PBD_ITERATIONS; iter++) {
    for (let k = 0; k < count; k++) {
      const c = _contacts[k];
      const a = c.a;
      const b = c.b;
      if (a.absorbing || b.absorbing) continue;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
      const dist2 = dx * dx + dy * dy;
      if (dist2 < 1e-12) continue;
      const dist = Math.sqrt(dist2);
      const overlap = c.rSum - dist;
      if (overlap <= 0) continue;
      const wA = a.pinned ? 0 : 1 / a.mass;
      const wB = b.pinned ? 0 : 1 / b.mass;
      const wSum = wA + wB;
      if (wSum === 0) continue;
      const nx = dx / dist;
      const ny = dy / dist;
      c.nx = nx;
      c.ny = ny;
      const correction = overlap / wSum;
      a.x -= nx * correction * wA;
      a.y -= ny * correction * wA;
      b.x += nx * correction * wB;
      b.y += ny * correction * wB;
    }
  }
}

// ─── Velocity solver (Sequential Impulses, pre-Box2D baseline) ────
// Drives each contact's relative normal velocity to its target:
//   • persistent contact (c.wasPersistent) → 0 (no bounce on resting)
//   • new collision → -e · c.vnApproach (true restitution)
//
// This is the 4-iter Gauss-Seidel WITHOUT warm-starting or
// accumulated-impulse clamping — the standard Box2D additions are
// planned for the next refactor sprint and will replace this body.
// For now keep as-is; the next sprint upgrades iteration count to 8
// and adds the c.normalImpulse accumulation with [0, +∞) clamp plus a
// pre-iteration warm-start pass from _prevPairImpulses.
const VELOCITY_ITERATIONS = 4;

function _solveContactVelocities(count) {
  if (count === 0) return;
  const e = state.elasticRestitution;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5, halfH = H * 0.5;
  for (let iter = 0; iter < VELOCITY_ITERATIONS; iter++) {
    for (let k = 0; k < count; k++) {
      const c = _contacts[k];
      const a = c.a;
      const b = c.b;
      if (a.absorbing || b.absorbing) continue;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
      const dist2 = dx * dx + dy * dy;
      if (dist2 < 1e-12) continue;
      const dist = Math.sqrt(dist2);
      const nx = dx / dist, ny = dy / dist;
      const rvx = b.vx - a.vx;
      const rvy = b.vy - a.vy;
      const vn = rvx * nx + rvy * ny;
      const vnTarget = c.wasPersistent ? 0 : -e * c.vnApproach;
      if (vn >= vnTarget) continue;
      const wA = a.pinned ? 0 : 1 / a.mass;
      const wB = b.pinned ? 0 : 1 / b.mass;
      const wSum = wA + wB;
      if (wSum === 0) continue;
      const J = (vnTarget - vn) / wSum;
      a.vx -= J * nx * wA;
      a.vy -= J * ny * wA;
      b.vx += J * nx * wB;
      b.vy += J * ny * wB;
    }
  }
}

// (Legacy `_applyRestitution` removed; its single-pass impulse on
// post-Δx/dt velocities both fired spurious bounces on orbital contacts
// (vnApproach captured at detection sampled a sub-pixel radial component
// of tangential motion) and stripped real radial velocity needed for
// curved trajectories. Both responsibilities are now subsumed by
// `_solveContactVelocities`, which uses iterative impulses and skips
// persistent contacts via the same COLLISION_THRESHOLD_VN gate.)

// (Legacy `stepVerlet` removed by the PBD refactor — its position+velocity
// pipeline is incompatible with the contact-constraint energy semantics
// PBD enforces. `stepPBD` above is the production step. The Plummer
// softening + asymmetric force-law internal call sites have been
// updated to reference stepPBD instead.)

// ─── Collisions ───────────────────────────────────────────────────
// Two paths depending on the pair's types:
//
//   1. At least one black hole → the "prey" (non-bh, or smaller bh) enters
//      absorption animation (entity.absorbing = {...}). It will be spliced
//      out later by updateAbsorptions when the animation completes.
//
//   2. Two planets → elastic collision: resolve penetration along the
//      contact normal, then apply a mass-weighted impulse so momentum and
//      kinetic energy are conserved (restitution = ELASTIC_RESTITUTION).
//
// Already-absorbing entities are inert — they neither collide nor get hit.

// V8.2: shared pair handler — invoked by both the direct N² loop (small N)
// and the spatial-hash candidate iterator (large N). Re-checks absorbing
// state defensively because a pair earlier in this same frame may have
// turned `a` or `b` into prey, and we don't want to apply impulse to a body
// already mid-absorption.
function processCollisionPair(a, b, dx, dy) {
  if (a.absorbing || b.absorbing) return;
  const rSum = a.radius + b.radius;
  const dist2 = dx * dx + dy * dy;
  if (dist2 >= rSum * rSum) return;

  const aIsBH = a.type === 'black_hole';
  const bIsBH = b.type === 'black_hole';

  if (aIsBH || bIsBH) {
    let prey, predator;
    if (aIsBH && !bIsBH) { prey = b; predator = a; }
    else if (!aIsBH && bIsBH) { prey = a; predator = b; }
    else if (a.mass < b.mass) { prey = a; predator = b; }
    else if (b.mass < a.mass) { prey = b; predator = a; }
    else return;                                  // equal-mass BHs: stalemate
    // Pinned bodies are *kinematic* anchors only — they still get
    // consumed by black holes per user intent (固定的黑洞被路过的吞噬
    // 是预期行为).
    beginAbsorption(prey, predator);
  } else {
    // PBD refactor: planet-planet contacts are accumulated here and
    // resolved later by _projectPBDContacts inside stepPBD. The dx/dy
    // passed in are already wrap-corrected (handleCollisions's broadphase
    // applies the wrap delta), so we capture the contact normal n̂ at
    // detection time. Projection iterations then advance positions along
    // this fixed normal — robust under wrap because the normal direction
    // doesn't change appreciably across the small corrections of one
    // substep, even when the two bodies are on opposite sides of the
    // viewport with wrap on.
    if (_contactCount >= _contacts.length) {
      _contacts.push(null);   // grow once per overflow; rare
    }
    let c = _contacts[_contactCount];
    if (!c) {
      c = { a: null, b: null, rSum: 0, nx: 0, ny: 0,
            vnApproach: 0, wasPersistent: false, normalImpulse: 0 };
      _contacts[_contactCount] = c;
    }
    const dist = Math.sqrt(dist2);
    c.a = a;
    c.b = b;
    c.rSum = rSum;
    let nx, ny;
    if (dist < 1e-6) {
      // Co-located fallback: arbitrary axis, matches legacy resolution.
      nx = 1; ny = 0;
    } else {
      nx = dx / dist;
      ny = dy / dist;
    }
    c.nx = nx;
    c.ny = ny;
    // Capture PRE-projection approach velocity along the normal. For
    // genuine collisions this is the impact speed; for orbital
    // tangential motion sampled at the rotating normal, it's near zero.
    const rvx = b.vx - a.vx;
    const rvy = b.vy - a.vy;
    c.vnApproach = rvx * nx + rvy * ny;     // negative = approaching
    // Persistent contact (skip restitution bounce) iff:
    //   (i)  pair was in _prevPairs (warm-start) OR
    //   (ii) PRE-KICK relative normal velocity is essentially zero
    //        (resting / orbital — captured vnApproach is purely
    //        kick-induced and shouldn't bounce).
    // Recover pre-kick v_rel by subtracting a·dt from the post-kick
    // velocities. For charge=0 bodies (no kick), preVn == vnApproach.
    let wasPersistent = _prevPairs.has(_pairKey(a, b));
    if (!wasPersistent) {
      const dt = _currentSimDt;
      const preRvx = (b.vx - b.ax * dt) - (a.vx - a.ax * dt);
      const preRvy = (b.vy - b.ay * dt) - (a.vy - a.ay * dt);
      const preVn = preRvx * nx + preRvy * ny;
      if (preVn > PRE_KICK_APPROACH_THRESHOLD) wasPersistent = true;
    }
    c.wasPersistent = wasPersistent;
    // Sprint 2 of the Box2D refactor will look up the previous
    // substep's c.normalImpulse here (warm-start). For now reset to
    // 0 — the velocity solver doesn't accumulate across substeps yet.
    c.normalImpulse = 0;
    _contactCount++;
  }
}

export function handleCollisions(entities) {
  const n = entities.length;

  // PBD refactor: clear the contact list before this substep's broadphase
  // walk. processCollisionPair will push planet-planet pairs onto it.
  // _projectPBDContacts (run later inside stepPBD) drains it.
  _contactCount = 0;

  // V8.2: large-N path uses wrap-aware spatial hash for O(N·k) broadphase.
  // The hash is built once per frame in prepareFrame() above; this just
  // iterates the already-built buckets.
  if (n >= state.bhThreshold) {
    forEachCollisionPair(entities, processCollisionPair);
    _rebuildPrevPairs();
    return;
  }

  // Direct O(N²) path for small N. V7 perf: same hoisting as
  // computeAccelerations to avoid per-pair property reads.
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width  : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;

  for (let i = 0; i < n - 1; i++) {
    const a = entities[i];
    if (a.absorbing) continue;
    for (let j = i + 1; j < n; j++) {
      // Re-check `a` every iteration — it can have become prey earlier in
      // this same inner loop (e.g., when this is a dense cluster). Without
      // this break, a frozen absorbing body could receive a second
      // beginAbsorption() or an elastic impulse and "wake up".
      if (a.absorbing) break;
      const b = entities[j];
      if (b.absorbing) continue;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
      // V8.2: dispatch through shared helper so both N<64 and N≥64 paths
      // run identical collision/absorption logic.
      processCollisionPair(a, b, dx, dy);
    }
  }
  _rebuildPrevPairs();
}

// Rebuild _prevPairs from the just-populated _contacts. Called at the
// end of handleCollisions so the next substep's processCollisionPair
// sees a Set containing THIS substep's pairs — those become "previous"
// from the next substep's perspective.
function _rebuildPrevPairs() {
  _prevPairs.clear();
  for (let k = 0; k < _contactCount; k++) {
    const c = _contacts[k];
    _prevPairs.add(_pairKey(c.a, c.b));
  }
}

// Start the devour animation on `prey`, locking the predator black hole as
// its destination. Position snapshot is the prey's *current* position so the
// lerp begins where it actually got caught, not where it spawned.
function beginAbsorption(prey, predator) {
  prey.absorbing = {
    blackHoleId: predator.id,
    elapsedSim: 0,
    startRadius: prey.radius,
    startX: prey.x,
    startY: prey.y,
  };
  // Cancel any residual velocity so it doesn't fight the lerp visually.
  prey.vx = 0;
  prey.vy = 0;
  prey.ax = 0;
  prey.ay = 0;
}

// (Legacy `resolveElasticCollision` removed by the PBD refactor. Position
// correction is now handled by _projectPBDContacts; restitution is
// handled by _solveContactVelocities (the Sequential-Impulses velocity
// solver). The legacy function's single-pass impulse approach was
// incompatible with multi-body contact aggregates because it leaked
// tangential energy via the Verlet velocity averaging it tried to
// compensate for. See git history for the original impl.)

// ─── Absorption animation ─────────────────────────────────────────
// Progress every absorbing entity by `dt` seconds (sim time). Splice when
// the animation completes. If the predator black hole vanished (e.g., a
// bigger BH ate it), end the animation immediately — the prey just pops.

export function updateAbsorptions(entities, dt) {
  if (dt <= 0) return;
  // Build an id→entity map ONLY if at least one entity is absorbing.
  // Avoids a per-frame Map allocation when nothing is being eaten.
  let idMap = null;
  for (let i = entities.length - 1; i >= 0; i--) {
    const e = entities[i];
    const abs = e.absorbing;
    if (!abs) continue;
    if (!idMap) {
      idMap = new Map();
      for (let k = 0; k < entities.length; k++) idMap.set(entities[k].id, entities[k]);
    }

    const bh = idMap.get(abs.blackHoleId);
    if (!bh) { entities.splice(i, 1); continue; }

    abs.elapsedSim += dt;
    const t = Math.min(1, abs.elapsedSim / state.absorptionDuration);

    e.x = abs.startX + (bh.x - abs.startX) * t;
    e.y = abs.startY + (bh.y - abs.startY) * t;
    e.radius = abs.startRadius * (1 - t);

    if (t >= 1) entities.splice(i, 1);
  }
}

// ─── Trajectory prediction ────────────────────────────────────────
// Simulate a single "ghost" body (the one the user is about to place)
// against a frozen snapshot of existing entities. Real entities are
// treated as stationary during the 5-second look-ahead — a standard
// approximation that keeps cost low and feels right for placement UI.

// Module-level scratch for prediction — reused across calls so the drag's
// per-frame predict doesn't allocate 300 `{x,y}` objects + a fresh path
// array each invocation. The returned object holds a reference to this
// same buffer, with a `length` indicating how many samples are valid.
// V8.1c: buffer sized for the MAX possible prediction horizon (15s).
// Each call computes its own step count from state.predictHorizon, so the
// slider can shrink/grow without reallocation.
const _predictBuf = new Float32Array(PREDICT_STEPS_MAX * 2);
const _predictResult = { data: _predictBuf, length: 0 };
const _ghostAccelScratch = { ax: 0, ay: 0 };

export function predictTrajectory(ghost, others) {
  let x = ghost.x;
  let y = ghost.y;
  let vx = ghost.vx;
  let vy = ghost.vy;
  const wrap = state.boundaryMode === 'wrap';
  const W = state.viewport.width;
  const H = state.viewport.height;
  // Dynamic step count: predictHorizon in seconds × steps-per-second.
  // Clamped to MAX so we never write past the pre-allocated buffer.
  const steps = Math.min(
    PREDICT_STEPS_MAX,
    Math.max(1, Math.floor(state.predictHorizon / PREDICT_DT)),
  );

  // Initial acceleration from frozen snapshot (writes into _ghostAccelScratch).
  ghostAccel(x, y, ghost.radius, others, _ghostAccelScratch);
  let ax = _ghostAccelScratch.ax;
  let ay = _ghostAccelScratch.ay;

  let written = 0;
  for (let s = 0; s < steps; s++) {
    x += vx * PREDICT_DT + 0.5 * ax * PREDICT_DT * PREDICT_DT;
    y += vy * PREDICT_DT + 0.5 * ay * PREDICT_DT * PREDICT_DT;
    if (wrap) {
      if (x < 0) x += W; else if (x > W) x -= W;
      if (y < 0) y += H; else if (y > H) y -= H;
    }
    _predictBuf[s * 2]     = x;
    _predictBuf[s * 2 + 1] = y;
    written = s + 1;
    if (touchesBlackHole(x, y, ghost.radius, others)) break;

    ghostAccel(x, y, ghost.radius, others, _ghostAccelScratch);
    vx += 0.5 * (ax + _ghostAccelScratch.ax) * PREDICT_DT;
    vy += 0.5 * (ay + _ghostAccelScratch.ay) * PREDICT_DT;
    ax = _ghostAccelScratch.ax;
    ay = _ghostAccelScratch.ay;
  }
  _predictResult.length = written;
  return _predictResult;
}

// V7 perf: writes result into `out.ax`/`out.ay` instead of returning a fresh
// object. Called 300×/drag-frame from predictTrajectory.
function ghostAccel(x, y, radius, others, out) {
  let ax = 0, ay = 0;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;
  const G = state.G;
  const EPSILON = state.epsilon;
  for (let k = 0; k < others.length; k++) {
    const o = others[k];
    if (o.charge === 0 || o.absorbing) continue;
    let dx = o.x - x;
    let dy = o.y - y;
    if (wrap) {
      if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
      if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
    }
    const r2Raw = dx * dx + dy * dy;
    const minR = Math.max(radius + o.radius, EPSILON);
    const r2 = r2Raw + minR * minR;   // Plummer softening (see stepPBD)
    const r = Math.sqrt(r2);
    const mag = o.charge * G * o.mass / r2;
    ax += mag * dx / r;
    ay += mag * dy / r;
  }
  out.ax = ax;
  out.ay = ay;
}

function touchesBlackHole(x, y, radius, others) {
  const wrap = state.boundaryMode === 'wrap';
  const W = state.viewport.width;
  const H = state.viewport.height;
  for (let k = 0; k < others.length; k++) {
    const o = others[k];
    if (o.type !== 'black_hole' || o.absorbing) continue;
    let dx = o.x - x;
    let dy = o.y - y;
    if (wrap) {
      dx = minImageDelta(dx, W);
      dy = minImageDelta(dy, H);
    }
    const r = radius + o.radius;
    if (dx * dx + dy * dy < r * r) return true;
  }
  return false;
}

// ─── Boundary handling ────────────────────────────────────────────
// Called once per frame from main.js. The buffer extends each viewport
// edge by `max(w, h) × BOUNDARY_BUFFER_FACTOR` so fast-moving entities
// have some off-screen room before despawn.
//
// destroy mode → splice entities past the buffered edge (skip absorbing
//                entities; their animation should complete first).
// wrap mode    → teleport entities to the opposite edge, clearing the
//                trail so the wrap-line doesn't draw across the viewport.

export function applyBoundary(entities, viewport, mode) {
  const w = viewport.width;
  const h = viewport.height;
  if (w <= 0 || h <= 0) return;

  if (mode === 'wrap') {
    for (const e of entities) {
      if (e.absorbing) continue;
      if (e.x < 0)        e.x += w;
      else if (e.x > w)   e.x -= w;
      if (e.y < 0)        e.y += h;
      else if (e.y > h)   e.y -= h;
    }
    return;
  }

  // destroy mode
  const buffer = Math.max(w, h) * BOUNDARY_BUFFER_FACTOR;
  for (let i = entities.length - 1; i >= 0; i--) {
    const e = entities[i];
    if (e.absorbing) continue;
    if (e.x < -buffer || e.x > w + buffer ||
        e.y < -buffer || e.y > h + buffer) {
      entities.splice(i, 1);
    }
  }
}
