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

// V8.3: prepare per-frame data structures. main.js calls this before the
// substep loop. The BH quadtree is built ONCE per frame — gravity is a
// continuous force (∝ 1/r²) and tolerates the SIM_DT × |v| position drift
// between substeps within Verlet's integration tolerance. The spatial hash
// is NOT built here (V8.3 fix): unlike gravity, the broadphase is a binary
// classifier — a 0.5 px drift that crosses a cell boundary causes a 100%
// miss for that pair, not a small numerical error. So the hash is rebuilt
// per-substep inside handleCollisions (see comment there).
export function prepareFrame(entities) {
  if (entities.length >= state.bhThreshold) {
    prepareBHTree(entities);
    // buildSpatialHash moved to handleCollisions for per-substep freshness.
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
// Read back by _solvePositionConstraints inside stepPBD. Reset at the top of
// every handleCollisions call so a substep never sees stale entries from
// the previous frame. Pre-allocated to a generous capacity for the
// typical N ≤ 50 scene; grown if a larger cluster appears.
let _contacts = new Array(256);
let _contactCount = 0;

// Contact persistence + warm-start data.
//
// _prevPairImpulses: Map<pairKey, {j, nx, ny}> — the previous
// substep's converged normal impulse for each pair, plus the normal
// direction it was applied along. Used to:
//   (1) Detect persistent contacts (Map.has(key) → wasPersistent=true,
//       which suppresses restitution bounce on resting contacts).
//   (2) Warm-start the SI velocity solver: seed each contact's
//       `c.normalImpulse` with the previous substep's value so the
//       solver converges in 1-2 iterations rather than 8+ on stacks.
//
// Storing the normal direction alongside the impulse magnitude lets
// us detect wrap-boundary sign flips. If applyBoundary teleported a
// body across an edge between substeps, the freshly-computed normal
// may point opposite to the stored one — warm-starting with that
// reversed direction would inject a spurious impulse. The
// processCollisionPair guard does a dot-product sign check and
// discards stale warm-start data when the normal flipped.
const _prevPairImpulses = new Map();
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
//
// Phase 1 WebGPU integration (architect decision A2.α): optional 3rd arg
// `injectedAccels` — a Float32Array of interleaved (ax_0, ay_0, ax_1, ay_1, …)
// from K1's `outputBuf` staging readback. When supplied, step B copies it
// into `_scratch` instead of running the CPU O(N²) computeAccelerations.
// `undefined` keeps the CPU path bit-identical to main (F1 acceptance).
export function stepPBD(entities, dt, injectedAccels) {
  const n = entities.length;
  if (n === 0 || dt === 0) return;
  ensureScratch(n);

  // Cache dt for processCollisionPair's pre-kick velocity calculation
  // (used by the wasPersistent gate, NOT by any deleted refund code).
  _currentSimDt = dt;

  // ── A. Reset pseudo-velocity accumulator (for step G) ──────────
  for (let i = 0; i < n; i++) {
    entities[i]._pvx = 0;
    entities[i]._pvy = 0;
  }

  // ── B. Compute accelerations at current positions ──────────────
  if (injectedAccels !== undefined) {
    for (let i = 0; i < n; i++) {
      _scratch[i].ax = injectedAccels[i * 2];
      _scratch[i].ay = injectedAccels[i * 2 + 1];
    }
  } else {
    computeAccelerations(entities, _scratch);
  }

  // ── C. Gravity kick: v += a · dt ───────────────────────────────
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned || e.absorbing) continue;
    const ax = _scratch[i].ax;
    const ay = _scratch[i].ay;
    e.vx += ax * dt;
    e.vy += ay * dt;
    // Cache a for downstream readers (prediction tooling).
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

  // ── F. Sequential-Impulses velocity solver (Box2D-style) ───────
  // 8-iter Gauss-Seidel with warm-starting + accumulated-impulse
  // clamp ≥ 0. Convergence on resting/stacked contacts comes from
  // the previous substep's seeded normalImpulse.
  _solveContactVelocities(_contactCount);

  // ── F'. Persist this substep's impulses for next substep's warm-start
  _rebuildPrevPairImpulses();

  // ── G. Pseudo-velocity NGS position solver (Box2D-style) ───────
  // 3-iter Gauss-Seidel with Baumgarte slop + maxCorrection cap.
  // Writes to body._pvx/_pvy; integrates into body.x/y at the end
  // without touching real velocity. This is the key "split impulses"
  // pattern that prevents position correction from injecting phantom
  // PE into the gravity field — the failure mode the old _dxGrav
  // energy-refund pass was patching after the fact.
  _solvePositionConstraints(_contactCount, entities, dt);

  // ── I. Pinned bodies hard-reset ────────────────────────────────
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned) { e.vx = 0; e.vy = 0; e.ax = 0; e.ay = 0; }
  }
}

// ─── Position solver (Box2D-style pseudo-velocity NGS) ────────────
// Splits position correction from real-velocity dynamics. Instead of
// writing penetration corrections directly to body positions (which
// would inject phantom PE into the gravity field), we accumulate them
// into a per-body pseudo-velocity (_pvx, _pvy) and integrate that
// into position at the end. Real velocity vx/vy is NEVER touched by
// this pass — that's the key Box2D 2.x design choice that prevents
// "position correction pumps KE into gravity orbit" failure mode the
// old _dxGrav refund pass was trying to compensate for.
//
// Baumgarte slop: don't push to zero overlap. Leave a tiny gap
// (LINEAR_SLOP_FRAC × rSum) so resting contacts under continuous
// gravity don't oscillate around exact-touch (which produces jitter
// near machine ε).
//
// maxCorrection: cap each iteration's per-contact correction to a
// fraction of rSum. Prevents teleporting when a fast-moving body
// penetrates deeply — corrections spread over multiple frames instead.
//
// Reference: Catto's "Iterative Dynamics with Temporal Coherence"
// (GDC 2005); Planck.js b2ContactSolver::SolvePositionConstraints.
const POS_ITERATIONS = 3;
const LINEAR_SLOP_FRAC = 0.005;       // unitless × rSum
const MAX_CORRECTION_FRAC = 0.2;      // unitless × rSum

function _solvePositionConstraints(count, entities, dt) {
  if (count === 0) return;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;
  const invDt = 1 / dt;

  for (let iter = 0; iter < POS_ITERATIONS; iter++) {
    for (let k = 0; k < count; k++) {
      const c = _contacts[k];
      const a = c.a;
      const b = c.b;
      if (a.absorbing || b.absorbing) continue;

      // Effective positions = current real x + accumulated pseudo
      // displacement (pvx*dt). Recompute geometry against the
      // effective positions so iterations converge against what the
      // pseudo-velocity has "already done".
      const axEff = a.x + a._pvx * dt;
      const ayEff = a.y + a._pvy * dt;
      const bxEff = b.x + b._pvx * dt;
      const byEff = b.y + b._pvy * dt;

      let dx = bxEff - axEff;
      let dy = byEff - ayEff;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
      const dist2 = dx * dx + dy * dy;
      if (dist2 < 1e-12) continue;
      const dist = Math.sqrt(dist2);
      const overlap = c.rSum - dist;
      const linearSlop = LINEAR_SLOP_FRAC * c.rSum;
      // Apply Baumgarte slop: only correct overlap that exceeds slop.
      // Negative result = no correction needed (already separated, or
      // separated within the slop tolerance).
      const want = overlap - linearSlop;
      if (want <= 0) continue;
      const maxCorrection = MAX_CORRECTION_FRAC * c.rSum;
      const correction = want > maxCorrection ? maxCorrection : want;
      const wA = a.pinned ? 0 : 1 / a.mass;
      const wB = b.pinned ? 0 : 1 / b.mass;
      const wSum = wA + wB;
      if (wSum === 0) continue;
      const nx = dx / dist;
      const ny = dy / dist;
      // Convert positional correction to pseudo-velocity by dividing
      // by dt. End-of-pass integration `x += _pvx * dt` then turns it
      // back into the same position delta — but real velocity is
      // never read or written. The split keeps KE/PE bookkeeping
      // honest in the gravity field.
      const lambdaOverDt = correction / wSum * invDt;
      a._pvx -= lambdaOverDt * nx * wA;
      a._pvy -= lambdaOverDt * ny * wA;
      b._pvx += lambdaOverDt * nx * wB;
      b._pvy += lambdaOverDt * ny * wB;
    }
  }

  // Integrate pseudo-velocity into real position. Real velocity is
  // untouched — this is the whole point of split impulses.
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.pinned || e.absorbing) continue;
    e.x += e._pvx * dt;
    e.y += e._pvy * dt;
  }
}

// ─── Velocity solver (Box2D-style Sequential Impulses) ────────────
// Drives each contact's relative normal velocity to its target:
//   • persistent contact (c.wasPersistent) → 0 (no bounce on resting)
//   • new collision → -e · c.vnApproach (true restitution)
//
// Box2D additions vs the previous 4-iter PBD-style solver:
//   (1) Warm-starting: each contact starts the substep with its
//       previous substep's converged normal impulse already applied
//       to bodies' velocities. Saves 4-6 iterations of convergence
//       work on resting / stable orbital contacts.
//   (2) Accumulated impulse clamping: the running c.normalImpulse
//       is clamped to [0, +∞) (contacts can only push apart, not
//       pull together). Each iteration computes a Δλ = newλ − oldλ
//       and applies that increment, not the raw λ.
//   (3) 8 iterations (vs 4) — Catto's default for Box2D 2.x,
//       sufficient for typical N ≤ 50 stacks.
//
// Reference: Catto's GDC 2006 "Fast and Simple Physics using
// Sequential Impulses"; Planck.js's b2ContactSolver::SolveVelocityConstraints.
//
// Bug fix (2026-05-21): for dense clusters the chain-depth of the contact
// graph exceeds 8 iterations — pressure impulses from the exterior do not
// propagate to the interior within one substep, leaving interior contacts
// to see unresisted gravity for 1-2 substeps. The result: visible body
// overlap and chronic jitter at high G + dense clusters. We linearly ramp
// iteration count by contact count: sparse scenes (orbits, small stacks)
// keep the 8-iteration budget; dense clusters get up to 24. Linear ramp
// avoids the visible "step" hitch a binary gate would cause as the user
// adds the threshold-crossing body.
//
// Chain depth for a 2D hex-packed disc of N bodies ≈ 2·√(N/π). At
// N=200 → depth ≈ 16. At N=300 → depth ≈ 20. 24 iterations covers
// depth 24 with margin. RAMP_LO/RAMP_HI define the linear interval.
const VEL_ITERATIONS_MIN = 8;
const VEL_ITERATIONS_MAX = 24;
const VEL_ITERATIONS_RAMP_LO = 60;    // sparse contact count — keep MIN
const VEL_ITERATIONS_RAMP_HI = 380;   // dense contact count — saturate at MAX

// Warm-start calibration for persistent (resting) contacts.
//
// Bug fix (2026-05-20): on hex-packed clusters straddling the Y wrap edge,
// vertical jitter would build up substep-after-substep ("逐级递加"). Root
// cause: the warm-start blindly applied the previous substep's full
// converged impulse `j_prev`. For a resting contact `j_prev` was sized to
// resist a large gravity-induced approach velocity, but by the next
// substep the position solver had already separated the pair — so the
// fresh approach velocity is small (only the differential-gravity push of
// one substep). Applying the full `j_prev` to that small approach velocity
// pushed the pair into a SEPARATING state; the accumulated-impulse clamp
// (c.normalImpulse ≥ 0) then zeroed the net impulse for that substep,
// leaving gravity's approach velocity completely unresisted. Position
// solver then took the hit, and gravity re-injected energy unchecked
// substep after substep — the runaway loop. Y-axis was hit harder because
// flat-top hex packing has 4 diagonal contacts per body (large |ny|) vs
// 2 in-row horizontal contacts (ny=0), so Y has ~4× more constraint
// pressure than X.
//
// Fix: for persistent contacts, store a gravity-calibrated warm-start
// (sized to resist exactly the next substep's expected approach velocity
// from differential gravity), with a small floor of j_prev to handle
// perturbations not captured by the gravity estimate. Fresh collisions
// still get the full j_prev so they converge in ≤2 iterations as before.
const WARM_START_PERSIST_FLOOR = 0.10;   // fraction of j_prev kept as floor

function _solveContactVelocities(count) {
  if (count === 0) return;
  // Adaptive iteration count — see VEL_ITERATIONS_* comment block above.
  // One comparison + arithmetic per substep (not per contact); negligible.
  const VELOCITY_ITERATIONS = count <= VEL_ITERATIONS_RAMP_LO
    ? VEL_ITERATIONS_MIN
    : count >= VEL_ITERATIONS_RAMP_HI
      ? VEL_ITERATIONS_MAX
      : VEL_ITERATIONS_MIN + Math.floor(
          (count - VEL_ITERATIONS_RAMP_LO) * (VEL_ITERATIONS_MAX - VEL_ITERATIONS_MIN)
          / (VEL_ITERATIONS_RAMP_HI - VEL_ITERATIONS_RAMP_LO));
  const e = state.elasticRestitution;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5, halfH = H * 0.5;

  // ── Warm-start pass ────────────────────────────────────────────
  // Apply each contact's carried-over impulse (set by
  // processCollisionPair from _prevPairImpulses) to body velocities
  // BEFORE the iteration loop. Uses the stored c.nx,c.ny normal —
  // sign-flip protection was already applied at lookup time, so this
  // is safe even across wrap-boundary teleports.
  for (let k = 0; k < count; k++) {
    const c = _contacts[k];
    if (c.normalImpulse <= 0) continue;
    const a = c.a, b = c.b;
    if (a.absorbing || b.absorbing) { c.normalImpulse = 0; continue; }
    const wA = a.pinned ? 0 : 1 / a.mass;
    const wB = b.pinned ? 0 : 1 / b.mass;
    const j = c.normalImpulse;
    a.vx -= j * c.nx * wA;
    a.vy -= j * c.ny * wA;
    b.vx += j * c.nx * wB;
    b.vy += j * c.ny * wB;
  }

  // ── 8 Gauss-Seidel iterations with accumulated-impulse clamp ──
  for (let iter = 0; iter < VELOCITY_ITERATIONS; iter++) {
    for (let k = 0; k < count; k++) {
      const c = _contacts[k];
      const a = c.a, b = c.b;
      if (a.absorbing || b.absorbing) continue;
      // Recompute normal from current positions (wrap-aware) so the
      // Jacobian matches the body's current geometry, not the stored
      // detection-time normal.
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
      const wA = a.pinned ? 0 : 1 / a.mass;
      const wB = b.pinned ? 0 : 1 / b.mass;
      const wSum = wA + wB;
      if (wSum === 0) continue;
      // Accumulated-impulse clamp (the key Box2D trick): the total
      // c.normalImpulse must be ≥ 0 (push-only constraint). Each iter
      // computes a desired increment lambdaDesired = (target − vn)/wSum,
      // tentatively adds it to c.normalImpulse, clamps the SUM, then
      // applies only the actual difference dLambda to velocities.
      const lambdaDesired = (vnTarget - vn) / wSum;
      const oldImpulse = c.normalImpulse;
      const newImpulse = oldImpulse + lambdaDesired > 0 ? oldImpulse + lambdaDesired : 0;
      const dLambda = newImpulse - oldImpulse;
      if (dLambda === 0) continue;
      c.normalImpulse = newImpulse;
      a.vx -= dLambda * nx * wA;
      a.vy -= dLambda * ny * wA;
      b.vx += dLambda * nx * wB;
      b.vy += dLambda * ny * wB;
    }
  }
}

// (Legacy `_applyRestitution` removed; its single-pass impulse on
// post-Δx/dt velocities both fired spurious bounces on orbital contacts
// (vnApproach captured at detection sampled a sub-pixel radial component
// of tangential motion) and stripped real radial velocity needed for
// curved trajectories. Both responsibilities are now subsumed by
// `_solveContactVelocities`, which uses iterative impulses and the
// `PRE_KICK_APPROACH_THRESHOLD`-gated `wasPersistent` flag to skip
// restitution on resting contacts.)

// (Legacy `stepVerlet` removed by the PBD refactor — its position+velocity
// pipeline is incompatible with the contact-constraint energy semantics
// PBD enforces. `stepPBD` above is the production step.)

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
    // resolved later by _solvePositionConstraints inside stepPBD. The dx/dy
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
      c = { a: null, b: null, rSum: 0, dist: 0, nx: 0, ny: 0,
            vnApproach: 0, wasPersistent: false, normalImpulse: 0 };
      _contacts[_contactCount] = c;
    }
    const dist = Math.sqrt(dist2);
    c.a = a;
    c.b = b;
    c.rSum = rSum;
    c.dist = dist;   // used by _rebuildPrevPairImpulses for pairwise Plummer r²
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
    // Warm-start lookup + persistence detection.
    const key = _pairKey(a, b);
    const prev = _prevPairImpulses.get(key);
    let wasPersistent = prev !== undefined;
    if (!wasPersistent) {
      // Pair wasn't in last substep's contact list, but it might still
      // be a "resting contact that was just brought into existence by
      // the kick". Use the PRE-KICK relative normal velocity as the
      // discriminator: if the bodies weren't approaching pre-kick, the
      // contact is geometrically persistent (the kick just pushed them
      // together in this substep's predict), not a fresh impact.
      const dt = _currentSimDt;
      const preRvx = (b.vx - b.ax * dt) - (a.vx - a.ax * dt);
      const preRvy = (b.vy - b.ay * dt) - (a.vy - a.ay * dt);
      const preVn = preRvx * nx + preRvy * ny;
      if (preVn > PRE_KICK_APPROACH_THRESHOLD) wasPersistent = true;
    }
    c.wasPersistent = wasPersistent;
    // Warm-start the velocity solver's accumulated impulse from the
    // previous substep — but ONLY if the normal direction hasn't flipped
    // (wrap-boundary teleport case). A negative dot product of the
    // stored normal with the fresh one means the pair geometry rotated
    // ~180°: stale warm-start would push in the wrong direction. In
    // that case discard the carryover and start fresh.
    if (prev !== undefined) {
      const sameSide = prev.nx * nx + prev.ny * ny;
      c.normalImpulse = (sameSide >= 0) ? prev.j : 0;
    } else {
      c.normalImpulse = 0;
    }
    _contactCount++;
  }
}

export function handleCollisions(entities) {
  const n = entities.length;

  // PBD refactor: clear the contact list before this substep's broadphase
  // walk. processCollisionPair will push planet-planet pairs onto it.
  // _solvePositionConstraints (run later inside stepPBD) drains it.
  _contactCount = 0;

  // V8.3: large-N path uses wrap-aware spatial hash for O(N·k) broadphase.
  // The hash is rebuilt HERE (per-substep), not in prepareFrame (V8.2's
  // per-frame approach caused Type-1 misses: bodies that drifted across
  // cell boundaries during the gravity-kick + predict steps would still be
  // registered in their old bucket, invisible to queries targeting their
  // new neighborhood — observable as persistent overlaps at the Y wrap
  // edge on dense BH-active clusters). After applyBoundary teleport at the
  // end of the previous substep, positions are baked in here and the hash
  // matches reality. BH tree stays per-frame (continuous force, tolerant
  // of drift) — see prepareFrame.
  if (n >= state.bhThreshold) {
    buildSpatialHash(entities);
    forEachCollisionPair(entities, processCollisionPair);
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
      // V8.2: dispatch through shared helper so both small-N (direct) and
      // large-N (spatial-hash) paths run identical collision logic. The
      // dispatch boundary is state.bhThreshold (default 256 — see state.js
      // for the rationale and how to tune).
      processCollisionPair(a, b, dx, dy);
    }
  }
}

// Rebuild _prevPairImpulses from the just-populated _contacts AFTER
// the velocity solver has converged for this substep. Called from
// stepPBD (not handleCollisions, because handleCollisions runs BEFORE
// the solver; we need post-solver impulse values).
//
// What we store depends on whether the contact was persistent or fresh:
//
//   FRESH (c.wasPersistent === false): the contact came into existence
//     this substep. Store the full c.normalImpulse so next substep's
//     warm-start can quickly re-converge if the pair stays in contact.
//
//   PERSISTENT (c.wasPersistent === true): the contact has been resting
//     across substeps. The velocity solver has already driven vN to 0
//     this substep; the only approach velocity NEXT substep will come
//     from PAIRWISE differential gravity along the normal. Sign
//     convention: c.nx, c.ny point from a toward b (set in
//     processCollisionPair as dx/dist where dx = b.x - a.x).
//
//     Previous formula (2026-05-20, commit d3d3e7d) used the NET
//     per-body cached acceleration: dvn_grav = (b.a - a.a) · n̂ · dt.
//     That works for isolated pairs but is WRONG for cluster interiors,
//     where opposing neighbor forces make b.a ≈ a.a ≈ 0 (forces
//     cancel) so dvn_grav ≈ 0 even though the A-B pairwise approach
//     impulse is significant. The under-calibrated warm-start let
//     interior contacts repeatedly under-converge under high G,
//     accumulating microscopic overlap until visible.
//
//     Fix (2026-05-21, this commit): compute the pairwise gravity
//     directly from the two bodies' charges and masses at the contact
//     distance (Plummer-softened: r² + rSum² ≈ c.dist² + c.rSum²).
//     This isolates the A-B pairwise force from the cluster's net
//     gravitational context. Signs follow the asymmetric charge model
//     (A receives force from B iff b.charge ≠ 0, magnitude scaled by
//     b.charge — and vice versa). Stored impulse is max(jGrav,
//     c.normalImpulse * 10%) — the 10% floor handles mixed-charge
//     scenarios where dvn_pair = 0 (e.g., A=+1, B=-1, equal mass:
//     both translate together, no relative motion — but the velocity
//     solver still needs a starting impulse for the persistent
//     constraint).
//
// Keys with c.normalImpulse ≤ 0 are skipped to prevent unbounded Map
// growth from transient brushing contacts that didn't accumulate
// meaningful impulse. Normal direction is stored alongside the
// magnitude for wrap-boundary sign-flip detection in next substep's
// processCollisionPair.
function _rebuildPrevPairImpulses() {
  _prevPairImpulses.clear();
  for (let k = 0; k < _contactCount; k++) {
    const c = _contacts[k];
    if (c.normalImpulse <= 0) continue;

    let jStore = c.normalImpulse;                  // fresh-collision default

    if (c.wasPersistent) {
      const a = c.a, b = c.b;
      const wA = a.pinned ? 0 : 1 / a.mass;
      const wB = b.pinned ? 0 : 1 / b.mass;
      const wSum = wA + wB;
      if (wSum > 0) {
        // Pairwise Plummer-softened r² at the actual contact distance.
        // Detection-time c.dist captures where the pair is right now;
        // c.rSum is the unsoftened separation that defines minR in the
        // computeAccelerations Plummer term. Their squares sum to the
        // softened r² used in the gravity formula. Using c.dist (rather
        // than rSum) avoids the overshoot risk if the position solver
        // has separated the bodies slightly past rSum since detection.
        const r2pw = c.dist * c.dist + c.rSum * c.rSum;
        // Pairwise acceleration magnitudes following the asymmetric
        // charge force law (see file header). Signs are carried by
        // charge: +1 attractive, −1 repulsive. accel_A_from_B is the
        // n̂-component of A's gravitational acceleration due to B
        // (= b.charge·G·m_B/r² since A is pulled in +n̂ direction when
        // b.charge=+1, in −n̂ when b.charge=−1). accel_B_from_A is the
        // (−n̂)-component of B's gravitational acceleration due to A,
        // which equals a.charge·G·m_A/r² (B is pulled in −n̂ direction
        // when a.charge=+1, +n̂ when a.charge=−1). Charge-zero source
        // contributes no force — the early return preserves that.
        // Pinned bodies have b.charge / a.charge intact; only the wA/wB
        // weight gates the impulse distribution downstream.
        const G = state.G;
        const accel_A_from_B = b.charge !== 0 ? b.charge * G * b.mass / r2pw : 0;
        const accel_B_from_A = a.charge !== 0 ? a.charge * G * a.mass / r2pw : 0;
        // Sum = relative approach acceleration along n̂. Positive when
        // bodies converge. Mixed-charge (+1,−1) at equal mass: both
        // bodies translate in −n̂ at the same rate → 0 (correct).
        const dvn_pair = (accel_A_from_B + accel_B_from_A) * _currentSimDt;
        const jGrav = dvn_pair > 0 ? dvn_pair / wSum : 0;
        jStore = Math.max(jGrav, c.normalImpulse * WARM_START_PERSIST_FLOOR);
      }
      // wSum === 0: both bodies pinned. _solveContactVelocities also
      // skips (wSum === 0 → `continue`), so the stored impulse is never
      // applied next substep — benign no-op. Keeping c.normalImpulse
      // matches what would be written if we lifted the wSum guard.
    }

    _prevPairImpulses.set(_pairKey(c.a, c.b), {
      j: jStore, nx: c.nx, ny: c.ny,
    });
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
// correction is now handled by _solvePositionConstraints; restitution is
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
