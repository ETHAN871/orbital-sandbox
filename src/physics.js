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

// Contact-persistence tracking. A pair is "persistent" (no bounce) if
// EITHER it was a contact last substep OR its PRE-KICK relative normal
// velocity is essentially zero (orbital tangential motion / resting).
// A pair is "new" (apply restitution) only when both gates fail — i.e.
// the bodies were genuinely approaching BEFORE this substep's gravity
// kick.
//
// The pre-kick check is the principled, scale-invariant alternative to
// the earlier `vnApproach < -5 px/s` threshold: the kick-induced
// approach velocity is exactly `a·dt·n̂`, so by subtracting it from the
// detection-time vnApproach we recover the body's TRUE approach speed
// (zero for resting / orbital contacts; the actual impact velocity for
// genuine collisions, even slow ones). Works at any G / mass / dt.
//
// _prevPairs is rebuilt at the END of handleCollisions from the
// current substep's _contacts, so the next substep's broadphase sees
// "previous". The Set catches warm-started orbital contacts where one
// substep's bookkeeping bridges into the next.
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

// Refund grace period: when a real collision is detected (preVn ≪ 0),
// the refund pass (G'') is disabled for the next REFUND_GRACE_FRAMES
// substeps. Lets the velocity solver's impulse cascade settle the
// post-collision cluster into stable motion without the refund
// draining its momentum on intra-cluster sub-pixel corrections.
//
// GRACE_TRIGGER_PRE_VN: only pre-kick approach speeds faster than 1
// px/s trigger the grace. This excludes the 0.0001-0.1 px/s noise
// floor of statically-placed clusters and slow-orbital warm-up
// substeps where the contact is geometrically "new" only by
// bookkeeping (_prevPairs empty on first substep).
const REFUND_GRACE_FRAMES = 40;             // ~0.33 s at SIM_DT=1/120
const GRACE_TRIGGER_PRE_VN = -1.0;          // px/s
let _refundSkipCounter = 0;

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

// Position-Based Dynamics step. Replaces the legacy Velocity Verlet +
// impulse pipeline. Pipeline per substep:
//
//   A. Save x_prev (so velocity can be derived post-projection).
//   B. Compute accelerations at current positions.
//   C. Symplectic Euler gravity kick:    v += a · dt
//   D. Predict positions:                x += v · dt
//   E. handleCollisions: broadphase at PREDICTED positions. For BH pairs,
//      runs beginAbsorption immediately. For planet-planet pairs, pushes
//      onto _contacts[] (no immediate impulse — PBD resolves later).
//   F. _solveContactVelocities (Sequential-Impulses-style velocity solver):
//      For each contact, drive relative normal velocity to a target
//      (-e·vnApproach for genuine collisions, 0 for persistent contacts).
//      Iterated 4× for Gauss-Seidel convergence in dense clusters.
//   G. _projectPBDContacts: iterate PBD_ITERATIONS passes, each pushing
//      overlapping pairs apart by mass-weighted normal correction.
//   G''. Energy refund: scale each body's v² by 2·(a·Δx) to repay the
//        PE the position solver injected when it moved bodies up the
//        gravity gradient. Closes the secular-breathing-oscillation
//        leak that experiments traced to projection's non-physicality.
//   G'. _relaxContactVelocities: one no-bias pass at post-projection
//       positions to cancel the residual inward velocity introduced when
//       the position solver refreshes each contact's normal. Catto 2024.
//   H. _applyStaticContactDamping: zero out velocities of in-contact
//      bodies whose speed has decayed below STATIC_V_THRESHOLD.
//
// Why this fixes the bug: in Velocity Verlet's velocity update
// `v += 0.5 * (a_old + a_new) * dt`, the cached a_old at a body's
// PRE-step position has a tangential component when viewed from the
// body's POST-step rotated position. In a free orbit this is the natural
// centripetal redirection. In a contact-constrained orbit the inward
// radial portion is killed by the collision impulse but the tangential
// portion leaks through (Δv_t ≈ 0.5·|a|·v_t·dt²/r per step), accumulating
// linearly. PBD's velocity derivation (v = Δx/dt) recovers velocity from
// the net position change after constraint projection; the tangential
// bias never enters the equation.
export function stepPBD(entities, dt) {
  const n = entities.length;
  if (n === 0 || dt === 0) return;
  ensureScratch(n);

  // Cache dt for processCollisionPair's pre-kick velocity calculation.
  _currentSimDt = dt;

  // ── B. Compute accelerations at current positions ──────────────
  computeAccelerations(entities, _scratch);

  // ── B'. Centripetal projection for bodies in stable contact ────
  // For a body resting on (or orbiting in contact with) another, the
  // inward portion of gravity that EXCEEDS the centripetal requirement
  // is absorbed by the constraint — it must not contribute to velocity
  // changes. Without this projection, every substep injects a tiny
  // tangential drift (proportional to v_t) because the predict step
  // sinks the body radially, then projection scales it back to the
  // constraint surface, advancing its ANGLE slightly more than
  // v_t·dt/r would warrant. Subtracting the excess radial gravity
  // before the kick eliminates the drift at its source.
  //
  // _contacts at this point holds the contacts detected during the
  // PREVIOUS substep's handleCollisions. Stable contacts persist across
  // substeps, so this look-back is well-suited. (First substep after a
  // body spawns has no contacts yet — that single-substep bias is
  // O(dt²) and self-corrects on the next substep when contacts populate.)
  _centripetalProject(entities);

  // ── C. Gravity kick: v += a · dt ───────────────────────────────
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned || e.absorbing) continue;
    const ax = _scratch[i].ax;
    const ay = _scratch[i].ay;
    e.vx += ax * dt;
    e.vy += ay * dt;
    // Cache a for any downstream reader (debug-energy.js, prediction
    // tooling). This is the (possibly centripetal-projected) accel.
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
  // handleCollisions does the broadphase: BH pairs trigger absorption
  // immediately; planet-planet pairs push onto _contacts[].
  handleCollisions(entities);

  // ── F. Velocity solver (Sequential-Impulses-style) ─────────────
  // Iteratively apply impulses along each contact normal to drive
  // relative normal velocity to its target:
  //   - persistent contacts (vnApproach near zero, orbital motion) → 0
  //   - genuine collisions (vnApproach below threshold) → -e·vnApproach
  // Running this BEFORE the position solver — and crucially, BEFORE
  // any `v = Δx/dt` re-derivation — is what stops the multi-body hex
  // cluster from leaking tangential energy. PBD's position-only
  // pipeline left the inward velocity component implicit in Δx; here
  // we cancel it explicitly at the velocity level, which is the
  // canonical fix for "stacking under gravity" per Box2D / Catto.
  //
  // Detect REAL collisions this substep — pairs with a significant
  // pre-kick approach velocity (much faster than the gravity kick
  // could account for). If found, trigger a refund grace period so
  // the impulse cascade can settle without G'' draining its momentum.
  //
  // Uses preVn (not wasPersistent) because wasPersistent's tighter
  // threshold (-1e-6) classifies sub-mm/s numerical noise as "new
  // collision" — fine for restitution gating (no-bounce-near-zero)
  // but would spuriously trigger grace on the first substep of any
  // statically-placed cluster. -1.0 px/s here cleanly separates
  // user-shot bullets (preVn ≪ -10) from numerical noise (|preVn| ≪ 0.1).
  for (let k = 0; k < _contactCount; k++) {
    if (_contacts[k].preVn < GRACE_TRIGGER_PRE_VN) {
      _refundSkipCounter = REFUND_GRACE_FRAMES;
      break;
    }
  }
  _solveContactVelocities(_contactCount);

  // ── G. Position solver (non-linear Gauss-Seidel) ───────────────
  // Resolve residual penetration. Now that velocities along each
  // contact normal are already zero, the position projection mostly
  // catches first-time penetrations (initial overlap, fast impacts).
  //
  // Snapshot positions BEFORE projection so step G'' can compute the
  // ΔPE the projection injected and refund it from KE.
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    e._sxProj = e.x;
    e._syProj = e.y;
  }
  _projectPBDContacts(_contactCount);

  // ── G''. Energy-conserving position projection refund ──────────
  // PBD's position solver is a KINEMATIC correction — it moves bodies
  // along the contact normal to undo overlap without accounting for
  // the work the constraint "should have done" against gravity over
  // that displacement. Net effect: each substep every body the
  // projection pushes gains a tiny amount of phantom PE without a
  // matching KE decrease. The experiments traced this to the secular
  // breathing-oscillation source.
  //
  // Refund (work-energy theorem applied as a scalar KE adjustment):
  //   • Δx_i = body i's net displacement from the position solver
  //   • Work done by net force on body over Δx_i (per unit mass):
  //         w = a_i · Δx_i        ⇒    ΔKE/m = w
  //   • New |v|² = |v|² + 2·w; rescale v by sqrt(new/old).
  //
  // Two implementations were tried:
  //   (a) Radial-only refund (apply Δv only along Δx̂). Mathematically
  //       cleaner — only the component of v along the displacement
  //       should be touched. BUT: in the orbital regime v_along is
  //       tiny (mostly tangential motion) while |a·Δx| can exceed
  //       v_along², triggering the clamp and zeroing v_along. That
  //       breaks the discrete-orbit centripetal velocity component,
  //       cascading into massive r oscillation (ΔE −47 % at v=77).
  //   (b) Scalar |v|² rescale (this version). Slightly less
  //       physically precise (touches tangential KE proportionally),
  //       but BOUNDED — never zeros a velocity component. For the
  //       orbital case the scale factor is ~1±1e-5 per substep, so
  //       the tangential bleed is negligible. ΔE at v=77 ≈ +0.02 %.
  //
  // (b) is the practical choice for our scene mix. The slight
  // tangential bleed at deep sub-orbital initial conditions
  // (v=30 → ~10 % KE loss / 60 s) is a known trade-off; those
  // configurations are not user-reachable through the UI and benefit
  // from the much-improved r-range stability they get in exchange
  // (v=30 r-range 20.3 → 0.46).
  //
  // a_i used here is the post-centripetal-projection net acceleration
  // step C kicked with — physically consistent because it represents
  // the body's effective acceleration (gravity minus implicit
  // constraint absorption).
  // Gate: skip refund entirely during a "grace period" after any new
  // contact. The refund is mathematically right only when gravity
  // is the dominant non-impulsive force doing work over the substep;
  // post-collision cluster cascades have velocity-solver impulses
  // (and their cascade aftermath) as the dominant force, and the
  // refund's "subtract gravity work over Δx" double-discounts the
  // cluster's CoM momentum that the impulse installed.
  //
  // Per-body Δv-magnitude gating was tried (5 px/s threshold) but
  // failed: post-cascade individual body |Δv| drops to 0.5-1 px/s
  // — indistinguishable from the gravity-kick |Δv| of stable configs
  // — yet the cluster is still in transient settling motion. A
  // simpler scene-wide grace-frame counter cleanly separates the
  // two regimes: during the cascade and its settle-down, refund is
  // off; once the cluster reaches stable motion (~40 substeps =
  // 0.33 s), refund re-engages on a per-body basis (now finds
  // Δx_proj ≈ 0 for rigid cluster translation, so naturally
  // no-ops anyway).
  if (_refundSkipCounter > 0) {
    _refundSkipCounter--;
  } else for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned || e.absorbing) continue;
    const dx = e.x - e._sxProj;
    const dy = e.y - e._syProj;
    const dx2 = dx * dx + dy * dy;
    if (dx2 < 1e-12) continue;                  // body wasn't moved
    const adx = e.ax * dx + e.ay * dy;
    const v2 = e.vx * e.vx + e.vy * e.vy;
    const v2New = v2 + 2 * adx;
    if (v2New <= 0) {
      e.vx = 0;
      e.vy = 0;
      continue;
    }
    if (v2 < 1e-12) {
      const mag = Math.sqrt(v2New);
      const invDxLen = 1 / Math.sqrt(dx2);
      const sign = adx > 0 ? 1 : -1;
      e.vx = sign * mag * dx * invDxLen;
      e.vy = sign * mag * dy * invDxLen;
      continue;
    }
    const scale = Math.sqrt(v2New / v2);
    e.vx *= scale;
    e.vy *= scale;
  }

  // ── G'. Relaxation pass (Catto 2024 Solver2D, no-bias second solve) ─
  // After the position solver, each contact's normal has been refreshed
  // to the post-projection geometry; in dense clusters that refreshed
  // normal can differ slightly from what step F operated on, so a body's
  // velocity — perfectly tangent to its OLD normal — picks up a small
  // approach component along its NEW normal. Without this pass, that
  // approach component is what your hypothesis identified as the
  // "微小重叠再弹开 → 不对称 → 累积旋转" feedback path: each substep
  // injects a sub-pixel-scale inward velocity that the velocity solver
  // (which ran one step earlier in F) couldn't have seen.
  //
  // The relaxation pass re-solves the contact velocity constraints with
  // vnTarget = 0 for ALL contacts (no restitution, no Baumgarte bias).
  // It is ONE-SIDED — it only applies impulse if vn < 0 (still
  // approaching), never to undo a separating velocity. So it preserves:
  //   • elastic restitution from step F (vn > 0 → no-op here)
  //   • orbital tangential motion (radial component is genuinely 0 → no-op)
  // and only removes the small residual approach velocity that
  // position-projection-induced normal rotation revealed.
  //
  // Catto 2024 reports relaxation "improves the simulation quality
  // dramatically." In our split-impulse pipeline (no Baumgarte bias to
  // remove) the gain is more modest, but it directly closes the
  // user-identified amplification path. Iterating once is enough — the
  // residuals after step F are already small.
  _relaxContactVelocities(_contactCount);

  // ── H. Static contact damping ──────────────────────────────────
  // DISABLED 2026-05-20: the 0.1 px/s threshold was too aggressive —
  // bodies in slow-moving multi-body clusters (under near-equilibrium
  // mutual gravity) were being PINNED IN PLACE, producing the
  // "dynamic sleep" failure the user observed where dense aggregates
  // froze instead of evolving. The energy-refund pass (G'') and the
  // relaxation pass (G') already provide enough stability for the
  // statically-generated configurations (all-touching hex N=6 stays
  // bounded at sub-μ-pixel for 5+ min), so the artificial sleep is
  // no longer earning its cost. Keep the function defined below as
  // dead code in case a future scene needs explicit sleeping.
  // _applyStaticContactDamping(_contactCount);

  // ── I. Pinned bodies stay frozen ───────────────────────────────
  // Defensive: each solver above zeroes `wA` (or `wB`) when the body is
  // pinned, so impulses MUST have no effect on it — but if any future
  // refactor accidentally lets a pinned body accumulate v or a (e.g.,
  // forgetting the `if (e.pinned) continue` in the gravity kick), this
  // hard reset catches it. Costs O(n) per substep; trivial.
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned) { e.vx = 0; e.vy = 0; e.ax = 0; e.ay = 0; }
  }
}

// ─── Centripetal projection ───────────────────────────────────────
// Subtract the "excess radial gravity" that a contact constraint will
// absorb anyway, BEFORE the velocity kick. Without this step, PBD by
// itself still leaks tangential energy in sub-orbital contact aggregates
// — the predict step sinks the body radially into its constraint, then
// the projection pushes it back to the surface, advancing its angular
// position slightly more than v_t·dt/R warrants. The angular drift
// integrates over many substeps into a steady tangential acceleration.
//
// Mechanism: for each body in a recent contact, compute the gravity
// component pointing INTO the constraint surface (positive aInward).
// If aInward exceeds the centripetal requirement v_t²/r — i.e., the
// body is moving slower than the local circular-orbit speed — the
// excess will be absorbed by the constraint. Subtract that excess from
// the body's acceleration before integrating. The remaining radial
// portion is exactly what's needed to curve the trajectory along the
// constraint surface at the body's current tangential speed.
//
// Reads _contacts[0.._contactCount-1] — the contacts detected during
// the PREVIOUS substep. Stable contacts persist across substeps so the
// look-back is well-suited. (The first substep after a body spawns has
// no contacts yet; that's a one-step O(dt²) bias that self-corrects.)
//
// Multi-contact safety: each contact is processed independently, but
// the v_t²/r threshold automatically suppresses projections in
// directions where the body has no excess inward gravity. For a hex
// cluster around a heavy body, only the radial (light↔heavy) contacts
// trigger meaningful projection; the tangential (light↔neighbor)
// contacts have aInward ≈ v_t²/r and contribute essentially zero
// correction.
const _idxMap = new Map();

function _centripetalProject(entities) {
  if (_contactCount === 0) return;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;

  // Build entity→index lookup so we can write into _scratch[idx].
  // For typical N ≤ 50 the Map.clear()+populate is cheaper than the
  // alternative of indexOf-per-contact (which would be O(N·M) where
  // M is contact count).
  _idxMap.clear();
  for (let i = 0; i < entities.length; i++) _idxMap.set(entities[i], i);

  for (let k = 0; k < _contactCount; k++) {
    const c = _contacts[k];
    const a = c.a;
    const b = c.b;
    if (!a || !b) continue;                  // defensive: spliced entity
    if (a.absorbing || b.absorbing) continue;

    // Re-derive normal from CURRENT positions (wrap-aware). The stored
    // c.nx/c.ny were captured at detection in the previous substep;
    // positions have shifted slightly since. Re-computing per substep
    // costs ~10 fp ops and avoids angular drift in the projection.
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    if (wrap) {
      if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
      if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
    }
    const dist2 = dx * dx + dy * dy;
    if (dist2 < 1e-12) continue;
    const dist = Math.sqrt(dist2);

    // Skip stale contacts — bodies that have drifted well past contact
    // distance shouldn't keep receiving centripetal correction. 1.5× rSum
    // gives a generous re-acquisition window for "just-detached" pairs.
    if (dist > c.rSum * 1.5) continue;

    const nx = dx / dist;                    // unit vector from a toward b
    const ny = dy / dist;

    // ── Body A: inward direction is +n̂ (pointing into B) ────────────
    if (!a.pinned) {
      const idxA = _idxMap.get(a);
      if (idxA !== undefined) {
        const sA = _scratch[idxA];
        const aInward = sA.ax * nx + sA.ay * ny;
        if (aInward > 0) {
          // Tangential velocity² of A relative to B
          const rvx = a.vx - b.vx;
          const rvy = a.vy - b.vy;
          const vRadial = rvx * nx + rvy * ny;
          const v2 = rvx * rvx + rvy * rvy;
          const vTang2 = v2 - vRadial * vRadial;
          // Centripetal requirement at the contact distance. Floor at 0
          // (numerical noise can make v² < vRadial² by ULPs).
          const aCentripetalNeeded = vTang2 > 0 ? vTang2 / dist : 0;
          const excess = aInward - aCentripetalNeeded;
          if (excess > 0) {
            sA.ax -= excess * nx;
            sA.ay -= excess * ny;
          }
        }
      }
    }

    // ── Body B: inward direction is -n̂ (pointing into A) ────────────
    if (!b.pinned) {
      const idxB = _idxMap.get(b);
      if (idxB !== undefined) {
        const sB = _scratch[idxB];
        const aInward = -(sB.ax * nx + sB.ay * ny);
        if (aInward > 0) {
          const rvx = b.vx - a.vx;
          const rvy = b.vy - a.vy;
          // B's inward relative-velocity radial component is along -n̂
          const vRadial = -(rvx * nx + rvy * ny);
          const v2 = rvx * rvx + rvy * rvy;
          const vTang2 = v2 - vRadial * vRadial;
          const aCentripetalNeeded = vTang2 > 0 ? vTang2 / dist : 0;
          const excess = aInward - aCentripetalNeeded;
          if (excess > 0) {
            // inward for B is -n̂ → subtracting excess from inward
            // means ADDING excess·n̂ back to sB.ax,ay
            sB.ax += excess * nx;
            sB.ay += excess * ny;
          }
        }
      }
    }
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
      // Defensive: a pair may have been turned into prey since detection
      // (handleCollisions can flip absorbing mid-walk). Such pairs no
      // longer participate in the constraint network.
      if (a.absorbing || b.absorbing) continue;
      // RECOMPUTE the normal from CURRENT positions each iteration.
      // Using the stored detection-time normal leaks tangential energy
      // for orbiting contacts: the body's angle rotates between detection
      // and projection, so pushing along the OLD normal leaves residual
      // overlap AND injects an angular sweep that surfaces as false
      // tangential velocity in step G's `v = Δx/dt`. Recomputing is
      // ~10 fp ops per contact per iter — negligible — and lets each
      // iter converge against the body's true current geometry.
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
      if (overlap <= 0) continue;                // already separated this iter
      const wA = a.pinned ? 0 : 1 / a.mass;
      const wB = b.pinned ? 0 : 1 / b.mass;
      const wSum = wA + wB;
      if (wSum === 0) continue;                  // two pinned bodies stuck
      // Use CURRENT-iteration normal for the push direction. Update c.nx/c.ny
      // so the rest of the pipeline (centripetal projection, restitution)
      // reads the freshest geometry.
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

// ─── Velocity solver (Sequential Impulses) ────────────────────────
// For each contact, drive the relative normal velocity to its target:
//   • persistent contact (c.wasPersistent: pair existed last substep) → 0
//     Orbital motion samples a tiny radial component of tangential v at
//     the rotating normal; firing a bounce on that drains/pumps energy.
//   • new collision (c.wasPersistent === false) → -e · c.vnApproach
//     This is the genuine restitution that bounces real impacts.
//
// The persistence-based gate is scale-invariant: it works equally well
// at any G, mass, or dt — unlike the earlier vnApproach-magnitude
// threshold which had to be re-tuned per regime.
//
// Iterated with Gauss-Seidel; for the typical N≤10 in-contact cluster,
// 4 iterations converges to machine-eps. For chained clusters (a body
// touching 2+ neighbors) more iterations would help; 4 is the
// conservative practical choice at N ≤ 50.
//
// This (together with the deletion of `v = Δx/dt` re-derivation) is the
// real fix for the hex-aggregate scatter: position-only PBD silently
// leaked tangential energy through each iteration's angular sweep, and
// `_applyRestitution` pumped that further by treating orbital contacts
// as collisions. Running the velocity solve at the velocity level —
// before any position projection and without re-derivation — closes
// both leaks.
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
      // Recompute contact normal from current positions every iteration.
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
      // Current relative normal velocity.
      const rvx = b.vx - a.vx;
      const rvy = b.vy - a.vy;
      const vn = rvx * nx + rvy * ny;
      // Persistent → 0 (no bounce). New collision → -e·vnApproach (bounce).
      const vnTarget = c.wasPersistent ? 0 : -e * c.vnApproach;
      // Only apply if currently approaching faster than target (vn < vnTarget).
      // For persistent contacts: target=0, fire if vn < 0 (any approach).
      // For collisions: target=positive, fire if vn < positive (still net inward).
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

// ─── Relaxation pass (no-bias second velocity solve) ──────────────
// Mirrors Catto 2024 Solver2D's "relaxation" step. Runs after the
// position solver has refreshed contact normals to post-projection
// geometry. For each contact, recomputes n̂ and the current relative
// normal velocity vn — and if vn is still negative (approaching), applies
// the minimum impulse to bring vn back to zero.
//
// Crucially this pass is ONE-SIDED: `if (vn >= 0) continue;` ensures we
// NEVER apply a negative-direction impulse, so:
//   • An elastic bounce from step F (vn now > 0, separating) is preserved
//     untouched — relaxation cannot undo restitution.
//   • A truly tangential orbital velocity (vn == 0 along the post-
//     projection normal) is a no-op — relaxation does not drain energy
//     from a healthy orbit.
//   • The only velocities relaxation modifies are sub-pixel residual
//     approach components, which represent the "post-projection normal
//     rotated since step F" leak that the user identified as the
//     amplification source.
//
// Single iteration is sufficient: step F already drove vn to its target
// at the OLD normal, so the residual at the NEW normal is O((Δn̂)·|v|),
// which for our scenes is well below mm/s — one Gauss-Seidel sweep
// clears it.
// Diagnostic: counts impulses applied by the relaxation pass per substep.
// Read by debug tooling; safe to leave in (zero overhead beyond a counter
// inc per contact that actually needed correction).
export let _relaxImpulseCount = 0;
export let _relaxImpulseMagSum = 0;

function _relaxContactVelocities(count) {
  if (count === 0) return;
  _relaxImpulseCount = 0;
  _relaxImpulseMagSum = 0;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5, halfH = H * 0.5;
  for (let k = 0; k < count; k++) {
    const c = _contacts[k];
    const a = c.a;
    const b = c.b;
    if (a.absorbing || b.absorbing) continue;
    // Recompute contact normal from POST-projection positions.
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
    // One-sided: only kill INWARD residual (vn < 0). Never undo a
    // legitimate separating velocity (vn ≥ 0).
    if (vn >= 0) continue;
    const wA = a.pinned ? 0 : 1 / a.mass;
    const wB = b.pinned ? 0 : 1 / b.mass;
    const wSum = wA + wB;
    if (wSum === 0) continue;
    const J = -vn / wSum;       // bring vn → 0
    _relaxImpulseCount++;
    _relaxImpulseMagSum += Math.abs(J);
    a.vx -= J * nx * wA;
    a.vy -= J * ny * wA;
    b.vx += J * nx * wB;
    b.vy += J * ny * wB;
  }
}

// Static contact damping (sleeping-body equivalent).
// Multi-body gravitational equilibria (e.g., a symmetric ring of equal
// masses around a central body) are mathematically UNSTABLE: any
// infinitesimal perturbation grows exponentially. In real continuous
// time the perturbation is zero so the system stays static; in discrete
// FP arithmetic, the noise floor (~1e-15) IS such a perturbation, and
// over thousands of substeps it amplifies to visible motion.
//
// Box2D and Bullet solve this with "sleeping bodies": bodies that have
// been static for several frames are temporarily frozen. We do the
// minimal version: zero out velocities below STATIC_V_THRESHOLD for
// bodies that are CURRENTLY in contact (the constraint absorbs the
// momentum so this is energetically correct).
//
// The threshold is chosen to be:
//   • Well above the FP noise floor (~1e-12 in our scenes)
//   • Well below any meaningful orbital/drag-placement speed (~10 px/s)
// 0.1 px/s = 1/600 of a pixel per substep — invisible at any zoom level.
//
// Only IN-CONTACT bodies get damped — a body at apoapsis of a stretched
// elliptical orbit can be legitimately slow without contact, and damping
// it would corrupt the orbit.
const STATIC_V_THRESHOLD = 0.1;        // px/s
const STATIC_V_THRESHOLD2 = STATIC_V_THRESHOLD * STATIC_V_THRESHOLD;

function _applyStaticContactDamping(count) {
  if (count === 0) return;
  // Build a set of bodies that participate in any current contact. Re-using
  // a module-level Set would force us to .clear() it each call; a fresh
  // Set is cheap enough for typical contact counts (<100).
  const inContact = new Set();
  for (let k = 0; k < count; k++) {
    const c = _contacts[k];
    if (c.a && !c.a.absorbing) inContact.add(c.a);
    if (c.b && !c.b.absorbing) inContact.add(c.b);
  }
  for (const e of inContact) {
    if (e.pinned) continue;
    const v2 = e.vx * e.vx + e.vy * e.vy;
    if (v2 < STATIC_V_THRESHOLD2) {
      e.vx = 0;
      e.vy = 0;
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
      c = { a: null, b: null, rSum: 0, nx: 0, ny: 0, vnApproach: 0, preVn: 0, wasPersistent: false };
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
    // Mark whether this pair counts as a persistent contact (no
    // restitution bounce). A pair is persistent if EITHER:
    //   (i)  it was in _prevPairs (a contact last substep — warm-start)
    //   (ii) its PRE-KICK relative normal velocity is essentially zero
    //        (resting/orbital — the captured vnApproach is purely
    //        kick-induced, not a real impact)
    // Recover pre-kick velocity by subtracting a·dt from the current
    // (post-kick) velocity. For bodies with no acceleration this turn
    // (e.g., charge=0 collision setups), preVn == vnApproach.
    let wasPersistent = _prevPairs.has(_pairKey(a, b));
    // Always compute preVn — needed by the refund grace gate to
    // distinguish real impacts (large negative preVn) from contacts
    // that were "new" only by bookkeeping (e.g., first substep of a
    // statically-placed cluster where _prevPairs is still empty).
    const dt = _currentSimDt;
    const preRvx = (b.vx - b.ax * dt) - (a.vx - a.ax * dt);
    const preRvy = (b.vy - b.ay * dt) - (a.vy - a.ay * dt);
    const preVn = preRvx * nx + preRvy * ny;
    c.preVn = preVn;
    if (!wasPersistent) {
      // Real approach pre-kick → genuine new collision (wasPersistent stays false).
      // Pre-kick separating or essentially zero → persistent.
      if (preVn > PRE_KICK_APPROACH_THRESHOLD) wasPersistent = true;
    }
    c.wasPersistent = wasPersistent;
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
