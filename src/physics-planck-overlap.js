// physics-planck-overlap.js — adaptive solver-iteration manager.
//
// Sole reason this module exists: planck.js's default 3 position iterations
// + 8 velocity iterations cannot resolve the constraint network in dense
// clusters (Mechanism ❷ in the 2026-05-23 over-penetration triage). The
// Sequential Impulses + NGS pattern is provably convergent given infinite
// iterations, but on a body touching 4+ neighbors simultaneously, 3 NGS
// iterations leave residual overlap that compounds substep-over-substep.
//
// Naive fix: just always run heavy (24 vel / 12 pos). Cost: doubles solver
// time on every frame including the >99% of frames where nobody's touching
// anyone. Empirically wipes out any 5000@60 target we have.
//
// Adaptive fix (this module): each step end, walk the contact list once
// and count contacts where `isTouching() === true`. The walk is cheap —
// O(C) where C is broadphase-AABB-overlap pairs, which is near-zero in
// sparse scenes. Next step's iteration counts scale linearly with the
// touching count, with a hysteresis cooldown so a contact-then-clear-then-
// contact pattern doesn't oscillate between heavy and light.
//
// Engine-agnostic by design: `recordPostStep(world)` takes an opaque world
// argument. If we ever swap planck for Rapier2D, only this method's
// internals need updating; the iteration-count state machine and the
// public surface stay identical.

// Default (planck-era) iteration range — exported for backward compat.
// Callers SHOULD pass engine-specific values via the constructor's
// `iters` parameter (Rapier path passes SOLVER_ITERATIONS_* / PGS_*).
// Constructors that omit `iters` fall through to these defaults so
// pre-existing planck call sites keep their original 8/3 baseline.
export const OVERLAP_VEL_ITER_BASE = 8;
export const OVERLAP_VEL_ITER_MAX  = 24;
export const OVERLAP_POS_ITER_BASE = 3;
export const OVERLAP_POS_ITER_MAX  = 12;

// Two-counter state machine:
//   _liveTouchingCount  — count from the most recent recordPostStep call.
//                         Drives decideIterations directly while above
//                         threshold.
//   _peakDuringCooldown — the largest count observed in the active
//                         escalation episode. Used while live count has
//                         dropped below threshold but cooldown still
//                         holds heavy mode. This prevents the regression
//                         the code-reviewer flagged: a fleeting 2-pair
//                         spike followed by zero count would otherwise
//                         "hold" at iter counts indistinguishable from
//                         baseline (`floor(0/4)+8 = 8`), defeating the
//                         entire cooldown.

export class AdaptiveOverlapManager {
  // `iters` (optional) overrides the engine-specific iteration range:
  //   { velBase, velMax, posBase, posMax }
  // When omitted, defaults to the planck-era 8/24/3/12 range so
  // existing planck-backend constructions keep their original baseline.
  // Rapier backend passes 4/8/2/4 — see SOLVER_ITERATIONS_* and
  // PGS_ITERATIONS_* in physics-rapier.js.
  constructor(state, iters) {
    this._state = state;
    this._velBase = iters?.velBase ?? OVERLAP_VEL_ITER_BASE;
    this._velMax  = iters?.velMax  ?? OVERLAP_VEL_ITER_MAX;
    this._posBase = iters?.posBase ?? OVERLAP_POS_ITER_BASE;
    this._posMax  = iters?.posMax  ?? OVERLAP_POS_ITER_MAX;
    this._liveTouchingCount = 0;
    this._peakDuringCooldown = 0;
    this._cooldownRemaining = 0;
  }

  decideIterations() {
    const threshold = this._state.overlapEscalateThreshold;
    const live = this._liveTouchingCount;
    if (live > threshold) {
      return this._scale(live);
    }
    if (this._cooldownRemaining > 0) {
      return this._scale(this._peakDuringCooldown);
    }
    return [this._velBase, this._posBase];
  }

  _scale(count) {
    const vi = Math.min(
      this._velMax,
      this._velBase + Math.floor(count / 4),
    );
    const pi = Math.min(
      this._posMax,
      this._posBase + Math.floor(count / 8),
    );
    return [vi, pi];
  }

  // Engine-agnostic: caller passes the number of touching contact pairs
  // computed this step. planck path uses world.getContactList() +
  // isTouching(); rapier path iterates world.contactPairsWith.
  // The state machine here doesn't care which engine produced the
  // count — that's how the manager stays portable.
  recordPostStep(touchingCount) {
    const threshold = this._state.overlapEscalateThreshold;
    const cooldown  = this._state.overlapCooldownFrames;
    const count = touchingCount | 0;
    this._liveTouchingCount = count;
    if (count > threshold) {
      this._cooldownRemaining = cooldown;
      if (count > this._peakDuringCooldown) this._peakDuringCooldown = count;
    } else if (this._cooldownRemaining > 0) {
      this._cooldownRemaining--;
      if (this._cooldownRemaining === 0) this._peakDuringCooldown = 0;
    }
  }

  reset() {
    this._liveTouchingCount = 0;
    this._peakDuringCooldown = 0;
    this._cooldownRemaining = 0;
  }

  get debug() {
    return {
      liveTouchingCount: this._liveTouchingCount,
      peakDuringCooldown: this._peakDuringCooldown,
      cooldownRemaining: this._cooldownRemaining,
    };
  }
}
