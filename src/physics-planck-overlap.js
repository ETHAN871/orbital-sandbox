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
  constructor(state) {
    this._state = state;
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
    return [OVERLAP_VEL_ITER_BASE, OVERLAP_POS_ITER_BASE];
  }

  _scale(count) {
    const vi = Math.min(
      OVERLAP_VEL_ITER_MAX,
      OVERLAP_VEL_ITER_BASE + Math.floor(count / 4),
    );
    const pi = Math.min(
      OVERLAP_POS_ITER_MAX,
      OVERLAP_POS_ITER_BASE + Math.floor(count / 8),
    );
    return [vi, pi];
  }

  recordPostStep(world) {
    const threshold = this._state.overlapEscalateThreshold;
    const cooldown  = this._state.overlapCooldownFrames;
    let count = 0;
    for (let c = world.getContactList(); c; c = c.getNext()) {
      if (c.isTouching()) count++;
    }
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
