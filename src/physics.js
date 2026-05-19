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
  G, EPSILON, PREDICT_STEPS, PREDICT_DT,
  ABSORPTION_DURATION, ELASTIC_RESTITUTION,
} from './state.js';

// ─── Force / acceleration ─────────────────────────────────────────
// Returns parallel arrays-of-zero accumulator filled in-place.
// We allocate once per step to avoid GC pressure in the inner loop.

export function computeAccelerations(entities, accels) {
  const n = entities.length;
  for (let i = 0; i < n; i++) { accels[i].ax = 0; accels[i].ay = 0; }

  for (let i = 0; i < n; i++) {
    const a = entities[i];
    // Absorbing entities are visually being eaten; they neither apply nor
    // receive gravity (avoids late-stage tugs that look like the body is
    // resisting absorption).
    if (a.absorbing) continue;
    for (let j = i + 1; j < n; j++) {
      const b = entities[j];
      if (b.absorbing) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const r2Raw = dx * dx + dy * dy;
      const minR = Math.max(a.radius + b.radius, EPSILON);
      const r2 = Math.max(r2Raw, minR * minR);
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

function ensureScratch(n) {
  while (_scratch.length < n) _scratch.push({ ax: 0, ay: 0 });
  // Trim when entity population shrinks substantially (e.g., after a clear or
  // mass black-hole consumption). Keeps the visible-state-only invariant
  // `_scratch[0..n-1] is meaningful` from drifting if loop bounds ever change.
  if (_scratch.length > n * 2 && _scratch.length > 16) {
    _scratch.length = Math.max(n, 16);
  }
}

export function stepVerlet(entities, dt) {
  const n = entities.length;
  if (n === 0 || dt === 0) return;
  ensureScratch(n);

  // Position update using cached acceleration from last step. Pinned bodies
  // are skipped — they neither move nor accelerate, only act as anchors.
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned) continue;
    e.x += e.vx * dt + 0.5 * e.ax * dt * dt;
    e.y += e.vy * dt + 0.5 * e.ay * dt * dt;
  }

  // Compute new accelerations at the new positions.
  computeAccelerations(entities, _scratch);

  // Velocity update averages old + new acceleration; then store new as next-step cache.
  // Pinned bodies have their kinematic state clamped to zero — even if the
  // user toggled pin while the body was in motion, this halts it cleanly.
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.pinned) { e.vx = 0; e.vy = 0; e.ax = 0; e.ay = 0; continue; }
    const newAx = _scratch[i].ax;
    const newAy = _scratch[i].ay;
    e.vx += 0.5 * (e.ax + newAx) * dt;
    e.vy += 0.5 * (e.ay + newAy) * dt;
    e.ax = newAx;
    e.ay = newAy;
  }
}

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

export function handleCollisions(entities) {
  const n = entities.length;
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
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const rSum = a.radius + b.radius;
      const dist2 = dx * dx + dy * dy;
      if (dist2 >= rSum * rSum) continue;          // not touching

      const aIsBH = a.type === 'black_hole';
      const bIsBH = b.type === 'black_hole';

      if (aIsBH || bIsBH) {
        // Pick prey: non-BH always loses to BH; between two BHs, smaller mass loses.
        let prey, predator;
        if (aIsBH && !bIsBH) { prey = b; predator = a; }
        else if (!aIsBH && bIsBH) { prey = a; predator = b; }
        else if (a.mass < b.mass) { prey = a; predator = b; }
        else if (b.mass < a.mass) { prey = b; predator = a; }
        else continue;                              // equal-mass BHs: stalemate, no effect
        // Pinned bodies are intentional anchors — they are immune to absorption.
        if (prey.pinned) continue;
        beginAbsorption(prey, predator);
      } else {
        resolveElasticCollision(a, b, dx, dy, dist2, rSum);
      }
    }
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

// Standard 2D elastic-impulse collision. Conserves momentum; conserves KE
// when ELASTIC_RESTITUTION = 1. Penetration is corrected proportional to
// inverse mass so the heavier body barely moves.
//
// NOTE on Verlet interaction: this function mutates position (penetration
// correction) and velocity (impulse). The Verlet integrator already cached
// new `ax`,`ay` for the *pre-separation* positions earlier in this substep.
// We deliberately do NOT recompute accelerations here — the O(dt²) staleness
// self-corrects on the next substep, and a re-pass would cost O(n²).
function resolveElasticCollision(a, b, dx, dy, dist2, rSum) {
  // Pinned bodies behave as infinite-mass anchors: inverse mass = 0 means
  // they receive zero impulse and zero positional correction, and all the
  // motion is absorbed by the non-pinned partner. Two pinned bodies stuck
  // overlapping is a user choice — just bail.
  const invMa = a.pinned ? 0 : 1 / a.mass;
  const invMb = b.pinned ? 0 : 1 / b.mass;
  const invMassSum = invMa + invMb;
  if (invMassSum === 0) return;

  const dist = Math.sqrt(dist2);
  if (dist === 0) {
    // Perfectly co-located: pick a single arbitrary axis (+x) and split mass-
    // weighted along it. Splitting along BOTH axes pushes only `rSum/2 * √2 / 2 ≈ 0.353·rSum`
    // of separation per body — still overlapping next frame and causing jitter.
    // Pushing rSum along one axis cleanly separates regardless of pinned status.
    a.x -= rSum * (invMa / invMassSum);
    b.x += rSum * (invMb / invMassSum);
    return;
  }
  const nx = dx / dist;
  const ny = dy / dist;

  // 1. Resolve penetration (positional correction, mass-weighted).
  const penetration = rSum - dist;
  if (penetration > 0) {
    const aShare = invMa / invMassSum;
    a.x -= nx * penetration * aShare;
    a.y -= ny * penetration * aShare;
    b.x += nx * penetration * (1 - aShare);
    b.y += ny * penetration * (1 - aShare);
  }

  // 2. Apply impulse if bodies are approaching.
  const rvx = b.vx - a.vx;
  const rvy = b.vy - a.vy;
  const velAlongNormal = rvx * nx + rvy * ny;
  if (velAlongNormal > 0) return;                  // already separating

  const j = -(1 + ELASTIC_RESTITUTION) * velAlongNormal / invMassSum;
  const ix = j * nx;
  const iy = j * ny;
  a.vx -= ix * invMa;
  a.vy -= iy * invMa;
  b.vx += ix * invMb;
  b.vy += iy * invMb;
}

// ─── Absorption animation ─────────────────────────────────────────
// Progress every absorbing entity by `dt` seconds (sim time). Splice when
// the animation completes. If the predator black hole vanished (e.g., a
// bigger BH ate it), end the animation immediately — the prey just pops.

export function updateAbsorptions(entities, dt) {
  if (dt <= 0) return;
  for (let i = entities.length - 1; i >= 0; i--) {
    const e = entities[i];
    const abs = e.absorbing;
    if (!abs) continue;

    const bh = entities.find(x => x.id === abs.blackHoleId);
    if (!bh) { entities.splice(i, 1); continue; }

    abs.elapsedSim += dt;
    const t = Math.min(1, abs.elapsedSim / ABSORPTION_DURATION);

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

export function predictTrajectory(ghost, others) {
  const path = new Array(PREDICT_STEPS);
  let x = ghost.x;
  let y = ghost.y;
  let vx = ghost.vx;
  let vy = ghost.vy;
  let ax = 0;
  let ay = 0;

  // Initial acceleration from frozen snapshot.
  ({ ax, ay } = ghostAccel(x, y, ghost.radius, others));

  for (let s = 0; s < PREDICT_STEPS; s++) {
    // Verlet position step.
    x += vx * PREDICT_DT + 0.5 * ax * PREDICT_DT * PREDICT_DT;
    y += vy * PREDICT_DT + 0.5 * ay * PREDICT_DT * PREDICT_DT;

    // Stop early if the ghost would already have been consumed by a black hole.
    if (touchesBlackHole(x, y, ghost.radius, others)) {
      path.length = s + 1;
      path[s] = { x, y };
      break;
    }

    const next = ghostAccel(x, y, ghost.radius, others);
    vx += 0.5 * (ax + next.ax) * PREDICT_DT;
    vy += 0.5 * (ay + next.ay) * PREDICT_DT;
    ax = next.ax;
    ay = next.ay;
    path[s] = { x, y };
  }
  return path;
}

function ghostAccel(x, y, radius, others) {
  let ax = 0, ay = 0;
  for (let k = 0; k < others.length; k++) {
    const o = others[k];
    if (o.charge === 0 || o.absorbing) continue;
    const dx = o.x - x;
    const dy = o.y - y;
    const r2Raw = dx * dx + dy * dy;
    const minR = Math.max(radius + o.radius, EPSILON);
    const r2 = Math.max(r2Raw, minR * minR);
    const r = Math.sqrt(r2);
    const mag = o.charge * G * o.mass / r2;
    ax += mag * dx / r;
    ay += mag * dy / r;
  }
  return { ax, ay };
}

function touchesBlackHole(x, y, radius, others) {
  for (let k = 0; k < others.length; k++) {
    const o = others[k];
    if (o.type !== 'black_hole' || o.absorbing) continue;
    const dx = o.x - x;
    const dy = o.y - y;
    const r = radius + o.radius;
    if (dx * dx + dy * dy < r * r) return true;
  }
  return false;
}

// ─── Trail maintenance ────────────────────────────────────────────
// Called by main loop after each frame to record a sample.
// The trail is capped at `maxLen` points; older samples drop off the front.

export function appendTrail(entity, maxLen) {
  if (maxLen <= 0) {
    entity.trail.length = 0;
    return;
  }
  entity.trail.push({ x: entity.x, y: entity.y });
  // Single splice handles both the steady-state +1 case AND a slider-slam
  // (e.g. 500 → 20) in one call, instead of N×O(n) Array.shift() invocations.
  if (entity.trail.length > maxLen) {
    entity.trail.splice(0, entity.trail.length - maxLen);
  }
}
