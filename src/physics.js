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

import { G, EPSILON, PREDICT_STEPS, PREDICT_DT } from './state.js';

// ─── Force / acceleration ─────────────────────────────────────────
// Returns parallel arrays-of-zero accumulator filled in-place.
// We allocate once per step to avoid GC pressure in the inner loop.

export function computeAccelerations(entities, accels) {
  const n = entities.length;
  for (let i = 0; i < n; i++) { accels[i].ax = 0; accels[i].ay = 0; }

  for (let i = 0; i < n; i++) {
    const a = entities[i];
    for (let j = i + 1; j < n; j++) {
      const b = entities[j];
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

  // Position update using cached acceleration from last step.
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    e.x += e.vx * dt + 0.5 * e.ax * dt * dt;
    e.y += e.vy * dt + 0.5 * e.ay * dt * dt;
  }

  // Compute new accelerations at the new positions.
  computeAccelerations(entities, _scratch);

  // Velocity update averages old + new acceleration; then store new as next-step cache.
  for (let i = 0; i < n; i++) {
    const e = entities[i];
    const newAx = _scratch[i].ax;
    const newAy = _scratch[i].ay;
    e.vx += 0.5 * (e.ax + newAx) * dt;
    e.vy += 0.5 * (e.ay + newAy) * dt;
    e.ax = newAx;
    e.ay = newAy;
  }
}

// ─── Collisions ───────────────────────────────────────────────────
// Any object whose center distance to a black hole is < sum-of-radii
// gets consumed (removed from the simulation). Black-hole vs black-hole:
// the smaller (lower mass) is consumed by the larger. Planet-vs-planet
// collisions are ignored per spec.

export function handleCollisions(entities) {
  for (let i = entities.length - 1; i >= 0; i--) {
    const a = entities[i];
    for (let j = 0; j < entities.length; j++) {
      if (i === j) continue;
      const b = entities[j];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const rSum = a.radius + b.radius;
      if (dx * dx + dy * dy < rSum * rSum) {
        // a touches b. Decide if a is consumed.
        const aDevoured =
          (b.type === 'black_hole' && a.type !== 'black_hole') ||
          (b.type === 'black_hole' && a.type === 'black_hole' && a.mass < b.mass);
        if (aDevoured) {
          entities.splice(i, 1);
          break; // i no longer valid
        }
      }
    }
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
    if (o.charge === 0) continue;
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
    if (o.type !== 'black_hole') continue;
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
