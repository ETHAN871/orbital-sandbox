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
  G, EPSILON, PREDICT_STEPS, PREDICT_DT,
  ABSORPTION_DURATION, ELASTIC_RESTITUTION,
  BOUNDARY_BUFFER_FACTOR,
} from './state.js';
import { prepareBHTree, computeAccelerationsBH } from './physics-barneshut.js';
import { buildSpatialHash, forEachCollisionPair } from './physics-spatial-hash.js';

// V8.2 dispatch threshold. Below this entity count, the O(N²) direct sum
// has lower constant factor than Barnes-Hut tree build + 9-ghost recursion;
// above it, BH wins decisively.
const BH_THRESHOLD = 64;

// V8.2: prepare per-frame data structures once. main.js calls this before
// the substep loop so the quadtree + spatial hash are built exactly 1× per
// frame instead of up-to-8× (once per substep). Position drift between
// substeps is bounded by SIM_DT × velocity and is within Verlet's
// integration tolerance.
export function prepareFrame(entities) {
  if (entities.length >= BH_THRESHOLD) {
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

// (pairDelta was V5; V7 inlined it into the N² hot paths to avoid object
//  allocation. minImageDelta is still used by ghostAccel/touchesBlackHole
//  where the call count is much lower.)

// ─── Force / acceleration ─────────────────────────────────────────
// Returns parallel arrays-of-zero accumulator filled in-place.
// We allocate once per step to avoid GC pressure in the inner loop.

export function computeAccelerations(entities, accels) {
  const n = entities.length;
  // V8.2: dispatch to Barnes-Hut for large N. The direct O(N²) sum below
  // wins at small N due to lower constant factor + cache locality.
  if (n >= BH_THRESHOLD) {
    computeAccelerationsBH(entities, accels);
    return;
  }
  for (let i = 0; i < n; i++) { accels[i].ax = 0; accels[i].ay = 0; }

  // V7 perf: hoist wrap/viewport reads out of the N² hot path and inline
  // pairDelta so the inner loop allocates zero objects.
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width  : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;

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
    resolveElasticCollision(a, b, dx, dy, dist2, rSum);
  }
}

export function handleCollisions(entities) {
  const n = entities.length;

  // V8.2: large-N path uses wrap-aware spatial hash for O(N·k) broadphase.
  // The hash is built once per frame in prepareFrame() above; this just
  // iterates the already-built buckets.
  if (n >= BH_THRESHOLD) {
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
      // V8.2: dispatch through shared helper so both N<64 and N≥64 paths
      // run identical collision/absorption logic.
      processCollisionPair(a, b, dx, dy);
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

// Module-level scratch for prediction — reused across calls so the drag's
// per-frame predict doesn't allocate 300 `{x,y}` objects + a fresh path
// array each invocation. The returned object holds a reference to this
// same buffer, with a `length` indicating how many samples are valid.
const _predictBuf = new Float32Array(PREDICT_STEPS * 2);
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

  // Initial acceleration from frozen snapshot (writes into _ghostAccelScratch).
  ghostAccel(x, y, ghost.radius, others, _ghostAccelScratch);
  let ax = _ghostAccelScratch.ax;
  let ay = _ghostAccelScratch.ay;

  let written = 0;
  for (let s = 0; s < PREDICT_STEPS; s++) {
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
    const r2 = Math.max(r2Raw, minR * minR);
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

// V8.1: appendTrail removed. Trails are now rendered as a global
// phosphor-decay canvas (see renderer.updateTrailCanvas), so no per-entity
// history is maintained in physics anymore.

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
      // V8.1: trail is a phosphor-decay canvas, not per-entity history;
      // the new dot just lands on the wrapped position and the old dot
      // (left on the other side) fades out naturally — no special case.
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
