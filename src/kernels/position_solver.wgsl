// position_solver.wgsl — K6: Jacobi pseudo-velocity position solver.
// Orbital Sandbox — WebGPU Physics, Phase 2e (dark-launch).
//
// Translates physics.js:347-417 _solvePositionConstraints. Mirrors K5
// structurally (per-contact accumulate + per-entity apply, fixed-point
// i32 atomics, FIXED_SCALE=16384) — just on pseudoVels instead of vels.
//
// Differences from K5:
//   - Effective positions: pos + pseudoVel * dt (physics.js:367-370)
//   - Baumgarte slop: skip correction if overlap ≤ LINEAR_SLOP_FRAC*rSum
//   - Cap correction at MAX_CORRECTION_FRAC*rSum per iter
//   - lambdaOverDt = correction/wSum/dt (pseudo-vel units)
//   - 3 iterations default (physics.js:343 POS_ITERATIONS = 3)
//
// 2e dark-launch: writes to gpuPVScratchBuf (separate from K2's
// pseudoVelsBuf). CPU stepPBD's position solver remains authoritative.

struct Contact {
  idxA: u32, idxB: u32, rSum: f32, dist: f32,
  nx: f32,   ny: f32,   vnApproach: f32, normalImpulse: f32,
  flags: u32, _p0: u32, _p1: u32, _p2: u32,
}
struct EntityMeta { mass: f32, chargeF: f32, radius: f32, flags: u32 }
struct K6Params { N: u32, contactCount: u32, dt: f32, _pad: u32 }

const FIXED_SCALE:         f32 = 16384.0;
const LINEAR_SLOP_FRAC:    f32 = 0.005;
const MAX_CORRECTION_FRAC: f32 = 0.2;
const FLAG_ABSORBING: u32 = 1u;
const FLAG_TOMBSTONE: u32 = 8u;
const FLAG_PINNED:    u32 = 2u;

@group(0) @binding(0) var<storage, read>       contacts   : array<Contact>;
@group(0) @binding(1) var<storage, read>       positions  : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> pseudoVels : array<vec2f>;
@group(0) @binding(3) var<storage, read_write> pvDelta    : array<atomic<i32>>;
@group(0) @binding(4) var<storage, read>       metas      : array<EntityMeta>;
@group(0) @binding(5) var<uniform>             params     : K6Params;

@compute @workgroup_size(256)
fn ps_accumulate(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= params.contactCount) { return; }
  let c   = contacts[t];
  let miA = metas[c.idxA];
  let miB = metas[c.idxB];
  if (((miA.flags | miB.flags) & (FLAG_ABSORBING | FLAG_TOMBSTONE)) != 0u) { return; }

  let pinA = (miA.flags & FLAG_PINNED) != 0u;
  let pinB = (miB.flags & FLAG_PINNED) != 0u;
  let wA = select(1.0 / miA.mass, 0.0, pinA);
  let wB = select(1.0 / miB.mass, 0.0, pinB);
  let wSum = wA + wB;
  if (wSum == 0.0) { return; }

  let dt = params.dt;
  let pAeff = positions[c.idxA] + pseudoVels[c.idxA] * dt;
  let pBeff = positions[c.idxB] + pseudoVels[c.idxB] * dt;
  let dxy = pBeff - pAeff;
  let dist2 = dot(dxy, dxy);
  if (dist2 < 1e-12) { return; }
  let dist = sqrt(dist2);
  let overlap = c.rSum - dist;
  let linearSlop = LINEAR_SLOP_FRAC * c.rSum;
  let want = overlap - linearSlop;
  if (want <= 0.0) { return; }
  let maxCorr = MAX_CORRECTION_FRAC * c.rSum;
  let correction = select(want, maxCorr, want > maxCorr);
  let nx = dxy.x / dist;
  let ny = dxy.y / dist;
  let lambdaOverDt = correction / wSum / dt;

  let dAx = i32(-lambdaOverDt * nx * wA * FIXED_SCALE);
  let dAy = i32(-lambdaOverDt * ny * wA * FIXED_SCALE);
  let dBx = i32( lambdaOverDt * nx * wB * FIXED_SCALE);
  let dBy = i32( lambdaOverDt * ny * wB * FIXED_SCALE);
  atomicAdd(&pvDelta[c.idxA * 2u],      dAx);
  atomicAdd(&pvDelta[c.idxA * 2u + 1u], dAy);
  atomicAdd(&pvDelta[c.idxB * 2u],      dBx);
  atomicAdd(&pvDelta[c.idxB * 2u + 1u], dBy);
}

@compute @workgroup_size(256)
fn ps_apply(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.N) { return; }
  let rawX = atomicExchange(&pvDelta[i * 2u],      0);
  let rawY = atomicExchange(&pvDelta[i * 2u + 1u], 0);
  let dvx = f32(rawX) / FIXED_SCALE;
  let dvy = f32(rawY) / FIXED_SCALE;
  pseudoVels[i] = pseudoVels[i] + vec2f(dvx, dvy);
}
