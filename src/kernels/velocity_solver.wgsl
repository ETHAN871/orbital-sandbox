// velocity_solver.wgsl — K5: Jacobi velocity solver.
// Orbital Sandbox — WebGPU Physics, Phase 2d (dark-launch via gpuVelScratch).
//
// Two entry points sharing ONE bind group layout (WGSL requires unique
// @binding per source file). vs_apply ignores `contacts` and `metas`
// bindings — they're present in its access set but accessed via dead
// reads removed by the compiler.
//
//   vs_accumulate (per-contact): reads velocities[], updates
//     contacts[t].normalImpulse (race-free per contact), atomicAdds
//     fixed-point Δv into velDelta[] for both endpoints.
//   vs_apply (per-entity): atomicExchange consumes the i32 delta and
//     applies to velocities[i], leaving slot zeroed for next iteration.
//
// Translates physics.js:483-562. SI/GS on CPU → Jacobi on GPU per
// blueprint §3.1 (red-black GS rejected: hex contact graph has triangles
// → line graph non-bipartite → 2-coloring impossible).
//
// FIXED_SCALE=16384 (2^14, blueprint G9). Per-slot max ~1000 px/s × FS
// = 1.6e7, well under i32 max 2.1e9.

struct Contact {
  idxA: u32, idxB: u32, rSum: f32, dist: f32,
  nx: f32,   ny: f32,   vnApproach: f32, normalImpulse: f32,
  flags: u32, _p0: u32, _p1: u32, _p2: u32,
}
struct EntityMeta { mass: f32, chargeF: f32, radius: f32, flags: u32 }
struct K5Params { N: u32, contactCount: u32, dt: f32, e: f32 }

const FIXED_SCALE:     f32 = 16384.0;
const FLAG_ABSORBING:  u32 = 1u;
const FLAG_TOMBSTONE:  u32 = 8u;
const FLAG_PINNED:     u32 = 2u;
const CONTACT_PERSIST: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> contacts     : array<Contact>;
@group(0) @binding(1) var<storage, read_write> velocities   : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velDelta     : array<atomic<i32>>;
@group(0) @binding(3) var<storage, read>       metas        : array<EntityMeta>;
@group(0) @binding(4) var<uniform>             params       : K5Params;
// bug-fix-2026-05-23: live contact count from K4 (same substep). Replaces
// params.contactCount which was 1-substep-stale (prevContactCount), causing
// K5 to process old contacts that K4 didn't actually detect this substep
// → visible collision-response lag.
@group(0) @binding(5) var<storage, read>       contactCount : array<u32, 1>;

@compute @workgroup_size(256)
fn vs_accumulate(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= contactCount[0]) { return; }
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

  let vA = velocities[c.idxA];
  let vB = velocities[c.idxB];
  let n  = vec2f(c.nx, c.ny);
  let rv = vB - vA;
  let vn = dot(rv, n);
  let vnNan = vn != vn;

  let vnTarget = select(-params.e * c.vnApproach, 0.0, (c.flags & CONTACT_PERSIST) != 0u);
  let dLambdaRaw = select((vnTarget - vn) / wSum, 0.0, vnNan);

  let oldImp = c.normalImpulse;
  let newImp = max(0.0, oldImp + dLambdaRaw);   // accumulated-impulse clamp ≥ 0
  let dLambda = newImp - oldImp;
  if (dLambda == 0.0) { return; }
  contacts[t].normalImpulse = newImp;

  let dAx = i32(-dLambda * c.nx * wA * FIXED_SCALE);
  let dAy = i32(-dLambda * c.ny * wA * FIXED_SCALE);
  let dBx = i32( dLambda * c.nx * wB * FIXED_SCALE);
  let dBy = i32( dLambda * c.ny * wB * FIXED_SCALE);
  atomicAdd(&velDelta[c.idxA * 2u],      dAx);
  atomicAdd(&velDelta[c.idxA * 2u + 1u], dAy);
  atomicAdd(&velDelta[c.idxB * 2u],      dBx);
  atomicAdd(&velDelta[c.idxB * 2u + 1u], dBy);
}

@compute @workgroup_size(256)
fn vs_apply(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.N) { return; }
  let rawX = atomicExchange(&velDelta[i * 2u],      0);
  let rawY = atomicExchange(&velDelta[i * 2u + 1u], 0);
  let dvx = f32(rawX) / FIXED_SCALE;
  let dvy = f32(rawY) / FIXED_SCALE;
  velocities[i] = velocities[i] + vec2f(dvx, dvy);
}
