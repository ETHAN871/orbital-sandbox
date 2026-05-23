// warm_start_calibrate.wgsl — K5a: per-contact warm-start seed.
// Orbital Sandbox — WebGPU Physics, Phase 2d (dark-launch).
//
// Translates physics.js:794-848 _rebuildPrevPairImpulses pairwise Plummer
// formula (persistent contacts) + blueprint §3.2 cold-start inheritance
// from min(entityMaxImpulse[idxA], entityMaxImpulse[idxB]) × 0.25.
//
// Fix #3 (d3d3e7d): r² = c.dist² + c.rSum² for the pairwise impulse,
// NOT the K1 net acceleration. 10% j_prev floor — in 2d j_prev=0 (K8
// pairImpulseTable lands in 2f). Structural form `max(jGrav, jPrev*0.10)`
// is ready; replace `0.0` line with table lookup in 2f.
//
// Output: writes contacts[t].normalImpulse only. Velocity warm-start
// application is automatic in K5 iter-0 (Δλ = jStore − 0 → atomicAdd).
//
// Per-contact workgroup_size 256. No shared memory, no barriers.

struct Contact {
  idxA: u32, idxB: u32, rSum: f32, dist: f32,
  nx: f32,   ny: f32,   vnApproach: f32, normalImpulse: f32,
  flags: u32, _p0: u32, _p1: u32, _p2: u32,
}
struct EntityMeta { mass: f32, chargeF: f32, radius: f32, flags: u32 }
struct PairCellMeta { keyA: u32, keyB: u32, j: f32, nx: f32, ny: f32 }
struct K5aParams {
  N: u32, contactCount: u32, dt: f32, G: f32,
  tableSize: u32, tableMask: u32, _pad0: u32, _pad1: u32,
}

const FLAG_ABSORBING:  u32 = 1u;
const FLAG_TOMBSTONE:  u32 = 8u;
const FLAG_PINNED:     u32 = 2u;
const CONTACT_PERSIST: u32 = 1u;
const COLD_SEED_FRAC:  f32 = 0.25;
const PERSIST_FLOOR:   f32 = 0.10;

@group(0) @binding(0) var<storage, read_write> contacts   : array<Contact>;
@group(0) @binding(1) var<storage, read>       metas      : array<EntityMeta>;
@group(0) @binding(2) var<storage, read>       maxImpulse : array<i32>;
@group(0) @binding(3) var<storage, read>       pairCellMeta  : array<PairCellMeta>;
@group(0) @binding(4) var<storage, read>       pairCellFlags : array<u32>;
@group(0) @binding(5) var<uniform>             params        : K5aParams;
// bug-fix-2026-05-23: live contact count from K4. See velocity_solver.wgsl.
@group(0) @binding(6) var<storage, read>       contactCount  : array<u32, 1>;

@compute @workgroup_size(256)
fn warm_start_calibrate(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= contactCount[0]) { return; }
  let c = contacts[t];
  let miA = metas[c.idxA];
  let miB = metas[c.idxB];
  let dead = ((miA.flags | miB.flags) & (FLAG_ABSORBING | FLAG_TOMBSTONE)) != 0u;
  if (dead) { contacts[t].normalImpulse = 0.0; return; }

  let pinA = (miA.flags & FLAG_PINNED) != 0u;
  let pinB = (miB.flags & FLAG_PINNED) != 0u;
  let wA   = select(1.0 / miA.mass, 0.0, pinA);
  let wB   = select(1.0 / miB.mass, 0.0, pinB);
  let wSum = wA + wB;
  if (wSum == 0.0) { contacts[t].normalImpulse = 0.0; return; }

  // Pairwise Plummer-softened r² (fix #3, physics.js:826).
  let r2pw = c.dist * c.dist + c.rSum * c.rSum;
  let accelA_from_B = select(miB.chargeF * params.G * miB.mass / r2pw, 0.0, miB.chargeF == 0.0);
  let accelB_from_A = select(miA.chargeF * params.G * miA.mass / r2pw, 0.0, miA.chargeF == 0.0);
  let dvn_pair = (accelA_from_B + accelB_from_A) * params.dt;
  let jGrav    = select(0.0, dvn_pair / wSum, dvn_pair > 0.0);

  var jStore: f32;
  if ((c.flags & CONTACT_PERSIST) != 0u) {
    // 2g — pairImpulseTable lookup (was literal 0.0 in 2d/2e/2f). Uses K8's
    // hash from the previous substep. Identical hash + probe to K8 in
    // rebuild_warm_start.wgsl. Miss → jPrev = 0 (cold-start path).
    let lo = min(c.idxA, c.idxB);
    let hi = max(c.idxA, c.idxB);
    let hash = (lo * 1000003u) ^ (hi * 999983u);
    var jPrev: f32 = 0.0;
    for (var probe: u32 = 0u; probe < 8u; probe = probe + 1u) {
      let slot = (hash + probe) & params.tableMask;
      if ((pairCellFlags[slot] & 1u) == 0u) { break; }   // empty → miss
      let m = pairCellMeta[slot];
      if (m.keyA == lo && m.keyB == hi) { jPrev = m.j; break; }
    }
    jStore = max(jGrav, jPrev * PERSIST_FLOOR);
  } else {
    let seedA = bitcast<f32>(maxImpulse[c.idxA]);
    let seedB = bitcast<f32>(maxImpulse[c.idxB]);
    let jSeed = min(seedA, seedB) * COLD_SEED_FRAC;
    jStore = max(jGrav, jSeed);
  }
  contacts[t].normalImpulse = jStore;
}
