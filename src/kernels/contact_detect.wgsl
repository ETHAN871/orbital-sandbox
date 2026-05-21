// contact_detect.wgsl — K4: 3×3 neighborhood contact detection.
// Orbital Sandbox — WebGPU Physics, Phase 2c.
//
// Per-entity workgroup_size 256. No shared memory, no barriers.
// Early `return` permitted.
//
// Translates physics-spatial-hash.js:109-141 + physics.js:597-689 to GPU.
// For each non-absorbed entity i, walk its 3×3 neighborhood (with wrap if
// active), test radius overlap, branch on IS_BH to emit either an
// AbsorptionEvent or a planet-planet Contact. Also writes entityMaxImpulse
// via atomicMax<i32>(bitcast<i32>(non-negative f32)) for K5a cold-start.
//
// Phase 2c simplification (atomicOr on metas.flags deferred to 2e):
// WGSL forbids atomic ops on struct fields not declared atomic<u32>,
// and EntityMeta.flags is plain u32 (K1's frozen contract). So K4
// emits AbsorptionEvent records only — CPU drains them per substep and
// uploads FLAG_ABSORBING to metasBuf via writeBuffer for the NEXT
// substep. 1-substep lag (prey acts normal for ~2 ms before freezing)
// accepted while 2c is dark-launch. Real atomicOr design reactivates
// in 2e via a separate `entityFlagsAtomicBuf` parallel to metasBuf.
//
// CRITICAL — fix #2 preservation: cell index formula MUST be IDENTICAL
// to K3 / K3c (floor + float div + per-axis cellSize). Diverge here and
// K4 reads ghosts.

struct EntityMeta { mass: f32, chargeF: f32, radius: f32, flags: u32 }

struct Contact {
  idxA:          u32,
  idxB:          u32,
  rSum:          f32,
  dist:          f32,
  nx:            f32,
  ny:            f32,
  vnApproach:    f32,
  normalImpulse: f32,
  flags:         u32,    // bit 0 = wasPersistent
  _pad0:         u32,
  _pad1:         u32,
  _pad2:         u32,    // 48 B total — blueprint §4
}

struct AbsorptionEvent {
  preyIdx:     u32,
  predatorIdx: u32,
  _pad0:       u32,
  _pad1:       u32,    // 16 B
}

struct K4Params {
  N:         u32,
  ncx:       u32,
  ncy:       u32,
  numCells:  u32,
  cellSizeX: f32,
  cellSizeY: f32,
  dt:        f32,
  wrap:      u32,    // 1 = wrap mode, 0 = destroy. W/H derived from ncx*cellSizeX.
}

const FLAG_ABSORBING: u32 = 1u;
const FLAG_TOMBSTONE: u32 = 8u;
const FLAG_IS_BH:     u32 = 4u;
const CONTACT_PERSISTENT: u32 = 1u;
const PRE_KICK_VN_THRESHOLD: f32 = -1e-6;
const STATUS_ABS_OVERFLOW:     u32 = 2u;
const STATUS_CONTACT_OVERFLOW: u32 = 8u;
const ABS_CAPACITY: u32 = 128u;

@group(0) @binding(0)  var<storage, read>       positions    : array<vec2f>;
@group(0) @binding(1)  var<storage, read>       velocities   : array<vec2f>;
@group(0) @binding(2)  var<storage, read>       accels       : array<vec2f>;
@group(0) @binding(3)  var<storage, read>       metas        : array<EntityMeta>;
@group(0) @binding(4)  var<storage, read>       cellCounts   : array<u32>;
@group(0) @binding(5)  var<storage, read>       cellOffsets  : array<u32>;
@group(0) @binding(6)  var<storage, read>       cellContents : array<u32>;
@group(0) @binding(7)  var<storage, read_write> contacts     : array<Contact>;
@group(0) @binding(8)  var<storage, read_write> contactCount : array<atomic<u32>>;
@group(0) @binding(9)  var<storage, read_write> absEvents    : array<AbsorptionEvent>;
@group(0) @binding(10) var<storage, read_write> absHead      : array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> maxImpulse   : array<atomic<i32>>;
@group(0) @binding(12) var<storage, read_write> statusFlags  : array<atomic<u32>>;
@group(0) @binding(13) var<uniform>             params       : K4Params;

@compute @workgroup_size(256)
fn contact_detect(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.N) { return; }
  let mi = metas[i];
  if ((mi.flags & (FLAG_ABSORBING | FLAG_TOMBSTONE)) != 0u) { return; }

  let pi  = positions[i];
  let vi  = velocities[i];
  let ai  = accels[i];
  let W   = f32(params.ncx) * params.cellSizeX;
  let H   = f32(params.ncy) * params.cellSizeY;
  let halfW = W * 0.5;
  let halfH = H * 0.5;
  let wrap = params.wrap != 0u;
  let maxContacts = 3u * params.N;

  // Fix #2 — IDENTICAL formula to K3/K3c.
  let cx_i = i32(min(u32(floor(pi.x / params.cellSizeX)), params.ncx - 1u));
  let cy_i = i32(min(u32(floor(pi.y / params.cellSizeY)), params.ncy - 1u));

  for (var dcx: i32 = -1; dcx <= 1; dcx = dcx + 1) {
    for (var dcy: i32 = -1; dcy <= 1; dcy = dcy + 1) {
      var qcx: i32 = cx_i + dcx;
      var qcy: i32 = cy_i + dcy;
      if (wrap) {
        qcx = (qcx + i32(params.ncx)) % i32(params.ncx);
        qcy = (qcy + i32(params.ncy)) % i32(params.ncy);
      } else {
        if (qcx < 0 || qcx >= i32(params.ncx)) { continue; }
        if (qcy < 0 || qcy >= i32(params.ncy)) { continue; }
      }
      let cellIdx = u32(qcy) * params.ncx + u32(qcx);
      if (cellIdx >= params.numCells) { continue; }
      let offset = cellOffsets[cellIdx];
      let count  = cellCounts[cellIdx];

      for (var k: u32 = 0u; k < count; k = k + 1u) {
        let j = cellContents[offset + k];
        if (j <= i) { continue; }   // de-dupe + self-skip
        let mj = metas[j];
        if ((mj.flags & (FLAG_ABSORBING | FLAG_TOMBSTONE)) != 0u) { continue; }

        var dxy = positions[j] - pi;
        if (wrap) {
          if (dxy.x >  halfW) { dxy.x = dxy.x - W; }
          else if (dxy.x < -halfW) { dxy.x = dxy.x + W; }
          if (dxy.y >  halfH) { dxy.y = dxy.y - H; }
          else if (dxy.y < -halfH) { dxy.y = dxy.y + H; }
        }

        let rSum  = mi.radius + mj.radius;
        let dist2 = dot(dxy, dxy);
        if (dist2 >= rSum * rSum) { continue; }

        // Cold-start impulse seed for K5a (blueprint §3.2). |accel[j]| × dt
        // is a non-negative proxy for the pairwise impulse magnitude; safe
        // for atomicMax<i32>(bitcast<i32>(f32)) ordering trick.
        let jEst = length(accels[j]) * params.dt;
        atomicMax(&maxImpulse[i], bitcast<i32>(jEst));
        atomicMax(&maxImpulse[j], bitcast<i32>(jEst));

        let aIsBH = (mi.flags & FLAG_IS_BH) != 0u;
        let bIsBH = (mj.flags & FLAG_IS_BH) != 0u;

        if (aIsBH || bIsBH) {
          // BH absorption — physics.js:606-616 prey/predator selection.
          var preyIdx: u32;
          var predIdx: u32;
          if (aIsBH && !bIsBH)        { preyIdx = j; predIdx = i; }
          else if (!aIsBH && bIsBH)   { preyIdx = i; predIdx = j; }
          else if (mi.mass < mj.mass) { preyIdx = i; predIdx = j; }
          else if (mj.mass < mi.mass) { preyIdx = j; predIdx = i; }
          else { continue; }   // equal-mass BH stalemate

          let slot = atomicAdd(&absHead[0], 1u);
          if (slot < ABS_CAPACITY) {
            absEvents[slot].preyIdx     = preyIdx;
            absEvents[slot].predatorIdx = predIdx;
            absEvents[slot]._pad0       = 0u;
            absEvents[slot]._pad1       = 0u;
          } else {
            atomicOr(&statusFlags[0], STATUS_ABS_OVERFLOW);
          }
        } else {
          // Planet-planet contact — physics.js:648-705.
          let dist = sqrt(dist2);
          var nx: f32;
          var ny: f32;
          if (dist < 1e-6) {
            nx = 1.0; ny = 0.0;
          } else {
            nx = dxy.x / dist; ny = dxy.y / dist;
          }
          let normal = vec2f(nx, ny);

          let vj = velocities[j];
          let rv = vj - vi;
          let vnApproach = dot(rv, normal);

          // wasPersistent — pre-kick approach test (physics.js:668-672).
          // No hash table in 2c → no warm-start lookup branch.
          let aj  = accels[j];
          let preRv = (vj - aj * params.dt) - (vi - ai * params.dt);
          let preVn = dot(preRv, normal);
          var cflags: u32 = 0u;
          if (preVn > PRE_KICK_VN_THRESHOLD) { cflags = CONTACT_PERSISTENT; }

          let slot = atomicAdd(&contactCount[0], 1u);
          if (slot < maxContacts) {
            contacts[slot].idxA          = i;
            contacts[slot].idxB          = j;
            contacts[slot].rSum          = rSum;
            contacts[slot].dist          = dist;
            contacts[slot].nx            = nx;
            contacts[slot].ny            = ny;
            contacts[slot].vnApproach    = vnApproach;
            contacts[slot].normalImpulse = 0.0;   // warm-start deferred to 2f
            contacts[slot].flags         = cflags;
            contacts[slot]._pad0         = 0u;
            contacts[slot]._pad1         = 0u;
            contacts[slot]._pad2         = 0u;
          } else {
            atomicOr(&statusFlags[0], STATUS_CONTACT_OVERFLOW);
          }
        }
      }
    }
  }
}
