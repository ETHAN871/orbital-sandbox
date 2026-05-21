// broadphase_count.wgsl — K3: per-cell entity histogram (atomic counting).
// Orbital Sandbox — WebGPU Physics, Phase 2b.
//
// Translates physics-spatial-hash.js lines 83-92 (buildSpatialHash bucket
// fill). For each non-absorbed entity i, compute its cell index and
// atomicAdd 1 to cellCounts[cellIdx]. Result feeds K3b prefix scan + K3c.
//
// CRITICAL — fix #2 (38a7aee, see docs/webgpu-blueprint.md §7):
//   ncx = max(1, floor(W / minCellSize))    [computed CPU-side]
//   cellSizeX = W / ncx                      [float divide, NOT integer]
//   cx = floor(pos.x / cellSizeX)            [floor, NOT ceil]
//   cy = floor(pos.y / cellSizeY)
//   cellIdx = cy * ncx + cx
// Any deviation (ceil, integer div, etc.) re-introduces the Y-wrap dead
// zone bug — see physics-spatial-hash.js:36-77 for the 40-line incident
// report. Test fixture B5 (per-substep broadphase) catches regression.
//
// INVARIANT — no barriers, no shared memory: per-entity map kernel. Each
// thread owns global_invocation_id.x and writes only to its cell's slot
// in cellCounts via atomicAdd. Early `return` permitted.

struct EntityMeta {
  mass:    f32,
  chargeF: f32,
  radius:  f32,
  flags:   u32,
}

struct BroadphaseParams {
  N:         u32,
  ncx:       u32,
  ncy:       u32,
  numCells:  u32,    // ncx * ncy (precomputed CPU-side)
  cellSizeX: f32,
  cellSizeY: f32,
  _pad0:     u32,
  _pad1:     u32,    // 32 B total
}

const FLAG_ABSORBING: u32 = 1u;
const FLAG_TOMBSTONE: u32 = 8u;

@group(0) @binding(0) var<storage, read>             positions  : array<vec2f>;
@group(0) @binding(1) var<storage, read>             metas      : array<EntityMeta>;
@group(0) @binding(2) var<storage, read_write>       cellCounts : array<atomic<u32>>;
@group(0) @binding(3) var<uniform>                   params     : BroadphaseParams;

@compute @workgroup_size(256)
fn broadphase_count(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.N) { return; }

  let flags = metas[i].flags;
  // ABSORBING (mid-eaten) + TOMBSTONE (dead, not yet compacted) excluded
  // from broadphase. Matches physics-spatial-hash.js:85.
  if ((flags & (FLAG_ABSORBING | FLAG_TOMBSTONE)) != 0u) { return; }

  let pos = positions[i];
  // Fix #2 — floor + float div, per-axis cell size.
  // K2's NaN guard upstream zeros bad positions to (0,0) so they safely
  // land in cell 0. u32(NaN) = 0 by WGSL spec is a backup safety net.
  let cx = min(u32(floor(pos.x / params.cellSizeX)), params.ncx - 1u);
  let cy = min(u32(floor(pos.y / params.cellSizeY)), params.ncy - 1u);
  let cellIdx = cy * params.ncx + cx;

  atomicAdd(&cellCounts[cellIdx], 1u);
}
