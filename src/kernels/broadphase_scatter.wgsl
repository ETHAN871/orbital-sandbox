// broadphase_scatter.wgsl — K3c: write entity indices into cellContents.
// Orbital Sandbox — WebGPU Physics, Phase 2b.
//
// Translates the final step of buildSpatialHash (physics-spatial-hash.js
// line 91, bucket.push(e)) to GPU. For each non-absorbed entity i, claim
// one slot in its cell's range via atomicAdd on cellWriteCursors, then
// write i into cellContents at cellOffsets[cellIdx] + cursor.
//
// Pre-condition: cellWriteCursors zeroed before dispatch (the JS wrapper
// emits encoder.clearBuffer immediately before this dispatch — reviewer
// 2b MEDIUM-1). K3b has populated cellOffsets. cellCounts intact (K4
// reads it for per-cell counts in the 3×3 neighborhood walk).
//
// CRITICAL — fix #2: same floor + float div + per-axis cell sizing as K3.
// Cells MUST agree with K3 — otherwise entities histogrammed into one
// cell scatter into a different one and K4 sees ghosts.
//
// INVARIANT — no barriers, no shared memory. Early return permitted.

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
  numCells:  u32,
  cellSizeX: f32,
  cellSizeY: f32,
  _pad0:     u32,
  _pad1:     u32,
}

const FLAG_ABSORBING: u32 = 1u;
const FLAG_TOMBSTONE: u32 = 8u;

@group(0) @binding(0) var<storage, read>             positions        : array<vec2f>;
@group(0) @binding(1) var<storage, read>             metas            : array<EntityMeta>;
@group(0) @binding(2) var<storage, read>             cellOffsets      : array<u32>;
@group(0) @binding(3) var<storage, read_write>       cellWriteCursors : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write>       cellContents     : array<u32>;
@group(0) @binding(5) var<uniform>                   params           : BroadphaseParams;

@compute @workgroup_size(256)
fn broadphase_scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.N) { return; }

  let flags = metas[i].flags;
  if ((flags & (FLAG_ABSORBING | FLAG_TOMBSTONE)) != 0u) { return; }

  let pos = positions[i];
  // Fix #2 — IDENTICAL formula to K3. If these diverge, K4 sees ghosts.
  let cx = min(u32(floor(pos.x / params.cellSizeX)), params.ncx - 1u);
  let cy = min(u32(floor(pos.y / params.cellSizeY)), params.ncy - 1u);
  let cellIdx = cy * params.ncx + cx;

  let insertOffset = atomicAdd(&cellWriteCursors[cellIdx], 1u);
  cellContents[cellOffsets[cellIdx] + insertOffset] = i;
}
