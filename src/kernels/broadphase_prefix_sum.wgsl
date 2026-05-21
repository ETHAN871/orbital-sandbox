// broadphase_prefix_sum.wgsl — K3b: exclusive prefix sum over cellCounts.
// Orbital Sandbox — WebGPU Physics, Phase 2b.
//
// Three-dispatch design (chosen over single-pass to respect WGSL's
// portable maxComputeInvocationsPerWorkgroup=1024 limit; RTX 5070 Ti caps
// at 1024 under WGSL even though the hardware supports more):
//
//   1. block_scan  — each workgroup (1024 threads) does an exclusive
//                    Blelloch scan over its 1024-cell block. Writes the
//                    in-block exclusive scan to cellOffsets[]. ALSO writes
//                    the block's total sum to blockTotals[wg.x].
//   2. spine_scan  — single workgroup (≤ 4 threads for ≤ 4096 cells)
//                    exclusively scans blockTotals[] in place.
//   3. apply_spine — each workgroup adds spine[wg.x] to every cell in its
//                    block, completing the global exclusive scan.
//
// At numCells ≤ 1024 spine has a single entry (always 0 for exclusive
// scan over one value) so apply_spine is structurally a no-op. We still
// dispatch it unconditionally per reviewer 2b LOW-1 (one fewer JS branch
// to maintain). 3 dispatches × ≤ 4 workgroups each is sub-millisecond.
//
// HIGH-1 fix (reviewer 2b round 1): block_scan captures the last cell's
// ORIGINAL value BEFORE Blelloch overwrites shared memory, then writes
// `blockTotals[wg] = scratch[N-1] + savedLastInput`. Without this the
// block total would be missing the last cell's own count.
//
// INVARIANT — block_scan + apply_spine use shared memory + barriers.
// No early returns after the first barrier (would skip subsequent
// barriers and deadlock other threads). spine_scan has no barriers so
// early return is permitted.

struct ScanParams {
  numCells:  u32,
  numBlocks: u32,    // ceil(numCells / 1024) — uploaded by JS
  _pad0:     u32,
  _pad1:     u32,    // 16 B
}

@group(0) @binding(0) var<storage, read>             cellCounts  : array<u32>;
@group(0) @binding(1) var<storage, read_write>       cellOffsets : array<u32>;
@group(0) @binding(2) var<storage, read_write>       blockTotals : array<u32>;
@group(0) @binding(3) var<uniform>                   params      : ScanParams;

const BLOCK_SIZE: u32 = 1024u;

var<workgroup> scratch : array<u32, 1024>;

// ── Dispatch 1: block_scan ─────────────────────────────────────────
@compute @workgroup_size(1024)
fn block_scan(
  @builtin(workgroup_id)           wg  : vec3u,
  @builtin(local_invocation_index) lid : u32,
) {
  let blockBase = wg.x * BLOCK_SIZE;
  let cellIdx   = blockBase + lid;
  let inRange   = cellIdx < params.numCells;

  // Load (or 0-pad past numCells) into shared.
  scratch[lid] = select(0u, cellCounts[cellIdx], inRange);
  workgroupBarrier();

  // HIGH-1 reviewer fix: capture last cell's ORIGINAL input BEFORE the
  // in-place Blelloch overwrites scratch[BLOCK_SIZE-1]. blockTotals
  // needs this to be the true reduction (exclusive-scan-output[N-1]
  // misses N-1's own input).
  let lastOriginal = scratch[BLOCK_SIZE - 1u];
  workgroupBarrier();

  // Blelloch up-sweep (reduce).
  var stride: u32 = 1u;
  loop {
    if (stride >= BLOCK_SIZE) { break; }
    let idx = (lid + 1u) * stride * 2u - 1u;
    if (idx < BLOCK_SIZE) {
      scratch[idx] = scratch[idx] + scratch[idx - stride];
    }
    workgroupBarrier();
    stride = stride * 2u;
  }

  // Zero last for exclusive scan.
  if (lid == 0u) { scratch[BLOCK_SIZE - 1u] = 0u; }
  workgroupBarrier();

  // Blelloch down-sweep.
  stride = BLOCK_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    let idx = (lid + 1u) * stride * 2u - 1u;
    if (idx < BLOCK_SIZE) {
      let tmp = scratch[idx - stride];
      scratch[idx - stride] = scratch[idx];
      scratch[idx] = scratch[idx] + tmp;
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (inRange) { cellOffsets[cellIdx] = scratch[lid]; }

  // Write block reduction: exclusive-scan[N-1] + original[N-1].
  if (lid == 0u) {
    blockTotals[wg.x] = scratch[BLOCK_SIZE - 1u] + lastOriginal;
  }
}

// ── Dispatch 2: spine_scan ─────────────────────────────────────────
// Exclusively scan blockTotals[0..numBlocks). Trivial since numBlocks ≤ 4.
@compute @workgroup_size(4)
fn spine_scan(
  @builtin(local_invocation_index) lid : u32,
) {
  if (lid >= params.numBlocks) { return; }
  // Serial scan from thread 0 only — numBlocks ≤ 4 so no parallelism gain.
  if (lid == 0u) {
    var running: u32 = 0u;
    for (var i: u32 = 0u; i < params.numBlocks; i = i + 1u) {
      let t = blockTotals[i];
      blockTotals[i] = running;
      running = running + t;
    }
  }
}

// ── Dispatch 3: apply_spine ────────────────────────────────────────
// Add blockTotals[wg.x] to every cell in this block.
@compute @workgroup_size(1024)
fn apply_spine(
  @builtin(workgroup_id)           wg  : vec3u,
  @builtin(local_invocation_index) lid : u32,
) {
  let cellIdx = wg.x * BLOCK_SIZE + lid;
  if (cellIdx >= params.numCells) { return; }
  cellOffsets[cellIdx] = cellOffsets[cellIdx] + blockTotals[wg.x];
}
