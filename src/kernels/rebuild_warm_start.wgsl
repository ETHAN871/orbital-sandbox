// rebuild_warm_start.wgsl — K8: write converged contact impulses into
// pairImpulseTable for next substep's K5a warm-start lookup.
// Orbital Sandbox — WebGPU Physics, Phase 2f (dark-launch).
//
// Spec: docs/webgpu-blueprint.md §3 K8 row + §12 G8 + §4 pairImpulseTable.
//
// Translates physics.js:794-848 _rebuildPrevPairImpulses to GPU.
// Per-thread t < contactCount: read contacts[t]'s converged j (post-K5),
// compute symmetric key (lo=min(idxA,idxB), hi=max(...)), hash, linear-
// probe up to 8 slots claiming the first empty one via
// atomicCompareExchangeWeak on cell's occupancy flag (bit 0).
//
// Pre-condition: encoder.clearBuffer(cellFlags) before dispatch
// (blueprint §3 K8 ¶ — zero all occupancy bits each substep; no
// tombstone bit needed).
//
// Overflow: 8 probes all occupied → atomicOr STATUS_HASH_OVERFLOW;
// the impulse is dropped (K5a cold-starts next substep for that pair).
// Acceptable per blueprint §3.2 limitation.
//
// Cell split across two buffers: cellMeta (plain {keyA,keyB,j,nx,ny})
// + cellFlags (array<atomic<u32>>). WGSL forbids declaring a non-atomic
// struct member alongside an atomic one inside the SAME array<Struct>
// without forcing per-field atomic ops on every read. The two-buffer
// layout keeps occupancy atomic while letting data fields be plain
// reads/writes once the slot owner is established.
//
// Hash: (lo×1000003u) ^ (hi×999983u) — large near-decimal primes;
// better avalanche than 3D voxel hashes. Two u32 keys support N>65535
// (blueprint §12 G8 fork-2 resolution).

struct Contact {
  idxA: u32, idxB: u32, rSum: f32, dist: f32,
  nx: f32,   ny: f32,   vnApproach: f32, normalImpulse: f32,
  flags: u32, _p0: u32, _p1: u32, _p2: u32,
}

struct PairCellMeta { keyA: u32, keyB: u32, j: f32, nx: f32, ny: f32 }

struct K8Params {
  contactCount: u32,
  tableSize:    u32,    // power-of-two cell count
  tableMask:    u32,    // tableSize - 1
  _pad:         u32,
}

struct EntityMeta { mass: f32, chargeF: f32, radius: f32, flags: u32 }

const MAX_PROBE: u32 = 8u;
const FLAG_ABSORBING:       u32 = 1u;
const FLAG_TOMBSTONE:       u32 = 8u;
const STATUS_HASH_OVERFLOW: u32 = 1u;
const OCCUPIED: u32 = 1u;

@group(0) @binding(0) var<storage, read>       contacts     : array<Contact>;
@group(0) @binding(1) var<storage, read>       metas        : array<EntityMeta>;
@group(0) @binding(2) var<storage, read_write> cellMeta     : array<PairCellMeta>;
@group(0) @binding(3) var<storage, read_write> cellFlags    : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> statusFlags  : array<atomic<u32>>;
@group(0) @binding(5) var<uniform>             params       : K8Params;
// bug-fix-2026-05-23: live contact count from K4. See velocity_solver.wgsl.
@group(0) @binding(6) var<storage, read>       contactCount : array<u32, 1>;

@compute @workgroup_size(256)
fn rebuild_warm_start(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= contactCount[0]) { return; }
  let c = contacts[t];
  // Skip contacts whose final impulse converged to ≤0 — physics.js:798
  // (prevents unbounded table fill from brushing pairs).
  if (c.normalImpulse <= 0.0) { return; }
  let mA = metas[c.idxA];
  let mB = metas[c.idxB];
  if (((mA.flags | mB.flags) & (FLAG_ABSORBING | FLAG_TOMBSTONE)) != 0u) { return; }

  let lo = min(c.idxA, c.idxB);
  let hi = max(c.idxA, c.idxB);
  let hash = (lo * 1000003u) ^ (hi * 999983u);

  for (var probe: u32 = 0u; probe < MAX_PROBE; probe = probe + 1u) {
    let slot = (hash + probe) & params.tableMask;
    let result = atomicCompareExchangeWeak(&cellFlags[slot], 0u, OCCUPIED);
    if (result.exchanged) {
      // Slot claimed; we own it. Plain writes are race-free now.
      cellMeta[slot].keyA = lo;
      cellMeta[slot].keyB = hi;
      cellMeta[slot].j    = c.normalImpulse;
      cellMeta[slot].nx   = c.nx;
      cellMeta[slot].ny   = c.ny;
      return;
    }
    // Slot occupied by another writer; continue probing.
  }

  // All 8 probes occupied → drop + flag.
  atomicOr(&statusFlags[0], STATUS_HASH_OVERFLOW);
}
