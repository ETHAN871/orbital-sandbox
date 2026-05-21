// gravity_accel.wgsl — K1: tiled all-pairs gravity acceleration kernel.
// Orbital Sandbox — WebGPU Physics, Phase 1.
//
// Force law (matches physics.js computeAccelerations lines 87–123 exactly):
//   For each entity i, accumulate force from every j ≠ i:
//     skip if j is absorbing (flags & FLAG_ABSORBING)
//     skip if j.charge == 0 (asymmetric force model — A receives force from
//                            B iff B.charge ≠ 0; sign carried by charge)
//     min-image wrap delta (dx, dy) per axis
//     Plummer soften: r² = dx² + dy² + max(rA+rB, epsilon)²
//     accel += b.charge · G · b.mass / r² · (dx, dy) / r
//
// Workgroup = 256 threads. Each thread owns entity i = global_invocation_id.x
// and writes to outputBuf[i] exactly once.
//
// Tiled inner loop: each tile cooperatively loads 256 source bodies into
// workgroup-shared memory via `var<workgroup>`, then all 256 threads in the
// workgroup reuse that tile for their inner loop. This amortizes the global
// memory load cost — at N=5,000, each EntityMeta is loaded once per tile,
// not once per pair.
//
// ────────────────────────────────────────────────────────────────────
// IMPLEMENTATION REQUIREMENT — barrier guards
// ────────────────────────────────────────────────────────────────────
// Absorbing-thread short-circuit and out-of-bounds (i >= N) handling MUST
// use boolean flags (i_absorbing, i_oob) to gate the accumulate body, NOT
// early `return` statements. Any `return` before workgroupBarrier() in this
// kernel will cause undefined behavior — the barrier requires ALL active
// invocations in the workgroup to reach it; an early return removes the
// thread from the active set.
//
// Correct pattern:
//   let i_absorbing = (mi.flags & FLAG_ABSORBING) != 0u;
//   for (var tile = ...) {
//     // cooperative load (all threads participate, even out-of-bounds)
//     workgroupBarrier();
//     if (i < N && !i_absorbing) { /* accumulate */ }
//     workgroupBarrier();
//   }
//   if (i < N && !i_absorbing) { outputBuf[i] = vec2f(lax, lay); }
//   else if (i < N)            { outputBuf[i] = vec2f(0.0); }
//   // NO early return anywhere above.
// ────────────────────────────────────────────────────────────────────
//
// Floating-point precision: f32 throughout. Per-substep accumulated error
// vs CPU's float64 path bounded at ~N×ε_f32 ≈ 6×10⁻⁴ relative at N=5000.
// Documented as expected fp32/fp64 divergence in docs/webgpu-blueprint.md §8.

// EntityMeta packed by physics-gpu-gravity.js. Layout:
//   mass (f32) | chargeF (f32) | radius (f32) | flags (u32)
// Total 16 bytes, aligned to 4 (WGSL §3.6.2 — struct align = max member align).
struct EntityMeta {
  mass:    f32,
  chargeF: f32,
  radius:  f32,
  flags:   u32,
}

// Per-dispatch parameters. Uniform buffer; 16-byte aligned (padded to 32).
struct Params {
  N:        u32,
  G:        f32,
  epsilon:  f32,
  W:        f32,   // viewport width  — 0 means no wrap
  H:        f32,   // viewport height — 0 means no wrap
  _pad0:    u32,
  _pad1:    u32,
  _pad2:    u32,
}

const FLAG_ABSORBING: u32 = 1u;   // bit 0 — see docs/webgpu-blueprint.md §4.1
const WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read>       positions : array<vec2f>;
@group(0) @binding(1) var<storage, read>       metas     : array<EntityMeta>;
@group(0) @binding(2) var<storage, read_write> outputBuf : array<vec2f>;
@group(0) @binding(3) var<uniform>             params    : Params;

// Workgroup-shared tile storage. 256 × 8 bytes = 2 KB for positions, plus
// 256 × 16 bytes = 4 KB for metadata. Total 6 KB — well under the WGSL
// spec minimum of 16 KB (maxComputeWorkgroupStorageSize).
var<workgroup> tile_pos  : array<vec2f,      256>;
var<workgroup> tile_meta : array<EntityMeta, 256>;

@compute @workgroup_size(256)
fn gravity_accel(
  @builtin(global_invocation_id)   gid : vec3u,
  @builtin(local_invocation_index) lid : u32,
) {
  let i     = gid.x;
  let N     = params.N;
  let G     = params.G;
  let eps   = params.epsilon;
  let W     = params.W;
  let H     = params.H;
  let halfW = W * 0.5;
  let halfH = H * 0.5;
  let wrap  = W > 0.0;

  // Capture this thread's entity (or sentinel if out-of-bounds).
  // i_oob and i_absorbing gate the accumulate body — NO early return.
  let i_oob = i >= N;
  var pi : vec2f      = vec2f(0.0, 0.0);
  var mi : EntityMeta = EntityMeta(0.0, 0.0, 0.0, 0u);
  if (!i_oob) {
    pi = positions[i];
    mi = metas[i];
  }
  let i_absorbing = (mi.flags & FLAG_ABSORBING) != 0u;
  let i_active    = !i_oob && !i_absorbing;

  // Thread-local accumulator. Read only by this thread; no atomics needed.
  var lax : f32 = 0.0;
  var lay : f32 = 0.0;

  let numTiles = (N + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

  for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
    // ── Cooperative tile load ──────────────────────────────────────
    // Each thread loads one source entity into shared memory. Threads
    // with src >= N load a zero-mass dummy so absent slots contribute
    // no force. All threads (including i_oob and i_absorbing) participate
    // in the load — required for workgroupBarrier correctness below.
    let src = t * WORKGROUP_SIZE + lid;
    if (src < N) {
      tile_pos[lid]  = positions[src];
      tile_meta[lid] = metas[src];
    } else {
      tile_pos[lid]  = vec2f(0.0, 0.0);
      tile_meta[lid] = EntityMeta(0.0, 0.0, 0.0, 0u);
    }
    workgroupBarrier();

    // ── Accumulate forces from this tile onto entity i ─────────────
    if (i_active) {
      for (var k : u32 = 0u; k < WORKGROUP_SIZE; k = k + 1u) {
        let j = t * WORKGROUP_SIZE + k;
        if (j >= N) { continue; }
        if (j == i) { continue; }   // self-skip

        let mj = tile_meta[k];
        // Skip absorbing sources (physics.js:89 — `if (b.absorbing) continue`).
        if ((mj.flags & FLAG_ABSORBING) != 0u) { continue; }
        // Skip charge-zero sources — asymmetric force model:
        // A receives force from B iff B.charge ≠ 0 (physics.js:113–116).
        if (mj.chargeF == 0.0) { continue; }

        var dx = tile_pos[k].x - pi.x;
        var dy = tile_pos[k].y - pi.y;

        // Min-image PBC (physics.js:92–95). Matches the CPU direct path
        // exactly. Note: NOT 9-ghost summation — see blueprint §3.1 for
        // the BH path's eliminated 9-ghost issue.
        if (wrap) {
          if (dx >  halfW) { dx = dx - W; }
          else if (dx < -halfW) { dx = dx + W; }
          if (dy >  halfH) { dy = dy - H; }
          else if (dy < -halfH) { dy = dy + H; }
        }

        let r2Raw = dx * dx + dy * dy;
        // Plummer softening (physics.js:107): r² = r²_raw + minR² where
        // minR = max(rA + rB, epsilon). The max() prevents division by
        // zero / Plummer collapse when bodies converge to the same point.
        let rSum = mi.radius + mj.radius;
        let minR = max(rSum, eps);
        let r2   = r2Raw + minR * minR;
        let r    = sqrt(r2);

        // Force magnitude (physics.js:114): mag = b.chargeF · G · b.mass / r²
        let mag = mj.chargeF * G * mj.mass / r2;
        lax = lax + mag * dx / r;
        lay = lay + mag * dy / r;
      }
    }
    // Barrier before the next tile's load so no thread starts loading the
    // next tile while another thread is still reading the current tile.
    workgroupBarrier();
  }

  // ── Write final output ─────────────────────────────────────────────
  // i_oob threads write nothing (their outputBuf slot doesn't exist).
  // i_absorbing threads write zero (they receive no gravity).
  // Active threads write their accumulated (lax, lay).
  if (!i_oob) {
    if (i_active) {
      outputBuf[i] = vec2f(lax, lay);
    } else {
      // i_absorbing case
      outputBuf[i] = vec2f(0.0, 0.0);
    }
  }
}
