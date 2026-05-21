// pseudo_integrate_boundary.wgsl — K7: pseudo-velocity integration + wrap boundary + pinned hard-reset.
// Orbital Sandbox — WebGPU Physics, Phase 2a.
//
// Translates physics.js _solvePositionConstraints end-step (lines 411-416)
// + applyBoundary wrap mode + stepPBD step I (pinned reset, lines 303-306).
// Destroy boundary stays CPU-only (entity array splice).
// Per-entity, one thread per entity, no inter-thread coordination.
//
// INVARIANT — no barriers, no shared memory:
// K7 is a per-entity map kernel. Each thread owns exactly one entity and
// writes to exactly one slot in positions[] and velocities[]. NO
// workgroup barriers, NO var<workgroup>. If a future shared-memory
// optimization is added (e.g. tiled pseudoVel prefetch), all early
// returns below MUST be converted to boolean gate variables matching
// the K1 pattern. See gravity_accel.wgsl §22-43 for that pattern.
//
// 2a status: written and unit-tested, but NOT wired into the substep
// pipeline yet. Wiring deferred to sub-phase 2e (when K6 lands and
// pseudoVels become GPU-resident — avoids round-trip uploads in 2a).
//
// 2e prerequisite: when K7 enters the substep loop, the CPU-side
// `applyBoundary` wrap path MUST be gated by `backend.name === 'webgpu'`.
// K7's toroidal modulo wrap and CPU's single-step if/add wrap would
// otherwise double-wrap (see blueprint §12 G2 follow-up).
//
// NaN guard (docs/webgpu-blueprint.md §3, G10): checks BOTH `pos` (the
// computed result after pseudo-vel integration) and `vel` (passed through
// unmodified — but still propagates if input was NaN). Either NaN → zero
// + atomicAdd(nanCheckBuf, 1).

struct EntityMeta {
  mass:    f32,
  chargeF: f32,
  radius:  f32,
  flags:   u32,
}

struct Params {
  N:  u32,
  dt: f32,
  W:  f32,   // viewport width  — 0 means no wrap (destroy mode handled CPU-side)
  H:  f32,   // viewport height — 0 means no wrap
}

const FLAG_ABSORBING: u32 = 1u;   // docs/webgpu-blueprint.md §4.1
const FLAG_PINNED:    u32 = 2u;

@group(0) @binding(0) var<storage, read_write>       positions   : array<vec2f>;
@group(0) @binding(1) var<storage, read_write>       velocities  : array<vec2f>;
@group(0) @binding(2) var<storage, read>             pseudoVels  : array<vec2f>;
@group(0) @binding(3) var<storage, read>             metas       : array<EntityMeta>;
@group(0) @binding(4) var<storage, read_write>       nanCheckBuf : array<atomic<u32>>;
@group(0) @binding(5) var<uniform>                   params      : Params;

@compute @workgroup_size(256)
fn pseudo_integrate_boundary(
  @builtin(global_invocation_id) gid: vec3u,
) {
  let i = gid.x;
  let N = params.N;
  if (i >= N) { return; }

  let flags = metas[i].flags;

  // PINNED: hard-reset velocity to zero; position unchanged.
  // Translates physics.js:303-306 (step I).
  if ((flags & FLAG_PINNED) != 0u) {
    velocities[i] = vec2f(0.0, 0.0);
    return;
  }

  // ABSORBING: CPU drives the shrink animation; GPU passes through.
  // (CPU writes the entity's position directly during animation lerp.)
  if ((flags & FLAG_ABSORBING) != 0u) {
    return;
  }

  // Integrate pseudo-velocity into position (split-impulse pattern;
  // real velocity is never modified by the position pass).
  var pos = positions[i] + pseudoVels[i] * params.dt;
  let vel = velocities[i];

  // NaN guard: pos (just computed) OR vel (pass-through; possibly stale NaN).
  // WGSL has no isNan() intrinsic guaranteed; (x != x) is the canonical
  // IEEE-754 test that works under all conforming compilers.
  let nanPos = pos.x != pos.x || pos.y != pos.y;
  let nanVel = vel.x != vel.x || vel.y != vel.y;
  if (nanPos || nanVel) {
    positions[i]  = vec2f(0.0, 0.0);
    velocities[i] = vec2f(0.0, 0.0);
    atomicAdd(&nanCheckBuf[0], 1u);
    return;
  }

  // Wrap boundary — toroidal modulo. W=H=0 means no-wrap (destroy mode
  // handled by CPU applyBoundary after readback).
  let W = params.W;
  let H = params.H;
  if (W > 0.0 && H > 0.0) {
    // ((x % W) + W) % W — robust for any signed input. WGSL `%` follows
    // C truncated division (can return negative); the double-modulo
    // form normalizes to [0, W). For pos.x exactly W: W % W = 0,
    // (0 + W) % W = 0. For pos.x slightly negative: (-eps % W) + W
    // = -eps + W, then % W ≈ W - eps. Both correct.
    pos.x = ((pos.x % W) + W) % W;
    pos.y = ((pos.y % H) + H) % H;
  }

  positions[i] = pos;
  // velocity unmodified — pass-through (writing back would be a no-op).
}
