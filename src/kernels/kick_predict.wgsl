// kick_predict.wgsl — K2: gravity kick + position predict + pseudoVel reset.
// Orbital Sandbox — WebGPU Physics, Phase 2a.
//
// Translates physics.js stepPBD steps A (pseudoVel reset), C (v += a·dt),
// D (x += v·dt). Per-entity, one thread per entity, no inter-thread coord.
// Workgroup_size 256 to match K1 and play nicely with the same buffer pool.
//
// INVARIANT — no barriers, no shared memory:
// K2 is a simple per-entity map kernel. Each thread owns exactly one entity
// (global_invocation_id.x == entity index) and writes to exactly one slot
// in positions[], velocities[], pseudoVels[]. There are NO workgroup
// barriers and NO var<workgroup> declarations. Do not add them — this
// kernel intentionally carries none of the cooperative-load complexity
// of K1. If you need inter-entity communication, it belongs in K4/K5/K6.
// Early `return`s are permitted ONLY because of this invariant.
//
// NaN guard (docs/webgpu-blueprint.md §3, G10): bad write → zero pos + zero
// vel + atomicAdd(nanCheckBuf, 1). CPU reads nanCheckBuf each frame; non-zero
// triggers swapToCpu('nan-propagation') + shadow restore (§6.3).
//
// Write pattern: K2 writes `velocities` IN-PLACE and a separate `outPositions`
// buffer. Reason: K1 reads `positions` and writes `accels`; if K2 also wrote
// `positions` in-place, downstream Phase 2b broadphase kernels would need
// to wait on K2's write before reading positions. With a separate
// `outPositions` we can chain K2 → K3 cleanly without buffer aliasing.
// Velocities have no downstream reader within the same substep, so in-place
// write is safe. Verified by reviewer round-2 audit (single-thread own-slot
// pattern, no cross-thread race).

struct EntityMeta {
  mass:    f32,
  chargeF: f32,
  radius:  f32,
  flags:   u32,
}

struct Params {
  N:   u32,
  dt:  f32,
  _p0: u32,   // pad to 16 B (uniform buffers must be 16-B aligned)
  _p1: u32,
}

const FLAG_ABSORBING: u32 = 1u;   // docs/webgpu-blueprint.md §4.1
const FLAG_PINNED:    u32 = 2u;

@group(0) @binding(0) var<storage, read>             positions    : array<vec2f>;
@group(0) @binding(1) var<storage, read_write>       velocities   : array<vec2f>;
@group(0) @binding(2) var<storage, read>             accels       : array<vec2f>;
@group(0) @binding(3) var<storage, read>             metas        : array<EntityMeta>;
@group(0) @binding(4) var<storage, read_write>       outPositions : array<vec2f>;
@group(0) @binding(5) var<storage, read_write>       pseudoVels   : array<vec2f>;
@group(0) @binding(6) var<storage, read_write>       nanCheckBuf  : array<atomic<u32>>;
@group(0) @binding(7) var<uniform>                   params       : Params;

@compute @workgroup_size(256)
fn kick_predict(
  @builtin(global_invocation_id) gid: vec3u,
) {
  let i  = gid.x;
  let N  = params.N;
  let dt = params.dt;

  if (i >= N) { return; }

  // Step A: pseudoVel reset fires unconditionally for all live entities,
  // including PINNED and ABSORBING — solver kernels in later phases may
  // accumulate into these slots and would carry stale values from the
  // prior substep without this clear.
  pseudoVels[i] = vec2f(0.0, 0.0);

  let flags = metas[i].flags;
  let inert = (flags & (FLAG_ABSORBING | FLAG_PINNED)) != 0u;

  if (inert) {
    outPositions[i] = positions[i];
    return;
  }

  let a = accels[i];
  var vel = velocities[i] + a * dt;       // Step C: v += a·dt
  var pos = positions[i]  + vel * dt;     // Step D: x += v·dt

  // WGSL has no isNan() intrinsic guaranteed by spec as of 2026-Q2.
  // The (x != x) form is the canonical IEEE-754 NaN test and works on
  // every conforming WebGPU implementation. Compilers may fold (x != x)
  // away under fast-math; the spec disallows that here because storage
  // buffer writes are observable side-effects.
  if (pos.x != pos.x || pos.y != pos.y || vel.x != vel.x || vel.y != vel.y) {
    pos = vec2f(0.0, 0.0);
    vel = vec2f(0.0, 0.0);
    atomicAdd(&nanCheckBuf[0], 1u);
  }

  velocities[i]   = vel;
  outPositions[i] = pos;
}
