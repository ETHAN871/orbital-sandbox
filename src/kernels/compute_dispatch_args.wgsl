// compute_dispatch_args.wgsl — 1-workgroup post-K4 kernel.
// Reads K4's `contactCount` (atomicAdd-counted by K4 main pass) and writes
// the dispatch-indirect args [ceil(count/256), 1, 1] used by K5/K5a/K6/K8
// for their per-contact accumulate passes.
//
// Replaces the prior "use prevContactCount on CPU side" pattern which
// caused either (a) stale slot processing when prev>current, or (b) new
// contacts skipped when prev<current. With indirect dispatch the per-
// contact passes get exactly ceil(current/256) workgroups, no more no
// less, computed on GPU within the same encoder.
//
// Per-entity passes (vs_apply, ps_apply) still use direct
// dispatchWorkgroups(ceil(N/256)) because N is known on CPU before submit.

@group(0) @binding(0) var<storage, read>       contactCount : array<u32, 1>;
@group(0) @binding(1) var<storage, read_write> dispatchArgs : array<u32, 3>;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(1)
fn compute_dispatch_args() {
  let n  = contactCount[0];
  let wg = (n + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  dispatchArgs[0] = wg;
  dispatchArgs[1] = 1u;
  dispatchArgs[2] = 1u;
}
