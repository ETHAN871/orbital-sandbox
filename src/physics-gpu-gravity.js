// physics-gpu-gravity.js — K1 (gravity_accel) GPU pipeline wrapper.
//
// Owns the WebGPU resources for the K1 compute kernel: bind-group layout,
// compute pipeline, positions / EntityMeta / output buffers, double-buffered
// readback staging, Params uniform. Exposed through a small handle used by
// physics-backend.js.
//
// Phase 1 dispatch model (architect decision A2.α — see docs/webgpu-blueprint.md
// §5 + the architect transcript): pipelined per-substep. After each substep's
// `applyBoundary`, the JS substep loop submits a fresh K1 dispatch with the
// just-finalized positions and starts a `mapAsync` readback on the next
// staging slot. The promise is awaited at the START of the *following*
// substep — by which time GPU + IPC have overlapped with the CPU's
// collision + solver work. The accels GPU produces are computed from the
// exact positions the CPU would read at substep boundary, so there is NO
// staleness (only the documented fp32-vs-fp64 divergence).
//
// Buffer sizing: capacity tracked as `nextPow2(N)` to amortize realloc
// across entity churn. Buffers reallocated when N exceeds capacity or
// drops below capacity / 4 (architect: avoid alloc/free thrash on
// single-add / single-remove patterns). The owning backend resets its
// priming flag whenever this wrapper grows or shrinks — invalidates any
// in-flight mapAsync against the destroyed buffers.

const WORKGROUP_SIZE = 256;

// Per-EntityMeta byte stride. Layout locked to docs/webgpu-blueprint.md §4.1:
//   mass (f32) | chargeF (f32) | radius (f32) | flags (u32) → 16 B.
const META_STRIDE = 16;

// vec2f position stride.
const POS_STRIDE = 8;

// Params uniform stride. Struct from gravity_accel.wgsl:
//   N (u32) | G (f32) | epsilon (f32) | W (f32) | H (f32) | 3 × pad (u32) → 32 B.
const PARAMS_BYTE_SIZE = 32;

// vec2f accel output stride.
const ACCEL_STRIDE = 8;

// Bit values match gravity_accel.wgsl FLAG_ABSORBING (and prepare for future
// Phase 2 kernels that read the rest). See docs/webgpu-blueprint.md §4.1.
export const FLAG_ABSORBING = 1 >>> 0;
export const FLAG_PINNED    = 2 >>> 0;
export const FLAG_IS_BH     = 4 >>> 0;
export const FLAG_TOMBSTONE = 8 >>> 0;

function nextPow2(n) {
  if (n <= 1) return 1;
  let v = n - 1;
  v |= v >>> 1; v |= v >>> 2; v |= v >>> 4; v |= v >>> 8; v |= v >>> 16;
  return (v + 1) >>> 0;
}

// Construct K1 wrapper. Caller passes the WebGPU device + the compiled WGSL
// source string (loaded via gpu-init.loadKernel). Returns a handle with the
// substep-loop API consumed by physics-backend.js.
export async function createGravityGPU(device, wgslSource) {
  const shaderModule = device.createShaderModule({
    label: 'K1 gravity_accel module',
    code: wgslSource,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'K1 bind group layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // positions
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // metas
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // outputBuf
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
    ],
  });

  const pipeline = device.createComputePipeline({
    label: 'K1 gravity_accel pipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: 'gravity_accel' },
  });

  const paramsBuf = device.createBuffer({
    label: 'K1 params uniform',
    size: PARAMS_BYTE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ── Mutable buffer set (reallocated on capacity change) ────────────
  let capacity = 0;
  let positionsBuf, metasBuf, outputBuf;
  let stagingBufs;          // length-2 ping-pong
  let bindGroup;            // re-created each time capacity changes

  // Reusable CPU-side typed arrays for upload. Re-allocated on grow.
  let posScratch;           // Float32Array, 2 × capacity
  let metaScratch;          // ArrayBuffer + DataView, capacity × META_STRIDE
  let metaScratchView;
  const paramsScratch = new ArrayBuffer(PARAMS_BYTE_SIZE);
  const paramsView    = new DataView(paramsScratch);

  function destroyAllocBuffers() {
    if (positionsBuf) { positionsBuf.destroy(); positionsBuf = null; }
    if (metasBuf)     { metasBuf.destroy();     metasBuf     = null; }
    if (outputBuf)    { outputBuf.destroy();    outputBuf    = null; }
    if (stagingBufs) {
      for (const sb of stagingBufs) { try { sb.destroy(); } catch {} }
      stagingBufs = null;
    }
    bindGroup = null;
  }

  function allocateBuffers(newCapacity) {
    destroyAllocBuffers();
    capacity = newCapacity;
    positionsBuf = device.createBuffer({
      label: 'K1 positions',
      size: capacity * POS_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    metasBuf = device.createBuffer({
      label: 'K1 metas',
      size: capacity * META_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    outputBuf = device.createBuffer({
      label: 'K1 outputBuf accels',
      size: capacity * ACCEL_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    stagingBufs = [
      device.createBuffer({
        label: 'K1 staging A',
        size: capacity * ACCEL_STRIDE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      device.createBuffer({
        label: 'K1 staging B',
        size: capacity * ACCEL_STRIDE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
    ];
    bindGroup = device.createBindGroup({
      label: 'K1 bind group',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionsBuf } },
        { binding: 1, resource: { buffer: metasBuf } },
        { binding: 2, resource: { buffer: outputBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    });
    posScratch     = new Float32Array(capacity * 2);
    metaScratch    = new ArrayBuffer(capacity * META_STRIDE);
    metaScratchView = new DataView(metaScratch);
  }

  // Grow if N > capacity, shrink if N < capacity / 4 (and capacity > 256).
  // Returns true if reallocation occurred — caller must reset priming.
  function growIfNeeded(N) {
    if (N > capacity) {
      const newCap = Math.max(256, nextPow2(N));
      allocateBuffers(newCap);
      return true;
    }
    if (capacity > 256 && N < capacity / 4) {
      // Floor at 256 (nextPow2(0)=1 would otherwise drop to a 1-slot buffer).
      // Lets a cleared scene reclaim memory down to the minimum instead of
      // keeping a 4096-entity buffer alive for the rest of the session.
      const newCap = Math.max(256, nextPow2(N));
      if (newCap < capacity) {
        allocateBuffers(newCap);
        return true;
      }
    }
    return false;
  }

  function uploadPositions(entities, N) {
    for (let i = 0; i < N; i++) {
      const e = entities[i];
      posScratch[i * 2]     = e.x;
      posScratch[i * 2 + 1] = e.y;
    }
    device.queue.writeBuffer(
      positionsBuf, 0,
      posScratch.buffer, posScratch.byteOffset,
      N * POS_STRIDE,
    );
  }

  function uploadMetaAll(entities, N) {
    for (let i = 0; i < N; i++) packMetaInto(metaScratchView, i * META_STRIDE, entities[i]);
    device.queue.writeBuffer(metasBuf, 0, metaScratch, 0, N * META_STRIDE);
  }

  // Patch a single EntityMeta slot. Used when `beginAbsorption` toggles
  // FLAG_ABSORBING on a specific entity between dispatches — see the
  // architect risk #2: the meta upload must precede the next dispatch
  // or that entity still contributes gravity for one substep on GPU
  // while CPU has already marked it absorbing.
  function uploadMetaOne(entity, idx) {
    if (!metasBuf) return;
    const buf = new ArrayBuffer(META_STRIDE);
    const view = new DataView(buf);
    packMetaInto(view, 0, entity);
    device.queue.writeBuffer(metasBuf, idx * META_STRIDE, buf, 0, META_STRIDE);
  }

  function packMetaInto(view, byteOffset, e) {
    const charge = e.charge;
    // Charge must be exactly one of the discrete sentinel values; the WGSL
    // kernel skips sources where `chargeF == 0.0` (asymmetric force model)
    // and any near-zero float would silently corrupt that gate.
    if (charge !== -1 && charge !== 0 && charge !== 1) {
      throw new Error(`physics-gpu-gravity: entity.charge must be -1, 0, or 1 (got ${charge})`);
    }
    let flags = 0;
    if (e.absorbing) flags |= FLAG_ABSORBING;   // truthy check matches physics.js:86
    if (e.pinned)    flags |= FLAG_PINNED;
    if (e.type === 'black_hole') flags |= FLAG_IS_BH;
    view.setFloat32(byteOffset + 0,  e.mass,     true);
    view.setFloat32(byteOffset + 4,  charge,     true);
    view.setFloat32(byteOffset + 8,  e.radius,   true);
    view.setUint32 (byteOffset + 12, flags >>> 0, true);
  }

  function uploadParams(N, G, epsilon, W, H) {
    paramsView.setUint32 (0,  N >>> 0, true);
    paramsView.setFloat32(4,  G,       true);
    paramsView.setFloat32(8,  epsilon, true);
    paramsView.setFloat32(12, W,       true);
    paramsView.setFloat32(16, H,       true);
    paramsView.setUint32 (20, 0, true);
    paramsView.setUint32 (24, 0, true);
    paramsView.setUint32 (28, 0, true);
    device.queue.writeBuffer(paramsBuf, 0, paramsScratch, 0, PARAMS_BYTE_SIZE);
  }

  // Encode dispatch + copy outputBuf into stagingBufs[stagingIdx].
  // Caller submits the encoder. Caller then calls readbackStaging(stagingIdx).
  function recordDispatch(encoder, N, stagingIdx) {
    const pass = encoder.beginComputePass({ label: 'K1 pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const groups = Math.ceil(N / WORKGROUP_SIZE);
    pass.dispatchWorkgroups(groups);
    pass.end();
    encoder.copyBufferToBuffer(
      outputBuf, 0,
      stagingBufs[stagingIdx], 0,
      N * ACCEL_STRIDE,
    );
  }

  // Schedule mapAsync READ on the staging buffer. Returns a Promise that
  // resolves to a Float32Array (length = 2 × N) holding interleaved
  // (ax, ay) pairs. The staging buffer is unmapped before the returned
  // array is yielded so the next dispatch's copyBufferToBuffer can target
  // the same slot — we return a *copy* of the mapped range.
  async function readbackStaging(stagingIdx, N) {
    const staging = stagingBufs[stagingIdx];
    await staging.mapAsync(GPUMapMode.READ, 0, N * ACCEL_STRIDE);
    const mapped = staging.getMappedRange(0, N * ACCEL_STRIDE);
    const out = new Float32Array(mapped.byteLength / 4);
    out.set(new Float32Array(mapped));
    staging.unmap();
    return out;
  }

  function destroy() {
    destroyAllocBuffers();
    if (paramsBuf) paramsBuf.destroy();
  }

  return {
    growIfNeeded,
    uploadPositions,
    uploadMetaAll,
    uploadMetaOne,
    uploadParams,
    recordDispatch,
    readbackStaging,
    destroy,
    get capacity() { return capacity; },
  };
}
