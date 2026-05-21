// physics-gpu-k5.js — K5a + K5 velocity solver. Phase 2d dark-launch.
// Writes to gpuVelScratchBuf (separate from gravityHandle.velocitiesBuf).
// CPU stepPBD's velocity solver still authoritative until 2g flip.

const WORKGROUP = 256;
const K5A_PARAMS_SIZE = 32;    // 8 × u32 — 2g: + tableSize, tableMask, 2×pad
const K5_PARAMS_SIZE  = 16;

export async function createK5GPU(device, wgslSources, gravityHandle, k4Handle, k8Handle) {
  const modK5a = device.createShaderModule({ label: 'K5a', code: wgslSources.k5a });
  const modK5  = device.createShaderModule({ label: 'K5',  code: wgslSources.k5  });

  const k5aLayout = device.createBindGroupLayout({ label: 'K5a bgl', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // contacts (rw)
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // metas
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // maxImpulse
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // pairCellMeta (K8 output)
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // pairCellFlags (K8 output)
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ]});
  const k5Layout = device.createBindGroupLayout({ label: 'K5 bgl', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ]});

  const [pipeK5a, pipeAccum, pipeApply] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'K5a', layout: device.createPipelineLayout({ bindGroupLayouts: [k5aLayout] }),
      compute: { module: modK5a, entryPoint: 'warm_start_calibrate' },
    }),
    device.createComputePipelineAsync({
      label: 'K5-accum', layout: device.createPipelineLayout({ bindGroupLayouts: [k5Layout] }),
      compute: { module: modK5, entryPoint: 'vs_accumulate' },
    }),
    device.createComputePipelineAsync({
      label: 'K5-apply', layout: device.createPipelineLayout({ bindGroupLayouts: [k5Layout] }),
      compute: { module: modK5, entryPoint: 'vs_apply' },
    }),
  ]);

  const k5aParamsBuf = device.createBuffer({ label: 'K5a params', size: K5A_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const k5ParamsBuf  = device.createBuffer({ label: 'K5 params',  size: K5_PARAMS_SIZE,  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  let capacity = 0;
  let velDeltaBuf, gpuVelScratchBuf, velStagingBuf, bgK5a, bgK5;
  const k5aScratch = new ArrayBuffer(K5A_PARAMS_SIZE);
  const k5aView    = new DataView(k5aScratch);
  const k5Scratch  = new ArrayBuffer(K5_PARAMS_SIZE);
  const k5View     = new DataView(k5Scratch);

  function destroyOwn() {
    if (velDeltaBuf)      { velDeltaBuf.destroy();      velDeltaBuf = null; }
    if (gpuVelScratchBuf) { gpuVelScratchBuf.destroy(); gpuVelScratchBuf = null; }
    if (velStagingBuf)    { velStagingBuf.destroy();    velStagingBuf = null; }
    bgK5a = null; bgK5 = null;
  }

  function allocateBuffers(cap) {
    destroyOwn();
    capacity = cap;
    velDeltaBuf = device.createBuffer({ label: 'velDelta', size: cap * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    gpuVelScratchBuf = device.createBuffer({ label: 'gpuVelScratch', size: cap * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    velStagingBuf = device.createBuffer({ label: 'gpuVelScratch staging', size: cap * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    rebuildBindGroups();
  }

  function rebuildBindGroups() {
    bgK5a = device.createBindGroup({ label: 'K5a bg', layout: k5aLayout, entries: [
      { binding: 0, resource: { buffer: k4Handle.contactsBuf } },
      { binding: 1, resource: { buffer: gravityHandle.metasBuf } },
      { binding: 2, resource: { buffer: k4Handle.maxImpulseBuf } },
      { binding: 3, resource: { buffer: k8Handle.cellMetaBuf } },
      { binding: 4, resource: { buffer: k8Handle.cellFlagsBuf } },
      { binding: 5, resource: { buffer: k5aParamsBuf } },
    ]});
    bgK5 = device.createBindGroup({ label: 'K5 bg', layout: k5Layout, entries: [
      { binding: 0, resource: { buffer: k4Handle.contactsBuf } },
      { binding: 1, resource: { buffer: gpuVelScratchBuf } },
      { binding: 2, resource: { buffer: velDeltaBuf } },
      { binding: 3, resource: { buffer: gravityHandle.metasBuf } },
      { binding: 4, resource: { buffer: k5ParamsBuf } },
    ]});
  }

  function growIfNeeded(N) {
    if (N > capacity) { allocateBuffers(Math.max(256, nextPow2(N))); return true; }
    if (capacity > 256 && N < capacity / 4) {
      const nc = Math.max(256, nextPow2(N));
      if (nc < capacity) { allocateBuffers(nc); return true; }
    }
    return false;
  }

  function uploadParams(N, contactCount, dt, G, e, epsilon) {
    k5aView.setUint32 ( 0, N >>> 0, true);
    k5aView.setUint32 ( 4, contactCount >>> 0, true);
    k5aView.setFloat32( 8, dt, true);
    k5aView.setFloat32(12, G, true);
    // 2g: tableSize/tableMask for pairImpulseTable lookup.
    const ts = k8Handle.tableSize;
    k5aView.setUint32 (16, ts >>> 0, true);
    k5aView.setUint32 (20, (ts > 0 ? ts - 1 : 0) >>> 0, true);
    // Bug-fix-2026-05-21: pass effectiveEpsilon for K5a's Plummer floor.
    // Without it, dense-cluster contacts seeded jGrav from rSum-only,
    // producing huge warm-start impulses that destabilized K5 Jacobi.
    k5aView.setFloat32(24, epsilon, true);
    k5aView.setUint32 (28, 0, true);
    device.queue.writeBuffer(k5aParamsBuf, 0, k5aScratch, 0, K5A_PARAMS_SIZE);
    k5View.setUint32 ( 0, N >>> 0, true);
    k5View.setUint32 ( 4, contactCount >>> 0, true);
    k5View.setFloat32( 8, dt, true);
    k5View.setFloat32(12, e, true);
    device.queue.writeBuffer(k5ParamsBuf, 0, k5Scratch, 0, K5_PARAMS_SIZE);
  }

  function recordDispatch(encoder, N, contactCount, iterCount) {
    if (N === 0) return;
    // ALWAYS seed scratch from K2's post-kick velocity. Even when there are
    // no contacts (contactCount===0), the backend reads back this staging
    // buffer and applies it to entities — without this copy the buffer
    // would hold stale/zero data and the readback would clobber the
    // entities' velocities. Phase 2g flip correctness.
    encoder.copyBufferToBuffer(gravityHandle.velocitiesBuf, 0, gpuVelScratchBuf, 0, N * 8);
    if (contactCount > 0) {
      const cWg = Math.ceil(contactCount / WORKGROUP);
      const eWg = Math.ceil(N / WORKGROUP);
      encoder.clearBuffer(velDeltaBuf, 0, capacity * 2 * 4);
      { const p = encoder.beginComputePass({ label: 'K5a' });
        p.setPipeline(pipeK5a); p.setBindGroup(0, bgK5a);
        p.dispatchWorkgroups(cWg); p.end(); }
      for (let iter = 0; iter < iterCount; iter++) {
        { const p = encoder.beginComputePass({ label: 'K5 accum' });
          p.setPipeline(pipeAccum); p.setBindGroup(0, bgK5);
          p.dispatchWorkgroups(cWg); p.end(); }
        { const p = encoder.beginComputePass({ label: 'K5 apply' });
          p.setPipeline(pipeApply); p.setBindGroup(0, bgK5);
          p.dispatchWorkgroups(eWg); p.end(); }
      }
    }
    encoder.copyBufferToBuffer(gpuVelScratchBuf, 0, velStagingBuf, 0, N * 8);
  }

  async function readbackVelocities(N) {
    if (capacity === 0 || N === 0) return new Float32Array(0);
    await velStagingBuf.mapAsync(GPUMapMode.READ, 0, N * 8);
    const mapped = velStagingBuf.getMappedRange(0, N * 8);
    const out = new Float32Array(mapped.byteLength / 4);
    out.set(new Float32Array(mapped));
    velStagingBuf.unmap();
    return out;
  }

  function destroy() {
    destroyOwn();
    try { k5aParamsBuf.destroy(); } catch {}
    try { k5ParamsBuf.destroy(); } catch {}
  }

  return {
    growIfNeeded, uploadParams, recordDispatch, readbackVelocities,
    onGravityRealloc() { if (capacity > 0) rebuildBindGroups(); },
    onK4Realloc()      { if (capacity > 0) rebuildBindGroups(); },
    onK8Realloc()      { if (capacity > 0) rebuildBindGroups(); },
    destroy,
    get capacity() { return capacity; },
    get gpuVelScratchBuf() { return gpuVelScratchBuf; },
    get velDeltaBuf()      { return velDeltaBuf; },
  };
}

function computeVelIter(count) {
  if (count <= 60)  return 8;
  if (count >= 380) return 24;
  return 8 + Math.floor((count - 60) * 16 / 320);
}
export { computeVelIter };

function nextPow2(n) {
  if (n <= 1) return 1;
  let v = n - 1;
  v |= v >>> 1; v |= v >>> 2; v |= v >>> 4; v |= v >>> 8; v |= v >>> 16;
  return (v + 1) >>> 0;
}
