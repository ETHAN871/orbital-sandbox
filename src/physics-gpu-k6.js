// physics-gpu-k6.js — K6 position solver. Phase 2e dark-launch.
// Mirrors physics-gpu-k5.js structurally; reads positions (K2's
// outPositions) instead of velocities; writes pseudoVels.

const WORKGROUP = 256;
const K6_PARAMS_SIZE = 16;
const POS_ITERATIONS = 3;

export async function createK6GPU(device, wgslSource, gravityHandle, k2Handle, k4Handle) {
  const mod = device.createShaderModule({ label: 'K6 position_solver', code: wgslSource });
  const layout = device.createBindGroupLayout({ label: 'K6 bgl', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ]});

  const [pipeAccum, pipeApply] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'K6-accum', layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: mod, entryPoint: 'ps_accumulate' },
    }),
    device.createComputePipelineAsync({
      label: 'K6-apply', layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: mod, entryPoint: 'ps_apply' },
    }),
  ]);

  const paramsBuf = device.createBuffer({ label: 'K6 params', size: K6_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const scratch = new ArrayBuffer(K6_PARAMS_SIZE);
  const view = new DataView(scratch);

  let capacity = 0;
  let pvDeltaBuf, gpuPVScratchBuf, bindGroup;

  function destroyOwn() {
    if (pvDeltaBuf)      { pvDeltaBuf.destroy();      pvDeltaBuf = null; }
    if (gpuPVScratchBuf) { gpuPVScratchBuf.destroy(); gpuPVScratchBuf = null; }
    bindGroup = null;
  }

  function allocateBuffers(cap) {
    destroyOwn();
    capacity = cap;
    pvDeltaBuf = device.createBuffer({ label: 'pvDelta', size: cap * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    gpuPVScratchBuf = device.createBuffer({ label: 'gpuPVScratch', size: cap * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    rebuildBindGroup();
  }

  function rebuildBindGroup() {
    bindGroup = device.createBindGroup({ label: 'K6 bg', layout, entries: [
      { binding: 0, resource: { buffer: k4Handle.contactsBuf } },
      { binding: 1, resource: { buffer: k2Handle.outPositionsBuf } },
      { binding: 2, resource: { buffer: gpuPVScratchBuf } },
      { binding: 3, resource: { buffer: pvDeltaBuf } },
      { binding: 4, resource: { buffer: gravityHandle.metasBuf } },
      { binding: 5, resource: { buffer: paramsBuf } },
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

  function uploadParams(N, contactCount, dt) {
    view.setUint32 ( 0, N >>> 0, true);
    view.setUint32 ( 4, contactCount >>> 0, true);
    view.setFloat32( 8, dt, true);
    view.setUint32 (12, 0, true);
    device.queue.writeBuffer(paramsBuf, 0, scratch, 0, K6_PARAMS_SIZE);
  }

  function recordDispatch(encoder, N, contactCount) {
    if (N === 0 || contactCount === 0) return;
    const cWg = Math.ceil(contactCount / WORKGROUP);
    const eWg = Math.ceil(N / WORKGROUP);
    encoder.copyBufferToBuffer(k2Handle.pseudoVelsBuf, 0, gpuPVScratchBuf, 0, N * 8);
    encoder.clearBuffer(pvDeltaBuf, 0, capacity * 2 * 4);
    for (let iter = 0; iter < POS_ITERATIONS; iter++) {
      { const p = encoder.beginComputePass({ label: 'K6 accum' });
        p.setPipeline(pipeAccum); p.setBindGroup(0, bindGroup);
        p.dispatchWorkgroups(cWg); p.end(); }
      { const p = encoder.beginComputePass({ label: 'K6 apply' });
        p.setPipeline(pipeApply); p.setBindGroup(0, bindGroup);
        p.dispatchWorkgroups(eWg); p.end(); }
    }
  }

  function destroy() {
    destroyOwn();
    try { paramsBuf.destroy(); } catch {}
  }

  return {
    growIfNeeded, uploadParams, recordDispatch,
    onGravityRealloc() { if (capacity > 0) rebuildBindGroup(); },
    onK2Realloc()      { if (capacity > 0) rebuildBindGroup(); },
    onK4Realloc()      { if (capacity > 0) rebuildBindGroup(); },
    destroy,
    get capacity() { return capacity; },
    get gpuPVScratchBuf() { return gpuPVScratchBuf; },
    get pvDeltaBuf()      { return pvDeltaBuf; },
  };
}

function nextPow2(n) {
  if (n <= 1) return 1;
  let v = n - 1;
  v |= v >>> 1; v |= v >>> 2; v |= v >>> 4; v |= v >>> 8; v |= v >>> 16;
  return (v + 1) >>> 0;
}
