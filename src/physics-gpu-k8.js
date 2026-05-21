// physics-gpu-k8.js — K8 rebuild_warm_start wrapper. Phase 2f dark-launch.
// Spec: docs/webgpu-blueprint.md §3 K8 + §12 G8 + §4 pairImpulseTable.

const WORKGROUP = 256;
const K8_PARAMS_SIZE  = 16;
const CELL_META_STRIDE = 20;   // keyA u32, keyB u32, j f32, nx f32, ny f32

// Table size = max(256, nextPow2(N * 4)) — 4× load factor per blueprint §4.
// Split layout (cellMeta + cellFlags) lets WGSL keep atomic flags without
// forcing per-field atomic on data reads.
function tableSizeFor(N) {
  if (N <= 64) return 256;
  let want = N * 4;
  let v = want - 1;
  v |= v >>> 1; v |= v >>> 2; v |= v >>> 4; v |= v >>> 8; v |= v >>> 16;
  return Math.max(256, (v + 1) >>> 0);
}

export async function createK8GPU(device, wgslSource, gravityHandle, k4Handle) {
  const module = device.createShaderModule({ label: 'K8 rebuild_warm_start', code: wgslSource });

  const layout = device.createBindGroupLayout({ label: 'K8 bgl', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // contacts
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // metas
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // cellMeta
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // cellFlags (atomic)
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // statusFlags (atomic)
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
  ]});

  const pipeline = await device.createComputePipelineAsync({
    label: 'K8', layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: 'rebuild_warm_start' },
  });

  const statusFlagBuf = device.createBuffer({ label: 'K8 statusFlags', size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const paramsBuf = device.createBuffer({ label: 'K8 params', size: K8_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const statusStaging = device.createBuffer({ label: 'K8 status staging', size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  let tableSize = 0;
  let cellMetaBuf, cellFlagsBuf, bindGroup;
  const scratch = new ArrayBuffer(K8_PARAMS_SIZE);
  const view = new DataView(scratch);

  function destroyOwn() {
    if (cellMetaBuf)  { cellMetaBuf.destroy();  cellMetaBuf = null; }
    if (cellFlagsBuf) { cellFlagsBuf.destroy(); cellFlagsBuf = null; }
    bindGroup = null;
  }

  function allocateTable(newSize) {
    destroyOwn();
    tableSize = newSize;
    cellMetaBuf = device.createBuffer({
      label: 'pairImpulseTable.meta',
      size: tableSize * CELL_META_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    cellFlagsBuf = device.createBuffer({
      label: 'pairImpulseTable.flags',
      size: tableSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    rebuildBindGroup();
  }

  function rebuildBindGroup() {
    bindGroup = device.createBindGroup({ label: 'K8 bg', layout, entries: [
      { binding: 0, resource: { buffer: k4Handle.contactsBuf } },
      { binding: 1, resource: { buffer: gravityHandle.metasBuf } },
      { binding: 2, resource: { buffer: cellMetaBuf } },
      { binding: 3, resource: { buffer: cellFlagsBuf } },
      { binding: 4, resource: { buffer: statusFlagBuf } },
      { binding: 5, resource: { buffer: paramsBuf } },
    ]});
  }

  function growIfNeeded(N) {
    const want = tableSizeFor(N);
    if (want > tableSize) { allocateTable(want); return true; }
    if (tableSize > 256 && want < tableSize / 4) {
      const nc = tableSizeFor(N);
      if (nc < tableSize) { allocateTable(nc); return true; }
    }
    return false;
  }

  function uploadParams(contactCount) {
    view.setUint32( 0, contactCount >>> 0, true);
    view.setUint32( 4, tableSize >>> 0, true);
    view.setUint32( 8, (tableSize - 1) >>> 0, true);
    view.setUint32(12, 0, true);
    device.queue.writeBuffer(paramsBuf, 0, scratch, 0, K8_PARAMS_SIZE);
  }

  function recordDispatch(encoder, contactCount) {
    if (contactCount === 0 || tableSize === 0) return;
    // §3 K8 ¶: zero all occupancy bits each substep (no tombstone).
    encoder.clearBuffer(cellFlagsBuf, 0, tableSize * 4);
    encoder.clearBuffer(statusFlagBuf, 0, 4);
    const pass = encoder.beginComputePass({ label: 'K8 rebuild_warm_start' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(contactCount / WORKGROUP));
    pass.end();
    encoder.copyBufferToBuffer(statusFlagBuf, 0, statusStaging, 0, 4);
  }

  async function readbackStatus() {
    await statusStaging.mapAsync(GPUMapMode.READ, 0, 4);
    const v = new Uint32Array(statusStaging.getMappedRange(0, 4).slice(0))[0];
    statusStaging.unmap();
    return v;
  }

  function destroy() {
    destroyOwn();
    try { statusFlagBuf.destroy(); } catch {}
    try { paramsBuf.destroy(); } catch {}
    try { statusStaging.destroy(); } catch {}
  }

  return {
    growIfNeeded, uploadParams, recordDispatch, readbackStatus,
    onGravityRealloc() { if (tableSize > 0) rebuildBindGroup(); },
    onK4Realloc()      { if (tableSize > 0) rebuildBindGroup(); },
    destroy,
    get tableSize()    { return tableSize; },
    get cellMetaBuf()  { return cellMetaBuf; },
    get cellFlagsBuf() { return cellFlagsBuf; },
  };
}
