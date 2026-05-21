// physics-gpu-broadphase.js — K3/K3b/K3c broadphase GPU wrapper.
// 2b dark-launch: dispatches each substep, output not consumed until K4 (2c).

const WORKGROUP_PER_ENTITY = 256;
const BLOCK_SIZE = 1024;
const MAX_CELLS = 4096;
const MAX_BLOCKS = MAX_CELLS / BLOCK_SIZE;
const BROADPHASE_PARAMS_SIZE = 32;
const SCAN_PARAMS_SIZE = 16;
const MIN_CELL_SIZE_FLOOR = 32;   // matches physics-spatial-hash.js:68

export async function createBroadphaseGPU(device, wgslSources, gravityHandle, k2Handle) {
  if (!gravityHandle || !k2Handle) throw new Error('createBroadphaseGPU requires gravityHandle + k2Handle');

  const k3Module  = device.createShaderModule({ label: 'K3 module',  code: wgslSources.k3 });
  const k3bModule = device.createShaderModule({ label: 'K3b module', code: wgslSources.k3b });
  const k3cModule = device.createShaderModule({ label: 'K3c module', code: wgslSources.k3c });

  const k3Layout = device.createBindGroupLayout({
    label: 'K3 bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const k3bLayout = device.createBindGroupLayout({
    label: 'K3b bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const k3cLayout = device.createBindGroupLayout({
    label: 'K3c bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const [k3Pipe, k3bBlockPipe, k3bSpinePipe, k3bApplyPipe, k3cPipe] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'K3', layout: device.createPipelineLayout({ bindGroupLayouts: [k3Layout] }),
      compute: { module: k3Module, entryPoint: 'broadphase_count' },
    }),
    device.createComputePipelineAsync({
      label: 'K3b block_scan', layout: device.createPipelineLayout({ bindGroupLayouts: [k3bLayout] }),
      compute: { module: k3bModule, entryPoint: 'block_scan' },
    }),
    device.createComputePipelineAsync({
      label: 'K3b spine_scan', layout: device.createPipelineLayout({ bindGroupLayouts: [k3bLayout] }),
      compute: { module: k3bModule, entryPoint: 'spine_scan' },
    }),
    device.createComputePipelineAsync({
      label: 'K3b apply_spine', layout: device.createPipelineLayout({ bindGroupLayouts: [k3bLayout] }),
      compute: { module: k3bModule, entryPoint: 'apply_spine' },
    }),
    device.createComputePipelineAsync({
      label: 'K3c', layout: device.createPipelineLayout({ bindGroupLayouts: [k3cLayout] }),
      compute: { module: k3cModule, entryPoint: 'broadphase_scatter' },
    }),
  ]);

  const cellCountsBuf       = device.createBuffer({ label: 'cellCounts',       size: MAX_CELLS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const cellOffsetsBuf      = device.createBuffer({ label: 'cellOffsets',      size: MAX_CELLS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const cellWriteCursorsBuf = device.createBuffer({ label: 'cellWriteCursors', size: MAX_CELLS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const blockTotalsBuf      = device.createBuffer({ label: 'blockTotals',      size: MAX_BLOCKS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bpParamsBuf         = device.createBuffer({ label: 'BroadphaseParams', size: BROADPHASE_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const scanParamsBuf       = device.createBuffer({ label: 'ScanParams',       size: SCAN_PARAMS_SIZE,       usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  let capacity = 0;
  let cellContentsBuf;
  let cellCountsStaging;
  let k3BindGroup, k3bBindGroup, k3cBindGroup;
  let lastNumCells = 0, lastNumBlocks = 0;

  const bpScratch = new ArrayBuffer(BROADPHASE_PARAMS_SIZE);
  const bpView    = new DataView(bpScratch);
  const scanScratch = new ArrayBuffer(SCAN_PARAMS_SIZE);
  const scanView    = new DataView(scanScratch);

  function destroyOwnBuffers() {
    if (cellContentsBuf)   { cellContentsBuf.destroy();   cellContentsBuf = null; }
    if (cellCountsStaging) { cellCountsStaging.destroy(); cellCountsStaging = null; }
    k3BindGroup = null; k3bBindGroup = null; k3cBindGroup = null;
  }

  function allocateBuffers(newCapacity) {
    destroyOwnBuffers();
    capacity = newCapacity;
    cellContentsBuf = device.createBuffer({
      label: 'cellContents', size: capacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    cellCountsStaging = device.createBuffer({
      label: 'cellCounts staging', size: MAX_CELLS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    rebuildBindGroups();
  }

  function rebuildBindGroups() {
    k3BindGroup = device.createBindGroup({
      label: 'K3 bg', layout: k3Layout,
      entries: [
        { binding: 0, resource: { buffer: k2Handle.outPositionsBuf } },
        { binding: 1, resource: { buffer: gravityHandle.metasBuf } },
        { binding: 2, resource: { buffer: cellCountsBuf } },
        { binding: 3, resource: { buffer: bpParamsBuf } },
      ],
    });
    k3bBindGroup = device.createBindGroup({
      label: 'K3b bg', layout: k3bLayout,
      entries: [
        { binding: 0, resource: { buffer: cellCountsBuf } },
        { binding: 1, resource: { buffer: cellOffsetsBuf } },
        { binding: 2, resource: { buffer: blockTotalsBuf } },
        { binding: 3, resource: { buffer: scanParamsBuf } },
      ],
    });
    k3cBindGroup = device.createBindGroup({
      label: 'K3c bg', layout: k3cLayout,
      entries: [
        { binding: 0, resource: { buffer: k2Handle.outPositionsBuf } },
        { binding: 1, resource: { buffer: gravityHandle.metasBuf } },
        { binding: 2, resource: { buffer: cellOffsetsBuf } },
        { binding: 3, resource: { buffer: cellWriteCursorsBuf } },
        { binding: 4, resource: { buffer: cellContentsBuf } },
        { binding: 5, resource: { buffer: bpParamsBuf } },
      ],
    });
  }

  function onGravityRealloc() { if (capacity > 0) rebuildBindGroups(); }
  function onK2Realloc()      { if (capacity > 0) rebuildBindGroups(); }

  function growIfNeeded(N) {
    if (N > capacity) {
      allocateBuffers(Math.max(256, nextPow2(N)));
      return true;
    }
    if (capacity > 256 && N < capacity / 4) {
      const newCap = Math.max(256, nextPow2(N));
      if (newCap < capacity) { allocateBuffers(newCap); return true; }
    }
    return false;
  }

  // Mirrors physics-spatial-hash.js:63-80 exactly. Returns grid metadata.
  function uploadParams(entities, N, viewportW, viewportH) {
    let maxR = 16;
    for (let i = 0; i < N; i++) {
      const r = entities[i].radius;
      if (r > maxR) maxR = r;
    }
    const minCellSize = Math.max(MIN_CELL_SIZE_FLOOR, Math.ceil(maxR * 2));
    const W = viewportW, H = viewportH;
    const ncx = Math.max(1, Math.floor(W / minCellSize));
    const ncy = Math.max(1, Math.floor(H / minCellSize));
    const cellSizeX = W > 0 ? W / ncx : minCellSize;
    const cellSizeY = H > 0 ? H / ncy : minCellSize;
    const numCells = ncx * ncy;
    const numBlocks = Math.ceil(numCells / BLOCK_SIZE);
    if (numCells > MAX_CELLS) {
      throw new Error(`broadphase: numCells ${numCells} > MAX_CELLS ${MAX_CELLS} (viewport ${W}×${H}, minCellSize ${minCellSize})`);
    }

    bpView.setUint32 (0,  N >>> 0,        true);
    bpView.setUint32 (4,  ncx >>> 0,      true);
    bpView.setUint32 (8,  ncy >>> 0,      true);
    bpView.setUint32 (12, numCells >>> 0, true);
    bpView.setFloat32(16, cellSizeX,      true);
    bpView.setFloat32(20, cellSizeY,      true);
    bpView.setUint32 (24, 0, true);
    bpView.setUint32 (28, 0, true);
    device.queue.writeBuffer(bpParamsBuf, 0, bpScratch, 0, BROADPHASE_PARAMS_SIZE);

    scanView.setUint32(0, numCells >>> 0,  true);
    scanView.setUint32(4, numBlocks >>> 0, true);
    scanView.setUint32(8, 0, true);
    scanView.setUint32(12, 0, true);
    device.queue.writeBuffer(scanParamsBuf, 0, scanScratch, 0, SCAN_PARAMS_SIZE);

    lastNumCells = numCells;
    lastNumBlocks = numBlocks;
    return { ncx, ncy, numCells, numBlocks, cellSizeX, cellSizeY, minCellSize };
  }

  function recordDispatch(encoder, N) {
    if (N === 0 || lastNumCells === 0) return;
    encoder.clearBuffer(cellCountsBuf,       0, MAX_CELLS * 4);
    encoder.clearBuffer(cellWriteCursorsBuf, 0, MAX_CELLS * 4);

    const groups = Math.ceil(N / WORKGROUP_PER_ENTITY);

    {
      const pass = encoder.beginComputePass({ label: 'K3 count' });
      pass.setPipeline(k3Pipe); pass.setBindGroup(0, k3BindGroup);
      pass.dispatchWorkgroups(groups); pass.end();
    }
    {
      const pass = encoder.beginComputePass({ label: 'K3b block_scan' });
      pass.setPipeline(k3bBlockPipe); pass.setBindGroup(0, k3bBindGroup);
      pass.dispatchWorkgroups(lastNumBlocks); pass.end();
    }
    {
      const pass = encoder.beginComputePass({ label: 'K3b spine_scan' });
      pass.setPipeline(k3bSpinePipe); pass.setBindGroup(0, k3bBindGroup);
      pass.dispatchWorkgroups(1); pass.end();
    }
    {
      const pass = encoder.beginComputePass({ label: 'K3b apply_spine' });
      pass.setPipeline(k3bApplyPipe); pass.setBindGroup(0, k3bBindGroup);
      pass.dispatchWorkgroups(lastNumBlocks); pass.end();
    }
    {
      const pass = encoder.beginComputePass({ label: 'K3c scatter' });
      pass.setPipeline(k3cPipe); pass.setBindGroup(0, k3cBindGroup);
      pass.dispatchWorkgroups(groups); pass.end();
    }
  }

  async function readbackCellCounts() {
    if (lastNumCells === 0) return new Uint32Array(0);
    const enc = device.createCommandEncoder({ label: 'cellCounts readback' });
    enc.copyBufferToBuffer(cellCountsBuf, 0, cellCountsStaging, 0, lastNumCells * 4);
    device.queue.submit([enc.finish()]);
    await cellCountsStaging.mapAsync(GPUMapMode.READ, 0, lastNumCells * 4);
    const mapped = cellCountsStaging.getMappedRange(0, lastNumCells * 4);
    const out = new Uint32Array(mapped.byteLength / 4);
    out.set(new Uint32Array(mapped));
    cellCountsStaging.unmap();
    return out;
  }

  function destroy() {
    destroyOwnBuffers();
    try { cellCountsBuf.destroy(); } catch {}
    try { cellOffsetsBuf.destroy(); } catch {}
    try { cellWriteCursorsBuf.destroy(); } catch {}
    try { blockTotalsBuf.destroy(); } catch {}
    try { bpParamsBuf.destroy(); } catch {}
    try { scanParamsBuf.destroy(); } catch {}
  }

  return {
    growIfNeeded,
    onGravityRealloc,
    onK2Realloc,
    uploadParams,
    recordDispatch,
    readbackCellCounts,
    destroy,
    get capacity() { return capacity; },
    get numCells() { return lastNumCells; },
    get cellCountsBuf()   { return cellCountsBuf; },
    get cellOffsetsBuf()  { return cellOffsetsBuf; },
    get cellContentsBuf() { return cellContentsBuf; },
  };
}

function nextPow2(n) {
  if (n <= 1) return 1;
  let v = n - 1;
  v |= v >>> 1; v |= v >>> 2; v |= v >>> 4; v |= v >>> 8; v |= v >>> 16;
  return (v + 1) >>> 0;
}
