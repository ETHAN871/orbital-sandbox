// physics-gpu-k4.js — K4 (contact_detect) GPU pipeline wrapper.
// Phase 2c dark-launch: emits AbsorptionEvents + Contact records, writes
// entityMaxImpulse. CPU drains absEvents to call beginAbsorption().

const WORKGROUP_SIZE   = 256;
const ABS_CAPACITY     = 128;
const ABS_EVENT_STRIDE = 16;
const CONTACT_STRIDE   = 48;
const K4_PARAMS_SIZE   = 32;

const STATUS_ABS_OVERFLOW     = 0x2;
const STATUS_CONTACT_OVERFLOW = 0x8;

export async function createK4GPU(device, wgslSource, gravityHandle, broadphaseHandle) {
  const module = device.createShaderModule({ label: 'K4 contact_detect', code: wgslSource });

  const bgl = device.createBindGroupLayout({ label: 'K4 bgl', entries: [
    { binding:  0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding:  1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding:  2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding:  3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding:  4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding:  5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding:  6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding:  7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding:  8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding:  9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ]});

  const pipeline = await device.createComputePipelineAsync({
    label: 'K4', layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: 'contact_detect' },
  });

  const absEventBuf   = device.createBuffer({ label: 'absEvents',   size: ABS_CAPACITY * ABS_EVENT_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const absHeadBuf    = device.createBuffer({ label: 'absHead',     size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const statusFlagBuf = device.createBuffer({ label: 'statusFlags', size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const k4ParamsBuf   = device.createBuffer({ label: 'K4Params',    size: K4_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const absEventsStaging  = device.createBuffer({ label: 'absEvents staging', size: ABS_CAPACITY * ABS_EVENT_STRIDE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const absHeadStaging    = device.createBuffer({ label: 'absHead staging',   size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const contactCntStaging = device.createBuffer({ label: 'contactCnt staging',size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const statusStaging     = device.createBuffer({ label: 'status staging',    size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  let capacity = 0;
  let contactsBuf, contactCountBuf, maxImpulseBuf, bindGroup;
  const paramsScratch = new ArrayBuffer(K4_PARAMS_SIZE);
  const paramsView    = new DataView(paramsScratch);

  function destroyOwn() {
    if (contactsBuf)     { contactsBuf.destroy();     contactsBuf = null; }
    if (contactCountBuf) { contactCountBuf.destroy(); contactCountBuf = null; }
    if (maxImpulseBuf)   { maxImpulseBuf.destroy();   maxImpulseBuf = null; }
    bindGroup = null;
  }

  function allocateBuffers(cap) {
    destroyOwn();
    capacity = cap;
    contactsBuf     = device.createBuffer({ label: 'contacts',     size: cap * 3 * CONTACT_STRIDE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    contactCountBuf = device.createBuffer({ label: 'contactCount', size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    maxImpulseBuf   = device.createBuffer({ label: 'maxImpulse',   size: cap * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    rebuildBindGroup();
  }

  function rebuildBindGroup() {
    bindGroup = device.createBindGroup({ label: 'K4 bg', layout: bgl, entries: [
      { binding:  0, resource: { buffer: gravityHandle.positionsBuf  } },
      { binding:  1, resource: { buffer: gravityHandle.velocitiesBuf } },
      { binding:  2, resource: { buffer: gravityHandle.accelsBuf     } },
      { binding:  3, resource: { buffer: gravityHandle.metasBuf      } },
      { binding:  4, resource: { buffer: broadphaseHandle.cellCountsBuf   } },
      { binding:  5, resource: { buffer: broadphaseHandle.cellOffsetsBuf  } },
      { binding:  6, resource: { buffer: broadphaseHandle.cellContentsBuf } },
      { binding:  7, resource: { buffer: contactsBuf }     },
      { binding:  8, resource: { buffer: contactCountBuf } },
      { binding:  9, resource: { buffer: absEventBuf }     },
      { binding: 10, resource: { buffer: absHeadBuf }      },
      { binding: 11, resource: { buffer: maxImpulseBuf }   },
      { binding: 12, resource: { buffer: statusFlagBuf }   },
      { binding: 13, resource: { buffer: k4ParamsBuf }     },
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

  function uploadParams(N, gridMeta, dt, wrap) {
    paramsView.setUint32 ( 0, N >>> 0, true);
    paramsView.setUint32 ( 4, gridMeta.ncx >>> 0, true);
    paramsView.setUint32 ( 8, gridMeta.ncy >>> 0, true);
    paramsView.setUint32 (12, gridMeta.numCells >>> 0, true);
    paramsView.setFloat32(16, gridMeta.cellSizeX, true);
    paramsView.setFloat32(20, gridMeta.cellSizeY, true);
    paramsView.setFloat32(24, dt, true);
    paramsView.setUint32 (28, wrap ? 1 : 0, true);
    device.queue.writeBuffer(k4ParamsBuf, 0, paramsScratch, 0, K4_PARAMS_SIZE);
  }

  function recordDispatch(encoder, N) {
    if (N === 0) return;
    encoder.clearBuffer(contactCountBuf, 0, 4);
    encoder.clearBuffer(absHeadBuf,      0, 4);
    encoder.clearBuffer(maxImpulseBuf,   0, capacity * 4);
    // bug-fix-2026-05-23: also clear the contacts buffer itself. K4
    // atomicAdd-writes contacts[0..currentCount); slots beyond that hold
    // STALE data from previous substeps. K5/K6/K8 over-dispatch to a
    // safe upper bound (live count read from contactCountBuf in WGSL),
    // and any worker indexing a stale slot would feed garbage into the
    // solver. Clearing zero-inits everything: dist=0 → solvers'
    // dist²<1e-12 guard early-returns. Cost: capacity * 144 bytes per
    // substep — negligible (e.g. capacity=1024 → 144KB clear).
    encoder.clearBuffer(contactsBuf, 0, capacity * 3 * CONTACT_STRIDE);
    const pass = encoder.beginComputePass({ label: 'K4 contact_detect' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
    pass.end();
    encoder.copyBufferToBuffer(absEventBuf,     0, absEventsStaging,  0, ABS_CAPACITY * ABS_EVENT_STRIDE);
    encoder.copyBufferToBuffer(absHeadBuf,      0, absHeadStaging,    0, 4);
    encoder.copyBufferToBuffer(contactCountBuf, 0, contactCntStaging, 0, 4);
    encoder.copyBufferToBuffer(statusFlagBuf,   0, statusStaging,     0, 4);
  }

  async function readback() {
    const [headRaw, eventsRaw, cntRaw, statusRaw] = await Promise.all([
      (async () => { await absHeadStaging.mapAsync(GPUMapMode.READ, 0, 4); const v = new Uint32Array(absHeadStaging.getMappedRange(0, 4).slice(0))[0]; absHeadStaging.unmap(); return v; })(),
      (async () => { await absEventsStaging.mapAsync(GPUMapMode.READ, 0, ABS_CAPACITY * ABS_EVENT_STRIDE); const r = new Uint32Array(absEventsStaging.getMappedRange(0, ABS_CAPACITY * ABS_EVENT_STRIDE).slice(0)); absEventsStaging.unmap(); return r; })(),
      (async () => { await contactCntStaging.mapAsync(GPUMapMode.READ, 0, 4); const v = new Uint32Array(contactCntStaging.getMappedRange(0, 4).slice(0))[0]; contactCntStaging.unmap(); return v; })(),
      (async () => { await statusStaging.mapAsync(GPUMapMode.READ, 0, 4); const v = new Uint32Array(statusStaging.getMappedRange(0, 4).slice(0))[0]; statusStaging.unmap(); return v; })(),
    ]);
    const head = Math.min(headRaw, ABS_CAPACITY);
    const events = [];
    for (let s = 0; s < head; s++) {
      events.push({ preyIdx: eventsRaw[s * 4], predatorIdx: eventsRaw[s * 4 + 1] });
    }
    return { events, head, contactCount: cntRaw, status: statusRaw };
  }

  function resetStatusFlags() {
    const zero = new Uint32Array([0]);
    device.queue.writeBuffer(statusFlagBuf, 0, zero.buffer, 0, 4);
  }

  function destroy() {
    destroyOwn();
    for (const b of [absEventBuf, absHeadBuf, statusFlagBuf, k4ParamsBuf,
                     absEventsStaging, absHeadStaging, contactCntStaging, statusStaging]) {
      try { b.destroy(); } catch {}
    }
  }

  return {
    growIfNeeded, uploadParams, recordDispatch, readback, resetStatusFlags,
    onGravityRealloc()    { if (capacity > 0) rebuildBindGroup(); },
    onBroadphaseRealloc() { if (capacity > 0) rebuildBindGroup(); },
    destroy,
    get capacity() { return capacity; },
    get contactsBuf()     { return contactsBuf; },
    get contactCountBuf() { return contactCountBuf; },
    get maxImpulseBuf()   { return maxImpulseBuf; },
    STATUS_ABS_OVERFLOW, STATUS_CONTACT_OVERFLOW,
  };
}

function nextPow2(n) {
  if (n <= 1) return 1;
  let v = n - 1;
  v |= v >>> 1; v |= v >>> 2; v |= v >>> 4; v |= v >>> 8; v |= v >>> 16;
  return (v + 1) >>> 0;
}
