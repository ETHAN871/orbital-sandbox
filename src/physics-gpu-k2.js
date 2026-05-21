// physics-gpu-k2.js — K2 (kick_predict) GPU pipeline wrapper.
//
// Owns the K2-specific WebGPU resources: bind-group layout, compute
// pipeline, `outPositionsBuf` (K2's predicted positions output),
// `pseudoVelsBuf` (K2 zeros + K6 will eventually write), `nanCheckBuf`
// (atomic u32 × 1 shared with K7), and a K2 Params uniform.
//
// Shares with the gravity (K1) handle: `positionsBuf`, `velocitiesBuf`,
// `metasBuf`, `accelsBuf` (K1's outputBuf), and `device`. The factory
// takes `gravityHandle` and binds those buffers directly — no duplicate
// allocation. Architect M1 resolution.
//
// Phase 2a dispatch model: K1's encoder records K2's compute pass right
// after K1's, in the same submit. WebGPU's compute-pass barrier ensures
// K1's writes to accels[] are visible to K2's reads. After K2 the
// encoder also `copyBufferToBuffer`s outPositionsBuf and velocitiesBuf
// to the K2 staging slots so CPU can read back the post-kick state.

const WORKGROUP_SIZE   = 256;
const POS_STRIDE       = 8;  // vec2f
const PARAMS_BYTE_SIZE = 16; // {N: u32, dt: f32, _p0: u32, _p1: u32}

export async function createK2GPU(device, wgslSource, gravityHandle) {
  if (!gravityHandle) throw new Error('createK2GPU requires a gravityHandle');

  const shaderModule = device.createShaderModule({
    label: 'K2 kick_predict module',
    code: wgslSource,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'K2 bind group layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // positions
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // velocities (kick writes in place)
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // accels (K1 out)
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // metas
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // outPositions
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // pseudoVels
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // nanCheckBuf
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
    ],
  });

  // Phase 2 trap-list: `createComputePipelineAsync` to keep first-frame
  // latency bounded (sync would stall 50-200 ms on shader compile).
  const pipeline = await device.createComputePipelineAsync({
    label: 'K2 kick_predict pipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: 'kick_predict' },
  });

  const paramsBuf = device.createBuffer({
    label: 'K2 params uniform',
    size: PARAMS_BYTE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // nanCheckBuf is shared by K2 + K7 (K7 binds it via gravityHandle in 2e).
  // Single u32 atomic counter; CPU reads it once per frame.
  const nanCheckBuf = device.createBuffer({
    label: 'nanCheckBuf (shared K2/K7)',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const nanStagingBuf = device.createBuffer({
    label: 'nanCheckBuf staging',
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  let capacity = 0;
  let outPositionsBuf;
  let pseudoVelsBuf;
  let posStagingBufs;   // length-2 ping-pong
  let velStagingBufs;   // length-2 ping-pong
  let bindGroup;

  const paramsScratch = new ArrayBuffer(PARAMS_BYTE_SIZE);
  const paramsView    = new DataView(paramsScratch);
  const nanZeroPattern = new Uint32Array([0]);

  function destroyOwnBuffers() {
    if (outPositionsBuf) { outPositionsBuf.destroy(); outPositionsBuf = null; }
    if (pseudoVelsBuf)   { pseudoVelsBuf.destroy();   pseudoVelsBuf   = null; }
    if (posStagingBufs) {
      for (const sb of posStagingBufs) { try { sb.destroy(); } catch {} }
      posStagingBufs = null;
    }
    if (velStagingBufs) {
      for (const sb of velStagingBufs) { try { sb.destroy(); } catch {} }
      velStagingBufs = null;
    }
    bindGroup = null;
  }

  function allocateBuffers(newCapacity) {
    destroyOwnBuffers();
    capacity = newCapacity;
    outPositionsBuf = device.createBuffer({
      label: 'K2 outPositions',
      size: capacity * POS_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    pseudoVelsBuf = device.createBuffer({
      label: 'pseudoVels',
      size: capacity * POS_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const stagingUsage = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;
    posStagingBufs = [
      device.createBuffer({ label: 'K2 pos staging A', size: capacity * POS_STRIDE, usage: stagingUsage }),
      device.createBuffer({ label: 'K2 pos staging B', size: capacity * POS_STRIDE, usage: stagingUsage }),
    ];
    velStagingBufs = [
      device.createBuffer({ label: 'K2 vel staging A', size: capacity * POS_STRIDE, usage: stagingUsage }),
      device.createBuffer({ label: 'K2 vel staging B', size: capacity * POS_STRIDE, usage: stagingUsage }),
    ];
    rebuildBindGroup();
  }

  // Bind group references gravityHandle's positionsBuf/velocitiesBuf/metasBuf
  // by reference. When gravityHandle reallocates, the buffer objects are
  // destroyed; caller must invoke onGravityRealloc() to rebuild.
  function rebuildBindGroup() {
    bindGroup = device.createBindGroup({
      label: 'K2 bind group',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: gravityHandle.positionsBuf  } },
        { binding: 1, resource: { buffer: gravityHandle.velocitiesBuf } },
        { binding: 2, resource: { buffer: gravityHandle.accelsBuf     } },
        { binding: 3, resource: { buffer: gravityHandle.metasBuf      } },
        { binding: 4, resource: { buffer: outPositionsBuf } },
        { binding: 5, resource: { buffer: pseudoVelsBuf } },
        { binding: 6, resource: { buffer: nanCheckBuf } },
        { binding: 7, resource: { buffer: paramsBuf } },
      ],
    });
  }

  function onGravityRealloc() {
    if (capacity > 0) rebuildBindGroup();
  }

  function growIfNeeded(N) {
    if (N > capacity) {
      const newCap = Math.max(256, nextPow2(N));
      allocateBuffers(newCap);
      return true;
    }
    if (capacity > 256 && N < capacity / 4) {
      const newCap = Math.max(256, nextPow2(N));
      if (newCap < capacity) { allocateBuffers(newCap); return true; }
    }
    return false;
  }

  function uploadParams(N, dt) {
    paramsView.setUint32 (0,  N >>> 0, true);
    paramsView.setFloat32(4,  dt,      true);
    paramsView.setUint32 (8,  0, true);
    paramsView.setUint32 (12, 0, true);
    device.queue.writeBuffer(paramsBuf, 0, paramsScratch, 0, PARAMS_BYTE_SIZE);
  }

  // Reset nanCheckBuf to 0. Caller invokes once per frame (NOT per substep)
  // so a NaN trip in substep S is still visible after substeps S+1..N for
  // the frame's CPU drain — see blueprint §3 NaN guards paragraph.
  function resetNanCounter() {
    device.queue.writeBuffer(nanCheckBuf, 0, nanZeroPattern.buffer, 0, 4);
  }

  function recordDispatch(encoder, N, stagingIdx) {
    const pass = encoder.beginComputePass({ label: 'K2 pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
    pass.end();
    encoder.copyBufferToBuffer(outPositionsBuf,           0, posStagingBufs[stagingIdx], 0, N * POS_STRIDE);
    encoder.copyBufferToBuffer(gravityHandle.velocitiesBuf, 0, velStagingBufs[stagingIdx], 0, N * POS_STRIDE);
    encoder.copyBufferToBuffer(nanCheckBuf, 0, nanStagingBuf, 0, 4);
  }

  async function readbackPositions(stagingIdx, N) {
    const sb = posStagingBufs[stagingIdx];
    await sb.mapAsync(GPUMapMode.READ, 0, N * POS_STRIDE);
    const mapped = sb.getMappedRange(0, N * POS_STRIDE);
    const out = new Float32Array(mapped.byteLength / 4);
    out.set(new Float32Array(mapped));
    sb.unmap();
    return out;
  }

  async function readbackVelocities(stagingIdx, N) {
    const sb = velStagingBufs[stagingIdx];
    await sb.mapAsync(GPUMapMode.READ, 0, N * POS_STRIDE);
    const mapped = sb.getMappedRange(0, N * POS_STRIDE);
    const out = new Float32Array(mapped.byteLength / 4);
    out.set(new Float32Array(mapped));
    sb.unmap();
    return out;
  }

  async function readbackNanCounter() {
    await nanStagingBuf.mapAsync(GPUMapMode.READ, 0, 4);
    const mapped = nanStagingBuf.getMappedRange(0, 4);
    const v = new Uint32Array(mapped.slice(0))[0];
    nanStagingBuf.unmap();
    return v;
  }

  function destroy() {
    destroyOwnBuffers();
    try { paramsBuf.destroy(); } catch {}
    try { nanCheckBuf.destroy(); } catch {}
    try { nanStagingBuf.destroy(); } catch {}
  }

  return {
    growIfNeeded,
    onGravityRealloc,
    uploadParams,
    resetNanCounter,
    recordDispatch,
    readbackPositions,
    readbackVelocities,
    readbackNanCounter,
    destroy,
    get capacity() { return capacity; },
    get pseudoVelsBuf()   { return pseudoVelsBuf; },
    get outPositionsBuf() { return outPositionsBuf; },
    get nanCheckBuf()     { return nanCheckBuf; },
  };
}

function nextPow2(n) {
  if (n <= 1) return 1;
  let v = n - 1;
  v |= v >>> 1; v |= v >>> 2; v |= v >>> 4; v |= v >>> 8; v |= v >>> 16;
  return (v + 1) >>> 0;
}
