// gpu-init.js — WebGPU adapter + device detection with graceful fallback.
//
// Phase 1 of the WebGPU acceleration project. See docs/webgpu-blueprint.md §6.
//
// The exported `detectBackend()` is the only async entry point. It returns
// one of:
//   { backend: 'cpu' }                                            // no WebGPU
//   { backend: 'cpu', reason: '...' }                             // forced or failed
//   { backend: 'webgpu', adapter, device, lostPromise }           // success
//
// `lostPromise` resolves when the device is lost (driver crash, user-
// triggered, tab backgrounded too long). The caller wires this to a
// fallback handler that swaps `currentBackend` to cpu and re-syncs CPU
// entities from the most recent shadow buffer — see blueprint §6.3.
//
// URL parameters honored:
//   ?backend=force-cpu   → skip detection, return { backend: 'cpu' }
//                          (F1 acceptance: bit-identical to main branch)
//   ?backend=verbose     → console.info each stage of detection

// Only the retained K1 gravity kernel: 256 threads, 6 KiB shared memory,
// three storage buffers. Retired contact kernels required much higher limits
// and unnecessarily disabled acceleration on otherwise capable adapters.
const REQUIRED_LIMITS = {
  maxStorageBufferBindingSize: 64 * 1024 * 1024,
  maxComputeWorkgroupStorageSize: 16 * 1024,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxStorageBuffersPerShaderStage: 3,
};

function queryParam(name) {
  // globalThis.location works in both window (main thread) and worker
  // contexts. Workers expose self.location with the worker script URL's
  // search params; main has window.location. Using globalThis lets this
  // file be imported into physics-rapier-worker.js without a ReferenceError.
  const search = (typeof globalThis !== 'undefined' && globalThis.location)
    ? globalThis.location.search : '';
  const params = new URLSearchParams(search);
  return params.get(name);
}

export function isVerbose() {
  return queryParam('backend') === 'verbose';
}

export async function detectBackend() {
  const verbose = isVerbose();
  const forced = queryParam('backend');

  if (forced === 'force-cpu') {
    if (verbose) console.info('[gpu-init] backend forced to CPU via ?backend=force-cpu');
    return { backend: 'cpu', reason: 'forced via URL param' };
  }

  if (!navigator.gpu) {
    if (verbose) console.info('[gpu-init] navigator.gpu unavailable → CPU');
    return { backend: 'cpu', reason: 'navigator.gpu unavailable' };
  }

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    if (verbose) console.warn('[gpu-init] requestAdapter threw:', e);
    return { backend: 'cpu', reason: 'requestAdapter threw: ' + e.message };
  }
  if (!adapter) {
    if (verbose) console.info('[gpu-init] requestAdapter returned null → CPU');
    return { backend: 'cpu', reason: 'no adapter' };
  }

  // Validate adapter limits BEFORE requesting device — requestDevice with
  // requiredLimits exceeding adapter.limits would throw synchronously.
  if (adapter.limits.maxStorageBufferBindingSize < REQUIRED_LIMITS.maxStorageBufferBindingSize) {
    return {
      backend: 'cpu',
      reason: `maxStorageBufferBindingSize ${adapter.limits.maxStorageBufferBindingSize} < ${REQUIRED_LIMITS.maxStorageBufferBindingSize}`,
    };
  }
  if (adapter.limits.maxComputeWorkgroupStorageSize < REQUIRED_LIMITS.maxComputeWorkgroupStorageSize) {
    return {
      backend: 'cpu',
      reason: `maxComputeWorkgroupStorageSize ${adapter.limits.maxComputeWorkgroupStorageSize} < ${REQUIRED_LIMITS.maxComputeWorkgroupStorageSize}`,
    };
  }
  if (adapter.limits.maxComputeInvocationsPerWorkgroup < REQUIRED_LIMITS.maxComputeInvocationsPerWorkgroup) {
    return {
      backend: 'cpu',
      reason: `maxComputeInvocationsPerWorkgroup ${adapter.limits.maxComputeInvocationsPerWorkgroup} < ${REQUIRED_LIMITS.maxComputeInvocationsPerWorkgroup}`,
    };
  }
  if (adapter.limits.maxComputeWorkgroupSizeX < REQUIRED_LIMITS.maxComputeWorkgroupSizeX) {
    return {
      backend: 'cpu',
      reason: `maxComputeWorkgroupSizeX ${adapter.limits.maxComputeWorkgroupSizeX} < ${REQUIRED_LIMITS.maxComputeWorkgroupSizeX}`,
    };
  }
  if (adapter.limits.maxStorageBuffersPerShaderStage < REQUIRED_LIMITS.maxStorageBuffersPerShaderStage) {
    return {
      backend: 'cpu',
      reason: `maxStorageBuffersPerShaderStage ${adapter.limits.maxStorageBuffersPerShaderStage} < ${REQUIRED_LIMITS.maxStorageBuffersPerShaderStage}`,
    };
  }

  let device;
  try {
    device = await adapter.requestDevice({
      requiredLimits: REQUIRED_LIMITS,
      label: 'orbital-sandbox-physics',
    });
  } catch (e) {
    if (verbose) console.warn('[gpu-init] requestDevice threw:', e);
    return { backend: 'cpu', reason: 'requestDevice threw: ' + e.message };
  }

  // Surface uncaptured validation errors to console — silent in the spec
  // by default, which makes shader bugs invisible.
  device.addEventListener('uncapturederror', ev => {
    console.error('[gpu-init] uncapturederror:', ev.error.message);
  });

  if (verbose) {
    console.info('[gpu-init] WebGPU device acquired:', {
      vendor: adapter.info?.vendor,
      architecture: adapter.info?.architecture,
      device: adapter.info?.device,
      description: adapter.info?.description,
    });
  }

  return {
    backend: 'webgpu',
    adapter,
    device,
    lostPromise: device.lost,   // Promise<GPUDeviceLostInfo> — resolves on loss
  };
}

// Fetch + return a WGSL kernel source string. Phase 1 only needs
// gravity_accel.wgsl. Done in JS so we can keep the WGSL file as the
// single source of truth (no inline copy in physics-gpu-gravity.js).
//
// Cached so subsequent requestor calls don't re-fetch.
const _wgslCache = new Map();

export async function loadKernel(relativePath) {
  if (_wgslCache.has(relativePath)) return _wgslCache.get(relativePath);
  const url = new URL(relativePath, import.meta.url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`loadKernel(${relativePath}): HTTP ${res.status}`);
  const src = await res.text();
  _wgslCache.set(relativePath, src);
  return src;
}
