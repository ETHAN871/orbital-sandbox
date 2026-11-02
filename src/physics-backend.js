// Rapier is the supported engine; the local JS solver is the offline fallback.
import { state } from './state.js';
import { prepareFrame, stepPBD, updateAbsorptions, applyBoundary, clearContactState } from './physics.js';

function makeCpuBackend() {
  return {
    name: 'cpu',
    prepareFrame,
    async step(entities, dt, viewport, boundaryMode) {
      stepPBD(entities, dt);
      updateAbsorptions(entities, dt);
      applyBoundary(entities, viewport, boundaryMode);
    },
    onEntityMetaMaybeChanged() {},
    destroy() { clearContactState(); },
  };
}

export async function createBackend() {
  let active = makeCpuBackend();
  let initialized = false;
  let destroyed = false;
  function ensureAlive() {
    if (destroyed) throw new Error('Physics backend has been destroyed');
  }
  return {
    get name() { return active.name; },
    async init(entities) {
      ensureAlive();
      if (initialized) return;
      const params = new URLSearchParams(globalThis.location?.search || '');
      const engine = params.get('engine');
      const forceCpu = engine === 'cpu' || (engine === null && params.get('backend') === 'force-cpu');
      if (!forceCpu) {
        let candidate;
        try {
          const RAPIER = (await import('@dimforge/rapier2d-compat')).default;
          const { makeRapierBackend, setRapier } = await import('./physics-rapier.js');
          setRapier(RAPIER);
          candidate = makeRapierBackend();
          await candidate.init(entities);
          active = candidate;
        } catch (error) {
          try { candidate?.destroy(); } catch {}
          active = makeCpuBackend();
          console.warn('[physics-backend] Rapier unavailable; using local CPU solver:', error);
        }
      }
      initialized = true;
      state.backendName = active.name;
    },
    prepareFrame(entities) { ensureAlive(); active.prepareFrame(entities); },
    async step(...args) { ensureAlive(); await active.step(...args); },
    onEntityMetaMaybeChanged() { ensureAlive(); active.onEntityMetaMaybeChanged(); },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      active.destroy();
    },
  };
}
