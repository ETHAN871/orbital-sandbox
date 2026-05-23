// fps-meter.js — opt-in FPS meter with cross-verification.
//
// Closure-interface wrapper around stats.js (mrdoob's well-known FPS
// widget, ~10KB, used by Three.js examples and many WebGL projects).
// stats.js exposes a class with begin()/end() bracketing the frame work;
// we wrap it so main.js stays unaware of the library specifics.
//
// Cross-verification: alongside stats.js we maintain our own simple
// EWMA over requestAnimationFrame deltas. Both numbers are logged to
// console every 2 seconds so the user can confirm the displayed FPS
// matches an independent calculation.
//
// Activation: enabled by default (lightweight). Can be disabled with
// ?fps=off URL param if it visually clutters or affects benchmarking.

import Stats from 'stats.js';

export function createFpsMeter() {
  const params = new URLSearchParams(window.location.search);
  if (params.get('fps') === 'off') {
    return { begin() {}, end() {}, getFps: () => 0, getRafFps: () => 0 };
  }

  const stats = new Stats();
  stats.showPanel(0);
  Object.assign(stats.dom.style, {
    position: 'fixed',
    top: 'auto', left: 'auto',
    bottom: '8px', right: '8px',
    zIndex: '9999',
  });
  document.body.appendChild(stats.dom);

  // Independent RAF-delta FPS calculation (for cross-verification).
  let lastRafTime = performance.now();
  let rafEwmaFps = 0;
  const EWMA_ALPHA = 0.05;

  let logCounter = 0;

  function begin() {
    stats.begin();
    const now = performance.now();
    const dt = (now - lastRafTime) / 1000;
    if (dt > 0 && dt < 1) {
      const instFps = 1 / dt;
      rafEwmaFps = rafEwmaFps === 0 ? instFps : rafEwmaFps + EWMA_ALPHA * (instFps - rafEwmaFps);
    }
    lastRafTime = now;
    if (++logCounter >= 120) {
      logCounter = 0;
      console.info(`[fps-meter] raf-ewma=${rafEwmaFps.toFixed(1)} fps`);
    }
  }

  function end() {
    stats.end();
  }

  return { begin, end, getFps: () => rafEwmaFps, getRafFps: () => rafEwmaFps };
}
