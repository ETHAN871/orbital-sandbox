// perf-monitor.js — Process monitor for diagnosing main-thread stalls.
//
// Motivation: the existing fps-meter logs raf-EWMA, which masks tail
// latency. A 100 ms stall every 30 frames produces "60 fps avg" but a
// horrible user experience (cursor freezes, hover ghost stuck in place).
// This module measures the right things for our scenario:
//
//   1. PerformanceObserver `longtask` — browser-native: ANY main-thread
//      task ≥ 50 ms is reported, including time inside Rapier WASM that
//      our JS-side timers can't see between phase marks.
//
//   2. Rolling max per-frame dt over a 5-second window (NOT the EWMA).
//
//   3. Per-phase ms breakdown (sync, gravity, worldStep, post). Phase
//      marks accumulate across substeps within the same frame so the
//      HUD shows "this RAF spent X ms in world.step total".
//
//   4. Live context: bodies, awake count, contact count (last substep),
//      adaptive iter count (current). Shown alongside the timing so a
//      stall can be correlated with what the scene was doing.
//
// Gated on ?perf=1 URL param. When disabled, all hooks are NO-OPS at
// runtime (the markPhase() / endFrame() call sites stay in code but
// resolve to a function that returns immediately — negligible overhead).

// (No imports — perf-monitor is intentionally state-free, just tracks
// timing data via performance.now() and PerformanceObserver.)

const FRAME_WINDOW_MS = 5000;        // rolling window for max-dt / spike counts
const HUD_UPDATE_INTERVAL_MS = 250;   // throttle HUD repaint to 4 Hz
const STALL_THRESHOLD_MS = 50;        // user-visible threshold (1 missed vsync at 20 Hz)

let enabled = false;
let hudEl = null;
let hudLastUpdate = 0;

// Per-frame phase accumulators (reset each endFrame).
// interFrame: browser-side work BETWEEN endFrame() and the next frame's
// first markPhase() — captures GC pauses, layout/composite, RAF
// scheduling. This is the gap that explained the user's stall: a 79ms
// dt where only 30ms was in tracked phases means 50ms was here.
const phaseAccum = {
  prepareFrame: 0, worldStep: 0, post: 0, render: 0, interFrame: 0, other: 0,
};
let currentPhaseName = null;
let currentPhaseT0 = 0;

// Frame-time samples + longtask events, ring-buffered by wallclock time.
const frameSamples = []; // { t, dt, phases: {...snapshot} }
const longTasks = [];    // { t, duration, name }

let frameLastT = 0;

// Live context set by the substep loop / step backend.
const ctx = {
  N: 0,
  awake: 0,
  contacts: 0,
  iters: { vel: 0, pos: 0 },
  substepsThisFrame: 0,
  backend: null,
  gpuOn: false,
};

// ── Init ────────────────────────────────────────────────────────────

export function initPerfMonitor() {
  try {
    // globalThis.location is available in workers (self.location) and main
    // (window.location). The worker has no HUD anyway — initPerfMonitor's
    // hudEl creation needs `document`, which doesn't exist in workers, so
    // this still gracefully no-ops when called from a worker context.
    const search = (typeof globalThis !== 'undefined' && globalThis.location)
      ? globalThis.location.search : '';
    const sp = new URLSearchParams(search);
    enabled = sp.get('perf') === '1';
  } catch { enabled = false; }
  if (!enabled) return;
  if (typeof document === 'undefined') {
    // Worker context — no HUD DOM. Mark enabled so phase tracking runs,
    // but skip element creation.
    return;
  }

  hudEl = document.createElement('div');
  hudEl.id = 'perf-hud';
  hudEl.style.cssText = [
    'position: fixed', 'top: 8px', 'right: 8px',
    'background: rgba(0, 0, 0, 0.72)', 'color: #d8dde6',
    'font: 11px/1.35 ui-monospace, "JetBrains Mono", Consolas, monospace',
    'padding: 8px 10px', 'border-radius: 4px',
    'pointer-events: none', 'z-index: 9999',
    'min-width: 220px', 'white-space: pre',
    'border: 1px solid rgba(255,255,255,0.08)',
  ].join(';');
  hudEl.textContent = 'perf-mon initializing…';
  document.body.appendChild(hudEl);

  // PerformanceObserver longtask — catches anything ≥ 50 ms regardless
  // of which JS/WASM frame it ran in. Non-invasive (no per-task cost).
  if (typeof PerformanceObserver !== 'undefined') {
    try {
      const obs = new PerformanceObserver(list => {
        const now = performance.now();
        for (const entry of list.getEntries()) {
          longTasks.push({ t: now, duration: entry.duration, name: entry.name || 'longtask' });
        }
        pruneWindow();
      });
      obs.observe({ entryTypes: ['longtask'] });
    } catch (e) {
      console.warn('[perf-monitor] longtask observer failed:', e);
    }
  }

  frameLastT = performance.now();
  if (typeof console !== 'undefined') {
    console.info('[perf-monitor] enabled — gated on ?perf=1');
  }
}

// ── Phase tracking ──────────────────────────────────────────────────

export function markPhase(name) {
  if (!enabled) return;
  const now = performance.now();
  if (currentPhaseName) {
    phaseAccum[currentPhaseName] = (phaseAccum[currentPhaseName] || 0) + (now - currentPhaseT0);
  }
  currentPhaseName = name;
  currentPhaseT0 = now;
}

export function endPhase() {
  if (!enabled || !currentPhaseName) return;
  const now = performance.now();
  phaseAccum[currentPhaseName] = (phaseAccum[currentPhaseName] || 0) + (now - currentPhaseT0);
  currentPhaseName = null;
}

// ── Frame boundaries ────────────────────────────────────────────────

export function endFrame() {
  if (!enabled) return;
  endPhase();
  const now = performance.now();
  const dt = now - frameLastT;
  frameLastT = now;
  // Snapshot per-frame phase totals
  const snapshot = { ...phaseAccum };
  frameSamples.push({ t: now, dt, phases: snapshot });
  // Reset accumulators for next frame
  for (const k in phaseAccum) phaseAccum[k] = 0;
  // Open the interFrame phase: any time from now until the next
  // markPhase() (which fires at the top of runFrame) is browser work
  // — GC, RAF scheduling, layout/composite. It accumulates into the
  // NEXT frame's snapshot, which is the right semantic because that's
  // when the user perceives the dt spike.
  currentPhaseName = 'interFrame';
  currentPhaseT0 = now;
  pruneWindow();

  // Throttled HUD repaint
  if (now - hudLastUpdate >= HUD_UPDATE_INTERVAL_MS) {
    hudLastUpdate = now;
    repaintHud(now);
  }
}

function pruneWindow() {
  const cutoff = performance.now() - FRAME_WINDOW_MS;
  while (frameSamples.length && frameSamples[0].t < cutoff) frameSamples.shift();
  while (longTasks.length && longTasks[0].t < cutoff) longTasks.shift();
}

// ── Live context ────────────────────────────────────────────────────

export function isEnabled() {
  return enabled;
}

export function setContext(c) {
  if (!enabled) return;
  if (c.N !== undefined) ctx.N = c.N;
  if (c.awake !== undefined) ctx.awake = c.awake;
  if (c.contacts !== undefined) ctx.contacts = c.contacts;
  if (c.iters !== undefined) ctx.iters = c.iters;
  if (c.substepsThisFrame !== undefined) ctx.substepsThisFrame = c.substepsThisFrame;
  if (c.backend !== undefined) ctx.backend = c.backend;
  if (c.gpuOn !== undefined) ctx.gpuOn = c.gpuOn;
}

// ── HUD render ──────────────────────────────────────────────────────

function repaintHud(_now) {
  if (!hudEl || frameSamples.length === 0) return;
  const dts = frameSamples.map(s => s.dt).sort((a, b) => a - b);
  const median = dts[dts.length >> 1];
  const p95 = dts[Math.floor(dts.length * 0.95)];
  const max = dts[dts.length - 1];
  const fps = 1000 / median;

  const stalls = frameSamples.filter(s => s.dt >= STALL_THRESHOLD_MS).length;
  const longTaskCount = longTasks.length;
  const worstLongTask = longTasks.reduce((m, e) => Math.max(m, e.duration), 0);

  const phaseStats = {};
  for (const name of Object.keys(phaseAccum)) {
    let sum = 0, maxP = 0;
    for (const s of frameSamples) {
      const v = s.phases[name] || 0;
      sum += v;
      if (v > maxP) maxP = v;
    }
    phaseStats[name] = {
      avg: frameSamples.length ? sum / frameSamples.length : 0,
      max: maxP,
    };
  }

  const fmt = (n, w = 5) => n.toFixed(1).padStart(w);
  const stallStyle = stalls > 0 ? '⚠' : ' ';

  const lines = [];
  lines.push(`${stallStyle} FPS  ${fps.toFixed(0).padStart(3)} | dt med ${fmt(median)} p95 ${fmt(p95)} max ${fmt(max)}`);
  lines.push(`  Stalls(${FRAME_WINDOW_MS/1000}s): ${stalls} | LongTasks: ${longTaskCount} worst ${fmt(worstLongTask)}`);
  lines.push(`  Bodies ${ctx.N} (awake ${ctx.awake}) | contacts ${ctx.contacts}`);
  lines.push(`  Iters vel/pos ${ctx.iters.vel}/${ctx.iters.pos} | substeps ${ctx.substepsThisFrame} | ${ctx.backend || '?'}${ctx.gpuOn ? '+gpu' : ''}`);
  lines.push('  ── phases (ms, avg/max) ──');
  const phaseOrder = Object.entries(phaseStats)
    .sort((a, b) => b[1].avg - a[1].avg)
    .filter(([_, s]) => s.avg > 0.01 || s.max > 0.5);
  for (const [name, s] of phaseOrder) {
    lines.push(`    ${name.padEnd(10)}${fmt(s.avg)} / ${fmt(s.max)}`);
  }
  hudEl.textContent = lines.join('\n');

  // Visual stall flash: red border when there's an active stall in window
  if (stalls > 0 || worstLongTask >= STALL_THRESHOLD_MS) {
    hudEl.style.border = '1px solid #f55';
  } else {
    hudEl.style.border = '1px solid rgba(255,255,255,0.08)';
  }
}

// ── Test helper: expose for preview-MCP querying ────────────────────

export function getStats() {
  if (!enabled) return { enabled: false };
  const dts = frameSamples.map(s => s.dt);
  // Top-5 worst frames with their phase breakdown — helps attribute
  // which phase caused each spike (or shows when the spike isn't
  // attributable to any tracked phase = unmeasured cost like GC).
  const worstFrames = [...frameSamples].sort((a, b) => b.dt - a.dt).slice(0, 5).map(s => {
    const phaseSum = Object.values(s.phases).reduce((a, b) => a + b, 0);
    return {
      t: +s.t.toFixed(0),
      dt: +s.dt.toFixed(1),
      phaseTotal: +phaseSum.toFixed(1),
      unaccountedMs: +(s.dt - phaseSum).toFixed(1),
      phases: Object.fromEntries(
        Object.entries(s.phases)
          .filter(([_, v]) => v > 0.1)
          .map(([k, v]) => [k, +v.toFixed(2)]),
      ),
    };
  });
  return {
    enabled: true,
    windowMs: FRAME_WINDOW_MS,
    frames: frameSamples.length,
    dtMin: dts.length ? Math.min(...dts) : 0,
    dtMax: dts.length ? Math.max(...dts) : 0,
    dtMedian: dts.length ? [...dts].sort((a,b)=>a-b)[dts.length>>1] : 0,
    longTaskCount: longTasks.length,
    longTaskMax: longTasks.reduce((m, e) => Math.max(m, e.duration), 0),
    stallCount: frameSamples.filter(s => s.dt >= STALL_THRESHOLD_MS).length,
    phases: { ...phaseAccum },
    ctx: { ...ctx },
    longTasksRaw: longTasks.map(e => ({ t: +(e.t).toFixed(0), duration: +(e.duration).toFixed(1) })).slice(-20),
    worstFrames,
  };
}

// Reset the rolling windows. Useful for separating "spawn-time spike"
// from "steady-state cost" in autonomous benchmarks.
export function resetStats() {
  frameSamples.length = 0;
  longTasks.length = 0;
}
