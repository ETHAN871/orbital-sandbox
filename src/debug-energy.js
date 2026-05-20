// debug-energy.js — Temporary diagnostic for the orbital-acceleration bug.
//
// When `state.__debugEnergy` is true, every ENERGY_LOG_EVERY frames this
// dumps the system's kinetic + potential energy and individual body
// speeds to console. Used to discriminate between:
//
//   - Real multi-body energy redistribution (TOTAL E stays bounded, but
//     individual KE_i flows between bodies)
//   - Verlet numerical drift (TOTAL E grows monotonically)
//   - Collision bookkeeping bug (TOTAL E jumps at discrete moments)
//   - Elliptical orbit phase illusion (KE oscillates periodically with
//     no secular trend)
//
// Pair potential is derived under the assumption that all charges are
// same-sign (q_A=q_B). Asymmetric mixed-charge pairs do NOT have a
// scalar potential (Newton's 3rd is intentionally violated for those)
// so their contribution to "E" would be ill-defined; we skip them in
// the energy sum with a warning. The user's reported scene is all + so
// the sum is valid.
//
// Remove this file once the diagnosis is complete.

import { state } from './state.js';

const ENERGY_LOG_EVERY = 60;  // ~1 log/sec at 60Hz

let _frameCounter = 0;
let _initialE = null;
let _initialKE = null;
let _skippedMixedPairsWarned = false;

export function maybeLogEnergy() {
  if (!state.__debugEnergy) return;
  _frameCounter++;
  if (_frameCounter % ENERGY_LOG_EVERY !== 0) return;

  const ents = state.entities.filter(e => !e.absorbing);
  const n = ents.length;
  if (n === 0) return;

  // ─── Kinetic energy + per-body speeds ─────────────────────────
  let KE = 0;
  const speeds = [];
  for (const e of ents) {
    const v2 = e.vx * e.vx + e.vy * e.vy;
    const speed = Math.sqrt(v2);
    if (!e.pinned) KE += 0.5 * e.mass * v2;   // pinned body kinetic energy is 0 by spec
    speeds.push({ id: e.id, mass: e.mass, speed, pinned: !!e.pinned });
  }

  // ─── Potential energy (Plummer-softened pair sum) ─────────────
  // Force law: a_A_from_B = q_B · G · m_B / r² · n̂(B→A).
  // For same-sign pair (q_A·q_B > 0) the symmetric pair potential is:
  //   U_AB = -q_B · G·m_A·m_B / sqrt(r² + minR²)
  //   (For ++ → well, for -- → barrier; same magnitude opposite sign.)
  // Mixed +- has no scalar potential — Newton's 3rd violated.
  const G = state.G;
  const EPSILON = state.epsilon;
  const wrap = state.boundaryMode === 'wrap';
  const W = state.viewport.width;
  const H = state.viewport.height;
  const halfW = W * 0.5;
  const halfH = H * 0.5;

  let PE = 0;
  let mixedPairCount = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = ents[i];
      const b = ents[j];
      const qProduct = a.charge * b.charge;
      if (qProduct === 0) continue;
      if (qProduct < 0) { mixedPairCount++; continue; }
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
      const r2Raw = dx * dx + dy * dy;
      const minR = Math.max(a.radius + b.radius, EPSILON);
      const r = Math.sqrt(r2Raw + minR * minR);
      PE += -b.charge * G * a.mass * b.mass / r;
    }
  }
  if (mixedPairCount > 0 && !_skippedMixedPairsWarned) {
    console.warn(`[energy] ${mixedPairCount} mixed-charge pair(s) skipped from PE sum (no scalar potential exists for asymmetric force).`);
    _skippedMixedPairsWarned = true;
  }

  const E = KE + PE;
  if (_initialE === null) { _initialE = E; _initialKE = KE; }
  const driftE = E - _initialE;
  const driftEAbs = Math.abs(_initialE) > 1e-6
    ? (driftE / Math.abs(_initialE) * 100).toFixed(2) + '%'
    : 'N/A';
  const driftKEPct = Math.abs(_initialKE) > 1e-6
    ? ((KE - _initialKE) / _initialKE * 100).toFixed(1) + '%'
    : 'N/A';

  const t = (_frameCounter / 60).toFixed(1).padStart(5);
  const speedStr = speeds
    .filter(s => !s.pinned)
    .map(s => s.speed.toFixed(1))
    .join(', ');

  console.log(
    `[E@${t}s] KE=${KE.toFixed(0).padStart(7)} ` +
    `PE=${PE.toFixed(0).padStart(8)} ` +
    `E=${E.toFixed(0).padStart(8)} ` +
    `ΔE=${(driftE >= 0 ? '+' : '') + driftE.toFixed(0)} (${driftEAbs}) ` +
    `ΔKE=${driftKEPct} | speeds: [${speedStr}]`
  );
}

export function resetEnergyTrace() {
  _frameCounter = 0;
  _initialE = null;
  _initialKE = null;
  _skippedMixedPairsWarned = false;
}
