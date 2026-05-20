// potential.js — Pure JS reference for the gravitational potential field.
//
// Mirrors the GLSL `EQUIPOTENTIAL.FS` formula (see shaders.js) exactly:
//
//   φ(x, y) = Σ_i  -G · q_i · m_i / max(r_i, ε)
//
// where r_i = sqrt((x - e_i.x)² + (y - e_i.y)²), G is the gravitational
// constant from state, ε is the softening floor (state.epsilon).
//
// Sign convention (matches physics.js):
//   q = +1 on entity → attractive for a probe (φ < 0 near it; "well")
//   q = -1 on entity → repulsive for a probe (φ > 0 near it; "peak")
//   q =  0 on entity → contributes nothing to φ
//   Absorbing entities are skipped (mid-absorption, transient state).
//
// This module is NOT on the production render path — that goes through
// the GLSL fragment shader for per-pixel parallelism. It is used by:
//   1. tests/potential.test.html — hand-computed-fixture cross-checks to
//      verify the formula's behaviour and the GPU shader's agreement.
//   2. renderer-webgl.js — seeds streamline directions on CPU each frame
//      via computeForceDirAt() so the streamline shader doesn't need to
//      reduce-loop over entities per-pixel.
//
// Keep the formula bit-for-bit identical between this file and the FS
// or the test fixtures lose their value as a GPU-vs-CPU oracle.

export function computePotentialAt(x, y, entities, G, epsilon) {
  let phi = 0;
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const dx = x - e.x;
    const dy = y - e.y;
    const r = Math.sqrt(dx * dx + dy * dy);
    const safeR = Math.max(r, epsilon);
    phi += -G * e.charge * e.mass / safeR;
  }
  return phi;
}

// computeForceDirAt — analytic gradient of -φ, used to seed streamline
// directions on CPU each frame so the streamline shader doesn't need to
// reduce-loop per-pixel. Returns the unit vector along the local force
// plus the magnitude (for thresholding low-field-zones).
//
//   F(x, y) = -∇φ = Σ_i  G · q_i · m_i · (e_i - p) / max(r, ε)³
//
// The (e_i - p) outward + the positive G·q·m means a +charge entity pulls
// a +probe toward itself (force points from probe to entity). For
// near-zero-magnitude fields (saddle / symmetry points), returns a stable
// fallback direction (1, 0) so the streamline still has a defined tangent.
export function computeForceDirAt(x, y, entities, G, epsilon) {
  let fx = 0;
  let fy = 0;
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const dx = e.x - x;
    const dy = e.y - y;
    const r2 = dx * dx + dy * dy;
    const r = Math.sqrt(r2);
    const safeR = Math.max(r, epsilon);
    const inv3 = 1 / (safeR * safeR * safeR);
    const k = G * e.charge * e.mass * inv3;
    fx += k * dx;
    fy += k * dy;
  }
  const mag = Math.sqrt(fx * fx + fy * fy);
  if (mag < 1e-6) return { x: 1, y: 0, mag: 0 };
  return { x: fx / mag, y: fy / mag, mag };
}
