// Reference potential for the legacy field illustration, not the pair dynamics.
// phi = sum(-G*q*m / sqrt(r^2 + epsilon^2)). Dynamics instead soften with
// max(radiusA + radiusB, epsilon) and use a single minimum image in wrap mode.
// This illustration sums nine images, a finite approximation (not an exact
// periodic Green function). GPU float32 and JS float64 are not bit-identical.
// computeForceDirAt is the analytic negative gradient of this reference.
export function computePotentialAt(x, y, entities, G, epsilon, boundary = null) {
  let phi = 0;
  const W = boundary ? boundary.width : 0;
  const H = boundary ? boundary.height : 0;
  const N = boundary ? 1 : 0;
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const baseK = -G * e.charge * e.mass;
    if (baseK === 0) continue;
    for (let oy = -N; oy <= N; oy++) {
      for (let ox = -N; ox <= N; ox++) {
        const dx = x - (e.x + ox * W);
        const dy = y - (e.y + oy * H);
        // Plummer softening: combined into a single sqrt with the
        // ε² term added inside. Saves one Math.max vs the old
        // max(sqrt(r²), ε) form and is mathematically smoother.
        const safeR = Math.sqrt(dx * dx + dy * dy + epsilon * epsilon);
        phi += baseK / safeR;
      }
    }
  }
  return phi;
}

// computeForceDirAt — analytic gradient of -φ, used to seed streamline
// directions on CPU each frame so the streamline shader doesn't need to
// reduce-loop per-pixel. Returns the unit vector along the local force
// plus the magnitude (for thresholding low-field-zones).
//
//   F(x, y) = -∇φ = Σ_i  G · q_i · m_i · (e_i - p) / (r² + ε²)^(3/2)
//
// The (e_i - p) outward + the positive G·q·m means a +charge entity pulls
// a +probe toward itself (force points from probe to entity). For
// near-zero-magnitude fields (saddle / symmetry points), returns a stable
// fallback direction (1, 0) so the streamline still has a defined tangent.
export function computeForceDirAt(x, y, entities, G, epsilon, boundary = null) {
  let fx = 0;
  let fy = 0;
  const W = boundary ? boundary.width : 0;
  const H = boundary ? boundary.height : 0;
  const N = boundary ? 1 : 0;
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const baseK = G * e.charge * e.mass;
    if (baseK === 0) continue;
    for (let oy = -N; oy <= N; oy++) {
      for (let ox = -N; ox <= N; ox++) {
        const dx = (e.x + ox * W) - x;
        const dy = (e.y + oy * H) - y;
        // Plummer softening: F_x = G·q·m·dx/(r²+ε²)^(3/2). Combining
        // the r²+ε² into one expression skips the max() branch.
        const r2Soft = dx * dx + dy * dy + epsilon * epsilon;
        const rSoft = Math.sqrt(r2Soft);
        const inv3 = 1 / (r2Soft * rSoft);
        const k = baseK * inv3;
        fx += k * dx;
        fy += k * dy;
      }
    }
  }
  const mag = Math.sqrt(fx * fx + fy * fy);
  if (mag < 1e-6) return { x: 1, y: 0, mag: 0 };
  return { x: fx / mag, y: fy / mag, mag };
}
