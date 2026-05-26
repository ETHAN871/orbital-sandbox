// field-lines.js — Option 4: numerical stream-function (ψ) contours
// for the curvilinear field-viz mode.
//
// HISTORY:
//  - v1: radial trajectories from each body (rejected — too spoky).
//  - v2: Jobard-Lefer evenly-spaced streamlines (rejected — still
//        bunched into bodies).
//  - v3 (this): orthogonal companion to equipotential via numerical
//        ψ. The level sets of φ and ψ together form a true
//        curvilinear grid that follows space curvature.
//
// MATH:
//   Our gravitational field F = -∇φ is curl-free (∇×F = 0 since F is
//   the gradient of a scalar). For any curl-free 2D vector field there
//   exists a stream function ψ such that:
//        ∂ψ/∂x =  F_y       ∂ψ/∂y = -F_x
//   Equivalently, ψ is obtained by path integration:
//        ψ(P) = ∫_O^P (F_y dx - F_x dy)
//   Path-independent because the integrand is exact. Level sets of ψ
//   are everywhere orthogonal to level sets of φ (gradient theorem).
//
//   We sample ψ on a coarse grid via a row-walk: first the top row
//   (walk right), then each column from the row above (walk down).
//   Marching squares extracts iso-contours. No atan2 → no branch cuts.
//   Multi-source case "just works" by superposition of the integrand.
//
//   Singularities at body interiors (φ → ∞ → ψ undefined) are masked
//   because (a) the Plummer-softened force is bounded near r=0, and
//   (b) renderer skips contour rendering inside body radii.
//
// Output format unchanged: array of Float32Array polylines
// [x0,y0,x1,y1,...] in CSS-px coords. Renderer consumes them via
// the existing LINE_SEG instanced pipeline.

const GRID_SPACING_PX = 10;   // ψ-sampling cell size; lower = more detail + cost
const PSI_CONTOUR_SLIDER_DIVISOR = 240;  // contour count = round(divisor / userSpacing)
const PSI_CONTOUR_MIN = 6;
const PSI_CONTOUR_MAX = 60;

/**
 * Compute ψ-contour polylines (Option 4 — orthogonal companion to φ).
 *
 * @param {Array} entities - state.entities
 * @param {{width:number,height:number}} viewport
 * @param {number} userSpacing - "场线间距" slider (CSS-px); maps to contour count
 * @param {number} G - gravitational constant
 * @param {number} epsilon - Plummer softening (px)
 * @param {boolean} wrap - true if state.boundaryMode === 'wrap'
 * @returns {Array<Float32Array>} polylines, each [x0,y0,x1,y1,...]
 */
export function computeFieldLines(entities, viewport, userSpacing, G, epsilon, wrap) {
  const polylines = [];
  if (entities.length === 0) return polylines;
  const W = viewport.width;
  const H = viewport.height;
  const eps2 = epsilon * epsilon;

  // Active bodies + their geometry for in-body masking later.
  const bodyData = [];
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    if (G * e.charge * e.mass === 0) continue;
    bodyData.push({ x: e.x, y: e.y, r: e.radius, Gqm: G * e.charge * e.mass });
  }
  if (bodyData.length === 0) return polylines;

  // ── Sample ψ on a coarse grid via path integral ────────────────────
  const GS = GRID_SPACING_PX;
  const cols = Math.ceil(W / GS) + 1;
  const rows = Math.ceil(H / GS) + 1;
  // Flat array for cache friendliness.
  const psi = new Float32Array(cols * rows);

  // Force at grid point (i, j) — pre-cached so we don't recompute.
  // Each entry is the 2D vector F = -∇φ in CSS-px / sim² units.
  const fx = new Float32Array(cols * rows);
  const fy = new Float32Array(cols * rows);

  for (let j = 0; j < rows; j++) {
    const y = j * GS;
    for (let i = 0; i < cols; i++) {
      const x = i * GS;
      let ax = 0, ay = 0;
      for (let b = 0; b < bodyData.length; b++) {
        const bd = bodyData[b];
        let dx = bd.x - x;
        let dy = bd.y - y;
        if (wrap) {
          if (dx >  W * 0.5) dx -= W;
          else if (dx < -W * 0.5) dx += W;
          if (dy >  H * 0.5) dy -= H;
          else if (dy < -H * 0.5) dy += H;
        }
        const r2 = dx * dx + dy * dy + eps2;
        const r = Math.sqrt(r2);
        const k = bd.Gqm / (r2 * r);
        ax += k * dx;
        ay += k * dy;
      }
      const idx = j * cols + i;
      fx[idx] = ax;
      fy[idx] = ay;
    }
  }

  // Integrate ψ. ψ[0][0] = 0; walk top row right, then each row down.
  psi[0] = 0;
  // Top row: ψ[0][i] = ψ[0][i-1] + avg(F_y over the horizontal edge) * GS
  for (let i = 1; i < cols; i++) {
    const fyAvg = 0.5 * (fy[i - 1] + fy[i]);
    psi[i] = psi[i - 1] + fyAvg * GS;
  }
  // For each subsequent row, integrate DOWN from the row above per column:
  // ψ[j][i] = ψ[j-1][i] - avg(F_x over the vertical edge) * GS
  for (let j = 1; j < rows; j++) {
    const rowBase = j * cols;
    const prevBase = (j - 1) * cols;
    for (let i = 0; i < cols; i++) {
      const fxAvg = 0.5 * (fx[prevBase + i] + fx[rowBase + i]);
      psi[rowBase + i] = psi[prevBase + i] - fxAvg * GS;
    }
  }

  // Determine ψ range from samples NOT inside any body (the singular
  // interior values would otherwise blow up the range).
  let psiMin = Infinity;
  let psiMax = -Infinity;
  for (let j = 0; j < rows; j++) {
    const y = j * GS;
    for (let i = 0; i < cols; i++) {
      const x = i * GS;
      if (insideAnyBody(x, y, bodyData)) continue;
      const v = psi[j * cols + i];
      if (v < psiMin) psiMin = v;
      if (v > psiMax) psiMax = v;
    }
  }
  if (!isFinite(psiMin) || !isFinite(psiMax) || psiMax - psiMin < 1e-6) {
    return polylines;
  }

  // Number of contours from user spacing.
  const N = Math.max(
    PSI_CONTOUR_MIN,
    Math.min(PSI_CONTOUR_MAX, Math.round(PSI_CONTOUR_SLIDER_DIVISOR / Math.max(8, userSpacing))),
  );
  // Place thresholds linearly across the ψ range. Linear spacing gives
  // ~constant visual density at typical (single-body) far field.
  const thresholds = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    thresholds[n] = psiMin + (psiMax - psiMin) * (n + 0.5) / N;
  }

  // Marching squares — output as line segments. Cell-level segments are
  // each pushed as a tiny 2-vertex polyline; the renderer's LINE_SEG
  // batch handles arbitrary counts efficiently.
  for (let n = 0; n < N; n++) {
    const thr = thresholds[n];
    for (let j = 0; j < rows - 1; j++) {
      for (let i = 0; i < cols - 1; i++) {
        const v00 = psi[j * cols + i];
        const v10 = psi[j * cols + i + 1];
        const v01 = psi[(j + 1) * cols + i];
        const v11 = psi[(j + 1) * cols + i + 1];
        let cs = 0;
        if (v00 > thr) cs |= 1;
        if (v10 > thr) cs |= 2;
        if (v11 > thr) cs |= 4;
        if (v01 > thr) cs |= 8;
        if (cs === 0 || cs === 15) continue;
        const x0 = i * GS;
        const y0 = j * GS;
        const x1 = x0 + GS;
        const y1 = y0 + GS;
        // Skip if the entire cell is inside a body — those values are
        // singular and the body sprite covers them anyway.
        if (insideAnyBody(x0 + GS * 0.5, y0 + GS * 0.5, bodyData)) continue;
        // Edge intersection points via linear interp.
        let pTopX = 0, pTopY = 0, pRightX = 0, pRightY = 0;
        let pBotX = 0, pBotY = 0, pLeftX = 0, pLeftY = 0;
        if (cs & 1) {
          // Top edge interpolation needed if v00 differs from v10 in sign vs threshold
          if ((cs ^ 0b0011) & 0b0011 || (cs ^ 0b0001) === 0) {
            // crossings handled below; populate only if used
          }
        }
        // Compute all four edge crossings unconditionally (cheaper than
        // conditional dispatch and only a few flops).
        pTopX = x0 + GS * lerpFrac(v00, v10, thr);
        pTopY = y0;
        pRightX = x1;
        pRightY = y0 + GS * lerpFrac(v10, v11, thr);
        pBotX = x0 + GS * lerpFrac(v01, v11, thr);
        pBotY = y1;
        pLeftX = x0;
        pLeftY = y0 + GS * lerpFrac(v00, v01, thr);
        // Segment lookup table by marching-squares case.
        switch (cs) {
          case 1: case 14: pushSeg(polylines, pTopX, pTopY, pLeftX, pLeftY); break;
          case 2: case 13: pushSeg(polylines, pTopX, pTopY, pRightX, pRightY); break;
          case 4: case 11: pushSeg(polylines, pRightX, pRightY, pBotX, pBotY); break;
          case 8: case 7:  pushSeg(polylines, pLeftX, pLeftY, pBotX, pBotY); break;
          case 3: case 12: pushSeg(polylines, pLeftX, pLeftY, pRightX, pRightY); break;
          case 6: case 9:  pushSeg(polylines, pTopX, pTopY, pBotX, pBotY); break;
          case 5: pushSeg(polylines, pTopX, pTopY, pRightX, pRightY);
                  pushSeg(polylines, pLeftX, pLeftY, pBotX, pBotY); break;
          case 10: pushSeg(polylines, pTopX, pTopY, pLeftX, pLeftY);
                   pushSeg(polylines, pRightX, pRightY, pBotX, pBotY); break;
        }
      }
    }
  }

  return polylines;
}

function insideAnyBody(x, y, bodyData) {
  for (let i = 0; i < bodyData.length; i++) {
    const b = bodyData[i];
    const dx = x - b.x;
    const dy = y - b.y;
    if (dx * dx + dy * dy < b.r * b.r) return true;
  }
  return false;
}

function lerpFrac(va, vb, thr) {
  if (vb === va) return 0.5;
  let t = (thr - va) / (vb - va);
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  return t;
}

function pushSeg(polylines, x0, y0, x1, y1) {
  const buf = new Float32Array(4);
  buf[0] = x0; buf[1] = y0; buf[2] = x1; buf[3] = y1;
  polylines.push(buf);
}
