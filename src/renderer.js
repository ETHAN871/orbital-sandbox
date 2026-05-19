// renderer.js — All canvas drawing. Pure: reads state, writes pixels.
//
// Layer order (back to front):
//   1. Background
//   2. Trail canvas (phosphor-decay, persistent off-screen)
//   3. Slingshot line + dashed prediction (only during drag)
//   4. Entities (planets + black holes)
//   5. Selected highlight ring

import { state } from './state.js';
import { resolveDisplayColor } from './entities.js';
import { ensureEntitySprite } from './sprite-cache.js';

// V8.1c: ABSORPTION_DURATION moved to state.absorptionDuration (UI-tunable).
// BG_COLOR replaced by state.bgColor (light/dark toggle button).
const SELECT_RING_COLOR = '#6b8cff';
const PREDICTION_DASH = [6, 6];
const PREDICTION_BATCHES = 8;   // sub-segments for fade-along-path effect
const RUBBER_BAND_DASH = [5, 4];
const HANDLE_LINE_WIDTH = 2;
const GHOST_FILL_ALPHA = 0.18;  // faint preview at placement point

// ─── V8.1 phosphor-decay trail canvas ─────────────────────────────
// Persistent off-screen canvas: each frame we (1) overlay a translucent
// black fill to fade existing pixels and (2) plot one small dot per
// active entity at its current position. Composite onto the main canvas
// as a single drawImage. Cost: O(N) per frame instead of O(N × history).

// V8.1c trail dot: solid interior + 1 px AA edge.
// Radius R = state.trailWidth / 2 (slider 1-6 px → R 0.5-3.0).
// For a pixel at distance d from the entity center:
//   d ≤ R - 0.5         → α = 255 (fully solid; thickness doesn't vary
//                          with sub-pixel position, fixing V8.1's "线宽
//                          不一致" perception bug)
//   R - 0.5 < d ≤ R + 0.5 → α linearly ramps from 255 down to 0
//   d > R + 0.5         → skip
// Writing uses max(new, existing) so the AA edge of a fresh dot never
// dims older bright trail pixels at the same pixel.

const _colorRgbCache = new Map();

let _trailCanvas = null;
let _trailCtx = null;
let _trailData = null;          // Canvas's ImageData — manipulated directly each frame.

function ensureTrailCanvas() {
  const w = state.viewport.width;
  const h = state.viewport.height;
  if (w <= 0 || h <= 0) return null;
  if (!_trailCanvas || _trailCanvas.width !== w || _trailCanvas.height !== h) {
    _trailCanvas = document.createElement('canvas');
    _trailCanvas.width = w;
    _trailCanvas.height = h;
    _trailCtx = _trailCanvas.getContext('2d');
    _trailData = _trailCtx.createImageData(w, h);   // all zero → fully transparent
  }
  return _trailCtx;
}

// Hard clear (e.g., on "clear sandbox"). Resets the entire ImageData buffer
// and pushes the cleared state to the canvas.
export function resetTrailCanvas() {
  if (!_trailData || !_trailCtx) return;
  _trailData.data.fill(0);
  _trailCtx.putImageData(_trailData, 0, 0);
}

// Apply LINEAR alpha decay + plot current-frame dots. Operates directly on
// the ImageData byte buffer because canvas composite ops (destination-out
// etc.) only support multiplicative/exponential decay, which leaves
// pixels at ~37% alpha after `lifetime` sec and reads as a persistent
// gray smear on the dark background.
//
// True linear: alpha decreases by a constant 255/lifetime per second of
// simulation time. After `lifetime` sim seconds the pixel hits 0 exactly
// and stays there. Called once per RAF from main.js with the sim-time
// delta (paused → freeze, fast-forward → fade faster).
//
// Cost: one Uint8ClampedArray sweep of the trail canvas (~2M ops at 1080p)
// + one putImageData. Hot path but unavoidable for strict linear decay.
export function updateTrailCanvas(simDeltaTime) {
  const tctx = ensureTrailCanvas();
  if (!tctx) return;
  const w = _trailCanvas.width;
  const h = _trailCanvas.height;
  const data = _trailData.data;             // Uint8ClampedArray, RGBA per pixel
  const len = data.length;

  // Slider 0-500 → lifetime in sim seconds = slider / 50.
  // slider 0 → instant wipe; slider 100 → 2 s; slider 500 → 10 s.
  const lifetime = state.trailLength / 50;

  if (lifetime <= 0) {
    // Instant wipe.
    data.fill(0);
    tctx.putImageData(_trailData, 0, 0);
    return;
  }

  if (simDeltaTime <= 0) {
    // Paused — leave trail buffer alone (don't fade, don't plot new dots).
    // The canvas already shows the last-pushed state.
    return;
  }

  // Linear alpha decrement per frame. dec = 255 × (Δt / lifetime).
  // For lifetime = 2 s and Δt ≈ 1/30 s (2 substeps at 60 Hz): dec ≈ 4.
  // After 60 frames (~ 2 s sim time) every pixel's alpha has dropped 255
  // and is fully transparent — exactly what "lifetime = 2 s" should mean.
  const decFloat = 255 * simDeltaTime / lifetime;
  if (decFloat >= 255) {
    data.fill(0);
  } else {
    // Math.ceil so we never undershoot — guarantees zero by lifetime.
    const dec = Math.max(1, Math.ceil(decFloat));
    for (let i = 3; i < len; i += 4) {
      const v = data[i] - dec;
      data[i] = v > 0 ? v : 0;
    }
  }

  // Plot solid-interior + 1 px AA edge dots at each non-absorbing entity.
  // Geometry parameters derived from the user's trailWidth slider (1-6 px).
  const R = state.trailWidth * 0.5;
  const R_INNER = R - 0.5;                    // strictly-solid radius
  const R_OUTER = R + 0.5;                    // outer AA edge (alpha → 0)
  const R_INNER2 = R_INNER > 0 ? R_INNER * R_INNER : 0;
  const R_OUTER2 = R_OUTER * R_OUTER;
  const BBOX = Math.ceil(R_OUTER);

  for (let k = 0; k < state.entities.length; k++) {
    const e = state.entities[k];
    if (e.absorbing) continue;
    const rgb = colorToRgb(e.color);
    const cr = rgb[0], cg = rgb[1], cb = rgb[2];
    const ex = e.x;
    const ey = e.y;
    const px = ex | 0;
    const py = ey | 0;
    // Single-pixel fast path: at width = 1 px (R = 0.5), the BBOX scan
    // produces a faint plus-shape (1 center pixel + 4 half-alpha cardinals).
    // Snap to one crisp center pixel instead.
    if (R <= 0.5) {
      if (px >= 0 && px < w && py >= 0 && py < h) {
        const idx = (py * w + px) * 4;
        if (255 > data[idx + 3]) {
          data[idx]     = cr;
          data[idx + 1] = cg;
          data[idx + 2] = cb;
          data[idx + 3] = 255;
        }
      }
      continue;
    }
    for (let dy = -BBOX; dy <= BBOX; dy++) {
      const yy = py + dy;
      if (yy < 0 || yy >= h) continue;
      const cy = (yy + 0.5) - ey;             // pixel-center → entity-center (Y)
      const cy2 = cy * cy;
      for (let dx = -BBOX; dx <= BBOX; dx++) {
        const xx = px + dx;
        if (xx < 0 || xx >= w) continue;
        const cx = (xx + 0.5) - ex;
        const d2 = cx * cx + cy2;
        if (d2 >= R_OUTER2) continue;         // outside dot — fast reject
        let a;
        if (d2 <= R_INNER2) {
          a = 255;                            // strictly solid interior
        } else {
          // Edge ramp: linear from 255 at R_INNER to 0 at R_OUTER (width 1 px).
          const d = Math.sqrt(d2);
          const ramp = (R_OUTER - d) * 255;
          a = ramp >= 255 ? 255 : (ramp | 0);
        }
        if (a === 0) continue;
        const idx = (yy * w + xx) * 4;
        if (a > data[idx + 3]) {
          data[idx]     = cr;
          data[idx + 1] = cg;
          data[idx + 2] = cb;
          data[idx + 3] = a;
        }
      }
    }
  }

  // Push manipulated buffer to the canvas in a single GPU transfer.
  tctx.putImageData(_trailData, 0, 0);
}

// Parse `e.color` ("hsl(...)" or "#rrggbb") to [r, g, b] 0-255. Cached so
// each unique color string is parsed only once across the session.
function colorToRgb(c) {
  const cached = _colorRgbCache.get(c);
  if (cached) return cached;
  let result;
  if (c.charCodeAt(0) === 0x23 /* '#' */) {
    const hex = c.slice(1);
    if (hex.length === 6) {
      result = [
        parseInt(hex.slice(0, 2), 16),
        parseInt(hex.slice(2, 4), 16),
        parseInt(hex.slice(4, 6), 16),
      ];
    } else if (hex.length === 3) {
      result = [
        parseInt(hex[0] + hex[0], 16),
        parseInt(hex[1] + hex[1], 16),
        parseInt(hex[2] + hex[2], 16),
      ];
    } else {
      result = [128, 128, 128];
    }
  } else {
    const m = c.match(/hsl\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)%\s*,\s*(\d+(?:\.\d+)?)%\s*\)/);
    result = m ? hslToRgb(+m[1], +m[2], +m[3]) : [128, 128, 128];
  }
  _colorRgbCache.set(c, result);
  return result;
}

function hslToRgb(h, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const v = l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return Math.round(255 * v);
  };
  return [f(0), f(8), f(4)];
}

export function drawScene(ctx) {
  const { width, height } = state.viewport;
  drawBackground(ctx, width, height);
  drawTrails(ctx);
  drawHoverGhost(ctx);
  drawDragPreview(ctx);
  drawEntities(ctx);
  drawSelectionRing(ctx);
}

// Show a translucent preview at the cursor when placement mode is active
// and the pointer is on the canvas — lets users gauge the radius before
// committing to a drag. Suppressed during an active drag (drag has its own
// ghost at the placement point) and in edit mode (clicking selects, not places).
function drawHoverGhost(ctx) {
  if (state.isEditMode) return;
  if (state.drag) return;
  if (!state.hoverPos) return;

  const color = resolveDisplayColor(
    state.pending.type, state.pending.charge,
    state.pending.type === 'planet' ? '#ffffff' : '#000000',
  );
  const r = state.pending.radius;
  const { x, y } = state.hoverPos;

  ctx.globalAlpha = 0.18;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawBackground(ctx, w, h) {
  ctx.fillStyle = state.bgColor;
  ctx.fillRect(0, 0, w, h);
}

// V8.1: composite the phosphor-decay trail canvas onto the main view.
// Fade + dot plotting happens in updateTrailCanvas (called per RAF from
// main.js); this just blits the result. drawImage handles alpha naturally.
function drawTrails(ctx) {
  if (!_trailCanvas) return;
  if (state.trailLength <= 0) return;
  ctx.drawImage(_trailCanvas, 0, 0);
}

function drawDragPreview(ctx) {
  const drag = state.drag;
  if (!drag) return;

  const previewColor = resolveDisplayColor(
    state.pending.type, state.pending.charge, drag.previewBaseColor,
  );
  const radius = state.pending.radius;

  // 1. Ghost circle at the *placement* point (where the body will spawn).
  //    Faint filled disk + thin outline so users can see "this is where it'll be".
  ctx.globalAlpha = GHOST_FILL_ALPHA;
  ctx.fillStyle = previewColor;
  ctx.beginPath();
  ctx.arc(drag.startX, drag.startY, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = previewColor;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(drag.startX, drag.startY, radius, 0, Math.PI * 2);
  ctx.stroke();

  // 2. Dashed "rubber band" from placement point to the cursor (handle).
  ctx.strokeStyle = previewColor;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(RUBBER_BAND_DASH);
  ctx.beginPath();
  ctx.moveTo(drag.startX, drag.startY);
  ctx.lineTo(drag.currentX, drag.currentY);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Hollow handle ring at the cursor (where the player is "pulling to").
  //    Size scales with drag distance up to a sensible cap so it's always visible.
  const dx = drag.currentX - drag.startX;
  const dy = drag.currentY - drag.startY;
  const dragDist = Math.hypot(dx, dy);
  const handleRadius = Math.max(6, Math.min(14, dragDist * 0.08 + 6));
  ctx.strokeStyle = previewColor;
  ctx.lineWidth = HANDLE_LINE_WIDTH;
  ctx.beginPath();
  ctx.arc(drag.currentX, drag.currentY, handleRadius, 0, Math.PI * 2);
  ctx.stroke();

  // 4. Prediction path — color = the entity's would-be display color.
  //    Goes opposite to the drag because launch velocity is negated.
  const path = drag.predictionPath;
  if (path && path.length >= 2) {
    drawDashedFadingPath(ctx, path, previewColor);
  }
}

// Draw a dashed polyline whose alpha decreases along the path (head full opacity,
// tail vanishing). We split the path into `PREDICTION_BATCHES` sub-strokes and
// apply globalAlpha per batch — canvas setLineDash can't fade individual dashes.
// In wrap mode we skip segments that jump across the boundary so the line
// doesn't draw a straight slash across the whole viewport.
// V7: `path` is now `{ data: Float32Array, length: number }` — interleaved
// (x, y) samples in a flat buffer instead of an array of objects.
function drawDashedFadingPath(ctx, path, color) {
  const n = path.length;
  const data = path.data;
  const batchSize = Math.max(2, Math.ceil(n / PREDICTION_BATCHES));
  const wrap = state.boundaryMode === 'wrap';
  const wrapX = wrap ? state.viewport.width * 0.5 : Infinity;
  const wrapY = wrap ? state.viewport.height * 0.5 : Infinity;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(PREDICTION_DASH);
  for (let b = 0; b < PREDICTION_BATCHES; b++) {
    const start = b * batchSize;
    if (start >= n - 1) break;
    const end = Math.min(n - 1, start + batchSize);
    const alpha = 1 - b / PREDICTION_BATCHES;     // 1.0 → ~0.125
    ctx.globalAlpha = alpha * 0.85;
    ctx.beginPath();
    let prevX = data[start * 2];
    let prevY = data[start * 2 + 1];
    ctx.moveTo(prevX, prevY);
    for (let i = start + 1; i <= end; i++) {
      const x = data[i * 2];
      const y = data[i * 2 + 1];
      if (Math.abs(x - prevX) > wrapX || Math.abs(y - prevY) > wrapY) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      prevX = x; prevY = y;
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
}

function drawEntities(ctx) {
  const wrap = state.boundaryMode === 'wrap';
  const W = state.viewport.width;
  const H = state.viewport.height;
  for (const e of state.entities) {
    drawOneEntity(ctx, e);
    if (!wrap || e.absorbing) continue;

    // Portal-style mirror: when the body straddles a wrap edge, draw a
    // ghost copy at the opposite edge so the user sees the body
    // continuously crossing instead of teleporting.
    const r = Math.max(0, e.radius);
    const nearLeft   = e.x < r;
    const nearRight  = e.x > W - r;
    const nearTop    = e.y < r;
    const nearBottom = e.y > H - r;
    if (nearLeft)   drawEntityAtMirror(ctx, e, W, 0);
    if (nearRight)  drawEntityAtMirror(ctx, e, -W, 0);
    if (nearTop)    drawEntityAtMirror(ctx, e, 0, H);
    if (nearBottom) drawEntityAtMirror(ctx, e, 0, -H);
    if (nearLeft && nearTop)     drawEntityAtMirror(ctx, e, W, H);
    if (nearLeft && nearBottom)  drawEntityAtMirror(ctx, e, W, -H);
    if (nearRight && nearTop)    drawEntityAtMirror(ctx, e, -W, H);
    if (nearRight && nearBottom) drawEntityAtMirror(ctx, e, -W, -H);
  }
}

// V7: drawOneEntity now takes explicit (drawX, drawY), so the mirror call
// no longer needs to mutate-and-restore the entity. Passes isMirror=true
// to suppress charge-glyph text render (visible only on the primary copy).
function drawEntityAtMirror(ctx, e, ox, oy) {
  drawOneEntity(ctx, e, e.x + ox, e.y + oy, true);
}

// V8.1: normal entities are rendered via the sprite cache (drawImage of a
// pre-baked off-screen canvas). This collapses ~5-12 canvas state writes
// per draw into 1 cheap blit and is especially valuable at boundary
// corners where each entity gets up to 9× mirror copies.
//
// Absorbing entities can't use the cache (their alpha changes per frame),
// so we keep the V7 arc-fallback path for them. Few entities are absorbing
// at any time, so the slow path's cost stays small.
function drawOneEntity(ctx, e, drawX, drawY /* isMirror unused with sprite cache */) {
  if (drawX === undefined) drawX = e.x;
  if (drawY === undefined) drawY = e.y;

  if (e.absorbing) {
    drawAbsorbingFallback(ctx, e, drawX, drawY);
    return;
  }

  const sprite = ensureEntitySprite(e);
  if (!sprite) return;                              // safety
  ctx.drawImage(sprite, drawX - sprite._ox, drawY - sprite._oy);
}

// Slow path for absorbing entities — same visuals as V7's arc-based code
// but parameterised on (drawX, drawY) so mirror copies render at offset.
function drawAbsorbingFallback(ctx, e, drawX, drawY) {
  const t = Math.min(1, e.absorbing.elapsedSim / state.absorptionDuration);
  ctx.globalAlpha = Math.max(0, 1 - t);
  const r = Math.max(0, e.radius);

  ctx.fillStyle = e.color;
  ctx.beginPath();
  ctx.arc(drawX, drawY, r, 0, Math.PI * 2);
  ctx.fill();

  if (e.type === 'black_hole') {
    ctx.strokeStyle = e.charge === -1
      ? 'rgba(0,0,0,0.55)'
      : 'rgba(120,180,255,0.75)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  ctx.globalAlpha = 1;
}

function drawSelectionRing(ctx) {
  if (state.selectedId === null) return;
  const sel = state.entities.find(e => e.id === state.selectedId);
  if (!sel) return;
  ctx.strokeStyle = SELECT_RING_COLOR;
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(sel.x, sel.y, sel.radius + 6, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
}
