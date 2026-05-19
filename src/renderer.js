// renderer.js — All canvas drawing. Pure: reads state, writes pixels.
//
// Layer order (back to front):
//   1. Background
//   2. Trail canvas (phosphor-decay, persistent off-screen)
//   3. Slingshot line + dashed prediction (only during drag)
//   4. Entities (planets + black holes)
//   5. Selected highlight ring

import { state, ABSORPTION_DURATION } from './state.js';
import { resolveDisplayColor } from './entities.js';
import { ensureEntitySprite } from './sprite-cache.js';

const BG_COLOR = '#0a0a0f';
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

const TRAIL_DOT_RADIUS = 1.5;   // px; visually a 3 px dot

let _trailCanvas = null;
let _trailCtx = null;

function ensureTrailCanvas() {
  const w = state.viewport.width;
  const h = state.viewport.height;
  if (w <= 0 || h <= 0) return null;
  if (!_trailCanvas || _trailCanvas.width !== w || _trailCanvas.height !== h) {
    _trailCanvas = document.createElement('canvas');
    _trailCanvas.width = w;
    _trailCanvas.height = h;
    _trailCtx = _trailCanvas.getContext('2d');
  }
  return _trailCtx;
}

// Hard clear (e.g., on "clear sandbox"). Currently called from ui.js's
// clear-button handler when the trail buffer should be visually wiped.
export function resetTrailCanvas() {
  if (_trailCtx) {
    _trailCtx.clearRect(0, 0, _trailCanvas.width, _trailCanvas.height);
  }
}

// Apply fade + draw current-frame dots. Called once per RAF from main.js
// with the simulation-time delta (so trail decays with sim clock, not
// wall clock — pause freezes trails, fast-forward fades faster).
export function updateTrailCanvas(simDeltaTime) {
  const tctx = ensureTrailCanvas();
  if (!tctx) return;

  // Slider value 0-500 mapped to lifetime in seconds: lifetime = slider / 50.
  // slider=0   → lifetime=0   → fadeAlpha=1   → instant clear (no trail visible)
  // slider=100 → lifetime=2s  → ~per-frame alpha matches a 2-second decay
  // slider=500 → lifetime=10s → very long trail
  //
  // Fade uses `destination-out` composite so each frame REDUCES existing
  // pixels' alpha by `a`, eventually leaving fully-transparent pixels.
  // Previous V8.1 used `source-over` with `rgba(10,10,15,a)` which left
  // residual opaque-dark pixels visible against the dark background as a
  // ghostly gray smear. With destination-out, faded trails truly vanish.
  const lifetime = state.trailLength / 50;
  if (lifetime <= 0) {
    tctx.clearRect(0, 0, _trailCanvas.width, _trailCanvas.height);
  } else if (simDeltaTime > 0) {
    const a = Math.min(1, simDeltaTime / lifetime);
    tctx.globalCompositeOperation = 'destination-out';
    tctx.fillStyle = `rgba(0, 0, 0, ${a})`;       // color ignored in destination-out; only alpha matters
    tctx.fillRect(0, 0, _trailCanvas.width, _trailCanvas.height);
    tctx.globalCompositeOperation = 'source-over'; // restore default for dot plotting below
  }
  // If simDeltaTime == 0 (paused) we apply NO fade AND skip plotting new
  // dots — trail is truly frozen. Otherwise plotting opaque dots over
  // existing pixels would saturate the center pixel of each entity's
  // current location, preventing decay on subsequent resume.
  if (simDeltaTime <= 0) return;

  // Plot a dot at each non-absorbing entity's current position.
  for (let i = 0; i < state.entities.length; i++) {
    const e = state.entities[i];
    if (e.absorbing) continue;
    tctx.fillStyle = e.color;
    tctx.beginPath();
    tctx.arc(e.x, e.y, TRAIL_DOT_RADIUS, 0, Math.PI * 2);
    tctx.fill();
  }
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
  ctx.fillStyle = BG_COLOR;
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
  const t = Math.min(1, e.absorbing.elapsedSim / ABSORPTION_DURATION);
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
