// renderer.js — V9.0a overlay (Canvas2D) UI rendering only.
//
// Hot path (background + trail FBO + entity sprite blits + wrap mirrors)
// moved to renderer-webgl.js. This file now draws only the UI layers that
// sit ABOVE the WebGL stage on a transparent overlay <canvas id="overlay">:
//
//   - hover ghost (placement preview at cursor)
//   - drag preview (placement ghost, rubber band, handle ring, prediction line)
//   - selection ring (edit mode)
//   - absorbing fallback (per-frame variable alpha — too dynamic to cache)
//
// V9.0b will migrate every layer here to WebGL shaders and delete this file
// entirely. Until then, the overlay 2D canvas is cleared each frame by
// main.js before drawOverlay() runs.

import { state } from './state.js';
import { resolveDisplayColor } from './entities.js';

const SELECT_RING_COLOR = '#6b8cff';
const PREDICTION_DASH = [6, 6];
const PREDICTION_BATCHES = 8;   // sub-segments for fade-along-path effect
const RUBBER_BAND_DASH = [5, 4];
const HANDLE_LINE_WIDTH = 2;
const GHOST_FILL_ALPHA = 0.18;  // faint preview at placement point

// ─── Public: draw the overlay (Canvas2D) ──────────────────────────
// Caller (main.js) is responsible for clearing the overlay canvas to
// transparent before calling this each frame.
export function drawOverlay(ctx) {
  drawHoverGhost(ctx);
  drawDragPreview(ctx);
  drawAbsorbingEntities(ctx);
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

  ctx.globalAlpha = GHOST_FILL_ALPHA;
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

function drawDragPreview(ctx) {
  const drag = state.drag;
  if (!drag) return;

  const previewColor = resolveDisplayColor(
    state.pending.type, state.pending.charge, drag.previewBaseColor,
  );
  const radius = state.pending.radius;

  // 1. Ghost circle at the *placement* point (where the body will spawn).
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

  // 2. Dashed "rubber band" from placement point to cursor.
  ctx.strokeStyle = previewColor;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(RUBBER_BAND_DASH);
  ctx.beginPath();
  ctx.moveTo(drag.startX, drag.startY);
  ctx.lineTo(drag.currentX, drag.currentY);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Hollow handle ring at the cursor.
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
  const path = drag.predictionPath;
  if (path && path.length >= 2) {
    drawDashedFadingPath(ctx, path, previewColor);
  }
}

// Draw a dashed polyline whose alpha decreases along the path. We split it
// into PREDICTION_BATCHES sub-strokes and apply globalAlpha per batch — canvas
// setLineDash can't fade individual dashes. In wrap mode we skip segments that
// jump across the boundary so the line doesn't slash across the viewport.
// `path` is `{ data: Float32Array (interleaved x,y), length: number }`.
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

// Per-frame variable alpha — can't be cached as a sprite. Drawn on overlay
// in V9.0a; will move to a dedicated WebGL shader in V9.0b.
function drawAbsorbingEntities(ctx) {
  for (const e of state.entities) {
    if (!e.absorbing) continue;
    drawAbsorbingFallback(ctx, e, e.x, e.y);
  }
}

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
