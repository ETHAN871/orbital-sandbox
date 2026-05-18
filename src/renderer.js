// renderer.js — All canvas drawing. Pure: reads state, writes pixels.
//
// Layer order (back to front):
//   1. Background
//   2. Historical trails
//   3. Slingshot line + dashed prediction (only during drag)
//   4. Entities (planets + black holes)
//   5. Selected highlight ring

import { state } from './state.js';
import { resolveDisplayColor } from './entities.js';

const BG_COLOR = '#0a0a0f';
const SLINGSHOT_COLOR = 'rgba(255,255,255,0.55)';
const SELECT_RING_COLOR = '#6b8cff';
const PREDICTION_DASH = [6, 6];
const PREDICTION_BATCHES = 8;   // sub-segments for fade-along-path effect

export function drawScene(ctx) {
  const { width, height } = state.viewport;
  drawBackground(ctx, width, height);
  drawTrails(ctx);
  drawDragPreview(ctx);
  drawEntities(ctx);
  drawSelectionRing(ctx);
}

function drawBackground(ctx, w, h) {
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, w, h);
}

function drawTrails(ctx) {
  if (state.trailLength <= 0) return;
  for (const e of state.entities) {
    const trail = e.trail;
    if (trail.length < 2) continue;
    ctx.strokeStyle = e.color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(trail[0].x, trail[0].y);
    for (let i = 1; i < trail.length; i++) ctx.lineTo(trail[i].x, trail[i].y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawDragPreview(ctx) {
  const drag = state.drag;
  if (!drag) return;

  // Slingshot reference line: from start to current pointer.
  ctx.strokeStyle = SLINGSHOT_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 4]);
  ctx.beginPath();
  ctx.moveTo(drag.startX, drag.startY);
  ctx.lineTo(drag.currentX, drag.currentY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Prediction path, color = would-be display color of the entity.
  const path = drag.predictionPath;
  if (!path || path.length < 2) return;

  const previewColor = resolveDisplayColor(
    state.pending.type, state.pending.charge, drag.previewBaseColor,
  );
  drawDashedFadingPath(ctx, path, previewColor);
}

// Draw a dashed polyline whose alpha decreases along the path (head full opacity,
// tail vanishing). We split the path into `PREDICTION_BATCHES` sub-strokes and
// apply globalAlpha per batch — canvas setLineDash can't fade individual dashes.
function drawDashedFadingPath(ctx, path, color) {
  const n = path.length;
  const batchSize = Math.max(2, Math.ceil(n / PREDICTION_BATCHES));
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
    ctx.moveTo(path[start].x, path[start].y);
    for (let i = start + 1; i <= end; i++) ctx.lineTo(path[i].x, path[i].y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
}

function drawEntities(ctx) {
  for (const e of state.entities) {
    drawOneEntity(ctx, e);
  }
}

function drawOneEntity(ctx, e) {
  // Body
  ctx.fillStyle = e.color;
  ctx.beginPath();
  ctx.arc(e.x, e.y, e.radius, 0, Math.PI * 2);
  ctx.fill();

  // Edge treatment so black holes remain visible on dark background.
  if (e.type === 'black_hole') {
    if (e.charge === -1) {
      // White black hole: dark thin rim for contrast
      ctx.strokeStyle = 'rgba(0,0,0,0.55)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else {
      // Black black hole: faint cyan accretion-disk hint
      ctx.strokeStyle = 'rgba(120,180,255,0.75)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // Charge sign glyph (subtle): a faint +/- in the middle for non-zero charges on planets.
  if (e.type === 'planet' && e.charge !== 0 && e.radius >= 10) {
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.font = `${Math.min(e.radius, 18)}px ui-sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(e.charge > 0 ? '+' : '−', e.x, e.y);
  }
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
