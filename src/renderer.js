// renderer.js — All canvas drawing. Pure: reads state, writes pixels.
//
// Layer order (back to front):
//   1. Background
//   2. Historical trails
//   3. Slingshot line + dashed prediction (only during drag)
//   4. Entities (planets + black holes)
//   5. Selected highlight ring

import { state, ABSORPTION_DURATION } from './state.js';
import { resolveDisplayColor } from './entities.js';

const BG_COLOR = '#0a0a0f';
const SELECT_RING_COLOR = '#6b8cff';
const PREDICTION_DASH = [6, 6];
const PREDICTION_BATCHES = 8;   // sub-segments for fade-along-path effect
const RUBBER_BAND_DASH = [5, 4];
const HANDLE_LINE_WIDTH = 2;
const GHOST_FILL_ALPHA = 0.18;  // faint preview at placement point

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
  // Absorbing entities fade alpha alongside the physics-driven shrink so they
  // visibly disappear into the black hole. Use elapsedSim directly so we
  // stay correct if the physics lerp curve ever changes (e.g., easing).
  if (e.absorbing) {
    const t = Math.min(1, e.absorbing.elapsedSim / ABSORPTION_DURATION);
    ctx.globalAlpha = Math.max(0, 1 - t);
  }

  // Body
  ctx.fillStyle = e.color;
  ctx.beginPath();
  ctx.arc(e.x, e.y, Math.max(0, e.radius), 0, Math.PI * 2);
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

  // Restore alpha so the absorbing-entity fade doesn't bleed into siblings.
  if (e.absorbing) ctx.globalAlpha = 1;
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
