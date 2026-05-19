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

function drawTrails(ctx) {
  if (state.trailLength <= 0) return;
  const W = state.viewport.width;
  const H = state.viewport.height;
  const wrap = state.boundaryMode === 'wrap';
  const wrapX = wrap ? W * 0.5 : Infinity;
  const wrapY = wrap ? H * 0.5 : Infinity;
  for (const e of state.entities) {
    const trail = e.trail;
    if (trail.length < 2) continue;
    ctx.strokeStyle = e.color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    let prevX = trail[0].x, prevY = trail[0].y;
    ctx.moveTo(prevX, prevY);
    for (let i = 1; i < trail.length; i++) {
      const x = trail[i].x, y = trail[i].y;
      // Skip segments that "teleport" across the wrap boundary — those
      // are the visual cousin of the entity's coordinate jump.
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
function drawDashedFadingPath(ctx, path, color) {
  const n = path.length;
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
    let prevX = path[start].x, prevY = path[start].y;
    ctx.moveTo(prevX, prevY);
    for (let i = start + 1; i <= end; i++) {
      const x = path[i].x, y = path[i].y;
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

// Temporarily offset the entity's coords, redraw, then restore. Cheaper
// than refactoring drawOneEntity to take explicit position params, and
// safe because the renderer is single-threaded.
function drawEntityAtMirror(ctx, e, ox, oy) {
  const sx = e.x, sy = e.y;
  e.x = sx + ox; e.y = sy + oy;
  drawOneEntity(ctx, e);
  e.x = sx; e.y = sy;
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

  // Pinned indicator: a yellow dashed ring just outside the body to signal
  // "this body won't move". Suppressed during absorption to avoid clutter.
  if (e.pinned && !e.absorbing) {
    ctx.strokeStyle = '#ffd24d';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(e.x, e.y, Math.max(0, e.radius) + 4, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
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
