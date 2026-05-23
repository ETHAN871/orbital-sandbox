// physics-spatial-hash.js — Wrap-aware uniform-grid broadphase.
//
// V8.2: replaces handleCollisions' O(N²) all-pairs scan with a cell-based
// near-neighbor query. Cell size ≈ 2 × max(radius) so any two overlapping
// bodies must share a cell or be in adjacent cells.
//
// Behaviour:
//   buildSpatialHash(entities) — rebuild grid each substep.
//   forEachCollisionPair(entities, cb) — invoke cb(a, b, dx, dy) once per
//     potentially-colliding unordered pair, with dx/dy already wrap-adjusted
//     (min-image) when state.boundaryMode === 'wrap'. Caller still does the
//     fine distance check (cb may early-out).
//
// Wrap handling: when an entity is near a viewport edge, its 8 neighbour
// cells include cells on the OPPOSITE side of the world. We use modular
// cell indices to look those up; their entities are at "real" coords on
// the far side, and dx/dy are min-image-corrected so the caller sees the
// short-way delta.

import { state } from './state.js';

// Per-axis cell size so cells exactly tile the viewport (no overflow).
// See buildSpatialHash for the rationale — Y wrap-neighbor miss fix.
let _cellSizeX = 64, _cellSizeY = 64;
let _cells = new Map();          // key "cx,cy" → entity[]
let _ncx = 0, _ncy = 0;          // grid dims (used for modular indexing in wrap mode)

export function buildSpatialHash(entities) {
  // Cell size MUST be ≥ 2 × maxR so the 3×3 cell-neighbourhood query
  // covers every pair within r_sum_max = 2 × maxR. Otherwise pairs whose
  // cell distance is 2+ but whose min-image position distance is still <
  // r_sum get silently dropped — manifesting as "one half of a portal-
  // straddling body fails to collide" because the boundary-spanning entity
  // can land 2 cells away from its peer via the wrap path.
  //
  // Bug fix (2026-05-21) — per-axis cellSize so cells exactly tile the
  // viewport. The previous implementation used a single _cellSize and
  // _ncx = ceil(W / _cellSize). When viewport.height isn't a multiple of
  // _cellSize, the last cell extends past the viewport edge and is mostly
  // empty. Modular wrap of the 3×3 query (qcy = ((acy-1) % _ncy + _ncy)
  // % _ncy) then points to that empty overflow cell rather than to the
  // cell that actually contains wrap-adjacent bodies. Result: at the Y
  // wrap edge, broadphase silently dropped wrap-spanning pairs (verified:
  // 17 missed pairs in a 99-body hex cluster at viewport 345×514, all in
  // Y direction; X was fine because viewport.width / cellSize happened to
  // overflow less, putting bodies near the right edge into the cell the
  // modular wrap correctly pointed to). User-visible symptom: bodies
  // overlapping at top/bottom of viewport, escalating Y jitter when BH is
  // enabled (BH dispatch also gates the spatial hash via state.bhThreshold).
  //
  // Fix: count cells with FLOOR(viewport / minCellSize) so each cell is
  // ≥ minCellSize (the 2×maxR guarantee is preserved), then compute the
  // ACTUAL per-axis cellSize as viewport/count. With ncx × cellSizeX
  // exactly = viewport.width, the modular wrap of cell -1 maps to cell
  // ncx-1 which is also the cell containing bodies near the right edge —
  // wrap-adjacency is now correctly found. Same for Y.
  //
  // V8.2 originally capped cellSize at 128 to keep cell count down; that
  // broke collisions for maxR > 64 (entities at max radius 80 had
  // r_sum=160 but cellSize=128). Cap removed — slightly fewer cells but
  // tighter geometric correctness, and per-cell density actually drops
  // with larger cells so the inner k-pair loop is no slower.
  let maxR = 16;
  for (let i = 0; i < entities.length; i++) {
    const r = entities[i].radius;
    if (r > maxR) maxR = r;
  }
  const minCellSize = Math.max(32, Math.ceil(maxR * 2));

  const W = state.viewport.width;
  const H = state.viewport.height;
  // Floor ensures each cell is ≥ minCellSize (preserves the 2×maxR
  // broadphase coverage guarantee). When the viewport is smaller than
  // minCellSize, ncx/ncy clamp to 1 — the entire viewport is one cell
  // and every pair is queried, so broadphase is trivially correct.
  _ncx = Math.max(1, Math.floor(W / minCellSize));
  _ncy = Math.max(1, Math.floor(H / minCellSize));
  // Actual cell size now exactly tiles the viewport per axis.
  _cellSizeX = W > 0 ? W / _ncx : minCellSize;
  _cellSizeY = H > 0 ? H / _ncy : minCellSize;

  _cells.clear();
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const cx = Math.floor(e.x / _cellSizeX);
    const cy = Math.floor(e.y / _cellSizeY);
    const key = cellKey(cx, cy);
    let bucket = _cells.get(key);
    if (!bucket) _cells.set(key, bucket = []);
    bucket.push(e);
  }
}

// Pack signed (cx, cy) into a single integer. Avoids string allocation on
// every Map.get/set in the hot path. Offset 0x8000 (32768) tolerates cell
// indices in [-32768, 32767] which covers any practical viewport / cell size.
function cellKey(cx, cy) { return ((cx + 0x8000) << 16) | (cy + 0x8000); }

// Iterate unordered pairs (a, b) such that a.id < b.id and a/b are in the
// same or adjacent cells. cb receives min-image (dx, dy) from a to b.
export function forEachCollisionPair(entities, cb) {
  const wrap = state.boundaryMode === 'wrap';
  const W = state.viewport.width;
  const H = state.viewport.height;
  const halfW = W * 0.5;
  const halfH = H * 0.5;

  for (let i = 0; i < entities.length; i++) {
    const a = entities[i];
    if (a.absorbing) continue;
    const acx = Math.floor(a.x / _cellSizeX);
    const acy = Math.floor(a.y / _cellSizeY);

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        let qcx = acx + dx;
        let qcy = acy + dy;
        if (wrap) {
          if (_ncx > 0) qcx = ((qcx % _ncx) + _ncx) % _ncx;
          if (_ncy > 0) qcy = ((qcy % _ncy) + _ncy) % _ncy;
        }
        const bucket = _cells.get(cellKey(qcx, qcy));
        if (!bucket) continue;
        for (let k = 0; k < bucket.length; k++) {
          const b = bucket[k];
          // De-dupe pairs and skip self.
          if (a.id >= b.id) continue;
          if (b.absorbing) continue;
          let pdx = b.x - a.x;
          let pdy = b.y - a.y;
          if (wrap) {
            if (pdx >  halfW) pdx -= W; else if (pdx < -halfW) pdx += W;
            if (pdy >  halfH) pdy -= H; else if (pdy < -halfH) pdy += H;
          }
          cb(a, b, pdx, pdy);
        }
      }
    }
  }
}
