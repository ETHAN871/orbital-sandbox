// physics-barneshut.js — O(N log N) gravity via Barnes-Hut on a d3-quadtree.
//
// V8.2: replaces the O(N²) all-pairs computeAccelerations for N ≥ THRESHOLD.
// Builds a fresh quadtree each substep (cheap at typical N), annotates each
// internal node with the signed-charge mass moment (sum of m·q) and its
// charge-weighted center, then accumulates force on every entity by
// recursing the tree with the standard θ criterion:
//   size / dist < THETA  → treat node as a single point (monopole)
//   else                 → recurse into children
//
// Wrap-mode (PBC): standard 8-ghost recursive traversal. For wrap, we
// recursively descend the same quadtree 9 times, each time with one of the
// 9 image offsets (the identity (0,0) and 8 toroidal images at ±W, ±H,
// ±W±H). Tree size stays O(N); only the per-entity traversal cost grows
// by ≤ 9× — but most ghosts get monopolized at distance, so real overhead
// is more like 2-3×.
//
// Charge model: a node's mq = Σ (m · q). When q is signed, mq can cancel —
// that's fine, it's still the right far-field monopole moment for our force
// rule (a_on_target = q_source · G · m_source / r²). The signed sum produces
// the correct net far-field force.

import { quadtree } from 'd3-quadtree';
import { state, G, EPSILON } from './state.js';

const THETA = 0.5;
const THETA2 = THETA * THETA;

// Annotate a freshly-built quadtree with charge-weighted center-of-mass
// for each node, via post-order visit (visitAfter).
function annotateNodes(tree) {
  tree.visitAfter(node => {
    if (node.length) {
      // Internal: combine from children.
      let mq = 0, mqx = 0, mqy = 0, absMq = 0;
      for (let i = 0; i < 4; i++) {
        const child = node[i];
        if (!child) continue;
        mq += child._mq;
        // Weighted by |mq| so cancellation between +/- doesn't make the
        // center degenerate (we still want a sensible CoM for theta).
        const aw = Math.abs(child._mq);
        mqx += child._cx * aw;
        mqy += child._cy * aw;
        absMq += aw;
      }
      node._mq = mq;
      if (absMq > 0) {
        node._cx = mqx / absMq;
        node._cy = mqy / absMq;
      } else {
        node._cx = 0;
        node._cy = 0;
      }
    } else {
      // Leaf: linked-list of data points (may be multiple co-located).
      let mq = 0, mqx = 0, mqy = 0, absMq = 0;
      let q = node;
      while (q) {
        const e = q.data;
        if (e && !e.absorbing && e.charge !== 0) {
          const eMq = e.mass * e.charge;
          mq += eMq;
          const aw = Math.abs(eMq);
          mqx += e.x * aw;
          mqy += e.y * aw;
          absMq += aw;
        }
        q = q.next;
      }
      node._mq = mq;
      if (absMq > 0) {
        node._cx = mqx / absMq;
        node._cy = mqy / absMq;
      } else {
        node._cx = 0;
        node._cy = 0;
      }
    }
  });
}

// Recursively accumulate force on `target` from `node`. `ox,oy` is the
// ghost-image offset added to all source positions in this traversal.
// `x0,y0,x1,y1` are the node's spatial bounds (pre-offset).
function accumulate(node, target, ox, oy, accumX, accumY, x0, y0, x1, y1) {
  if (!node) return;

  // Internal node: theta criterion against center of mass (with offset).
  if (node.length) {
    // NOTE: do NOT short-circuit on node._mq === 0 here. A subtree with net
    // zero charge (e.g. one +1 body and one -1 body of equal mass) is NOT
    // force-neutral in the near-field — the bodies are at different
    // positions so their forces don't cancel from a finite-distance target.
    // The theta criterion handles this correctly: if far enough, monopole
    // contribution is exactly zero (correct); if close, we recurse and
    // sum individual forces (correct). The bug fix moves the zero check
    // INSIDE the monopole branch.
    const ncx = node._cx + ox;
    const ncy = node._cy + oy;
    const dx = ncx - target.x;
    const dy = ncy - target.y;
    const r2 = dx * dx + dy * dy;
    if (r2 <= 0) {
      // Degenerate (target sits on CoM): can't compute force direction.
      // Still need to recurse into children rather than abandon — they
      // may contain real bodies at non-degenerate positions.
    } else {
      const size = Math.max(x1 - x0, y1 - y0);
      if (size * size < THETA2 * r2) {
        // Far enough: treat whole subtree as a monopole. If mq=0 the
        // contribution is zero — skip the math but still return (it's
        // correctly captured as a single-point interaction).
        if (node._mq !== 0) {
          const minR = Math.max(target.radius, EPSILON);
          const r2Soft = Math.max(r2, minR * minR);
          const r = Math.sqrt(r2Soft);
          const mag = node._mq * G / r2Soft;
          accumX.v += mag * dx / r;
          accumY.v += mag * dy / r;
        }
        return;
      }
    }
    // Recurse into the 4 children with the same offset.
    const mx = (x0 + x1) * 0.5;
    const my = (y0 + y1) * 0.5;
    accumulate(node[0], target, ox, oy, accumX, accumY, x0, y0, mx, my);
    accumulate(node[1], target, ox, oy, accumX, accumY, mx, y0, x1, my);
    accumulate(node[2], target, ox, oy, accumX, accumY, x0, my, mx, y1);
    accumulate(node[3], target, ox, oy, accumX, accumY, mx, my, x1, y1);
    return;
  }

  // Leaf: walk the linked list of data points at this coord.
  let q = node;
  while (q) {
    const e = q.data;
    if (e && e !== target && !e.absorbing && e.charge !== 0) {
      const dx = (e.x + ox) - target.x;
      const dy = (e.y + oy) - target.y;
      const minR = Math.max(target.radius + e.radius, EPSILON);
      const r2Raw = dx * dx + dy * dy;
      const r2 = Math.max(r2Raw, minR * minR);
      const r = Math.sqrt(r2);
      const mag = e.charge * G * e.mass / r2;
      accumX.v += mag * dx / r;
      accumY.v += mag * dy / r;
    }
    q = q.next;
  }
}

// Reusable scratch for traversal accumulators (avoids per-call allocation).
const _accumX = { v: 0 };
const _accumY = { v: 0 };

// Per-frame state: tree + extent are built once via prepareBHTree() and
// reused across all substeps within that frame. Position drift between
// substeps is bounded by SIM_DT × |v|, well within Verlet's existing
// integration error.
let _tree = null;
let _extent = null;

export function prepareBHTree(entities) {
  if (entities.length === 0) {
    _tree = null;
    _extent = null;
    return;
  }
  // Filter absorbing entities up-front so they don't bloat the tree (they
  // contribute zero force anyway since the annotate pass sets _mq = 0).
  const sources = [];
  for (let i = 0; i < entities.length; i++) {
    if (!entities[i].absorbing) sources.push(entities[i]);
  }
  const tree = quadtree()
    .x(d => d.x)
    .y(d => d.y)
    .addAll(sources);
  annotateNodes(tree);
  _tree = tree;
  _extent = tree.extent();
}

export function computeAccelerationsBH(entities, accels) {
  const n = entities.length;
  for (let i = 0; i < n; i++) { accels[i].ax = 0; accels[i].ay = 0; }
  if (n === 0 || !_tree || !_extent) return;

  const x0 = _extent[0][0], y0 = _extent[0][1];
  const x1 = _extent[1][0], y1 = _extent[1][1];
  const root = _tree.root();
  if (!root) return;

  const wrap = state.boundaryMode === 'wrap';
  const W = state.viewport.width;
  const H = state.viewport.height;
  // 9 image offsets in wrap mode; just the identity otherwise.
  const offsets = wrap
    ? [[0,0], [W,0], [-W,0], [0,H], [0,-H], [W,H], [W,-H], [-W,H], [-W,-H]]
    : [[0,0]];

  for (let i = 0; i < n; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    _accumX.v = 0; _accumY.v = 0;
    for (let k = 0; k < offsets.length; k++) {
      const ox = offsets[k][0], oy = offsets[k][1];
      accumulate(root, e, ox, oy, _accumX, _accumY, x0 + ox, y0 + oy, x1 + ox, y1 + oy);
    }
    accels[i].ax = _accumX.v;
    accels[i].ay = _accumY.v;
  }
}
