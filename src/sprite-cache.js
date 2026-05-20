// sprite-cache.js — LRU cache of pre-rendered entity sprites.
//
// V8.1: instead of redrawing each entity's body + edge + glyph + pinned ring
// via per-call Canvas2D state changes (slow when × 9 mirror copies near a
// boundary corner), we bake each unique (type, color, radius, charge, pinned)
// combination into a small off-screen canvas once. renderer-webgl.js lazily
// uploads each canvas to a GL texture (_spriteTexMap) and draws instances
// via VS_ENTITY / FS_ENTITY — far cheaper than re-issuing arc+fillStyle calls.
//
// Cache strategy: ordered Map (insertion order = LRU). On get-hit, the entry
// is moved to the end (most recent). When the cache exceeds SPRITE_CACHE_MAX
// the oldest key is evicted. Per-entity sprites are also stored on the entity
// itself (`e._sprite`, `e._spriteKey`) so the typical frame hits zero Map
// operations — only when an entity's visual params change does the cache get
// consulted.
//
// Absorbing entities are NOT cached: their alpha changes every frame, which
// would invalidate the sprite each frame. Renderer handles them via the
// fallback arc path.

const SPRITE_CACHE_MAX = 200;
const SPRITE_PADDING_PX = 8;   // room for pinned ring + antialias

const cache = new Map();       // key → HTMLCanvasElement (canvas with ._ox/._oy)

function getSprite(type, color, radius, charge, pinned) {
  // Quantize radius to 0.5 px to coalesce mid-drag slider variants AND so
  // the sprite is rendered at the same radius the cache key reflects (no
  // visual/cached-key divergence).
  const rQ = Math.round(radius * 2) / 2;
  const key = makeKeyQuantized(type, color, rQ, charge, pinned);
  const existing = cache.get(key);
  if (existing) {
    // LRU touch: remove + re-insert moves it to the end.
    cache.delete(key);
    cache.set(key, existing);
    return existing;
  }
  const sprite = renderSprite(type, color, rQ, charge, pinned);
  if (cache.size >= SPRITE_CACHE_MAX) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey !== undefined) cache.delete(oldestKey);
  }
  cache.set(key, sprite);
  return sprite;
}

export function ensureEntitySprite(e) {
  // Skip cache for absorbing entities — alpha is continuously changing.
  if (e.absorbing) {
    e._sprite = null;
    e._spriteKey = null;
    return null;
  }
  const rQ = Math.round(e.radius * 2) / 2;
  const key = makeKeyQuantized(e.type, e.color, rQ, e.charge, e.pinned);
  if (e._spriteKey !== key) {
    e._spriteKey = key;
    e._sprite = getSprite(e.type, e.color, e.radius, e.charge, e.pinned);
  }
  return e._sprite;
}

function makeKeyQuantized(type, color, rQ, charge, pinned) {
  return `${type}|${color}|${rQ}|${charge}|${pinned ? 1 : 0}`;
}

function renderSprite(type, color, radius, charge, pinned) {
  const r = Math.max(0, radius);
  const size = Math.max(2, Math.ceil(r * 2 + SPRITE_PADDING_PX * 2));
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const cx = size * 0.5;
  const cy = size * 0.5;

  // Body
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();

  // Black-hole edge for visibility on dark background.
  if (type === 'black_hole') {
    if (charge === -1) {
      ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    } else {
      ctx.strokeStyle = 'rgba(120,180,255,0.75)';
    }
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Charge sign glyph for non-zero charged planets (≥ 10 px so it reads).
  if (type === 'planet' && charge !== 0 && r >= 10) {
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.font = `${Math.min(r, 18)}px ui-sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(charge > 0 ? '+' : '−', cx, cy);
  }

  // Pinned indicator: yellow dashed ring just outside the body.
  if (pinned) {
    ctx.strokeStyle = '#ffd24d';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Offset = where the entity's logical (x, y) sits within the sprite. With
  // centered geometry, ox/oy = half of canvas size. renderer-webgl.js reads
  // these as `iOffset` per-instance to position the sprite quad correctly.
  canvas._ox = cx;
  canvas._oy = cy;
  return canvas;
}
