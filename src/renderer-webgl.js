// renderer-webgl.js — WebGL 2 renderer (full pipeline).
//
// Every visible pixel is produced by a WebGL 2 shader. (A previous Canvas2D
// overlay path was retired; this is the only renderer.)
//
// Layer order (back to front):
//   1. clear with state.bgColor                              (drawScene)
//   2. trail FBO blit (straight-alpha blend)                 (drawScene)
//   3. entity sprite instanced draw + 9-ghost wrap mirrors   (drawScene)
//   --- UI overlay layers below run via drawUI() ---
//   4. hover ghost (placement preview)                       (drawUI)
//   5. drag preview ghost circle + rubber band + handle ring (drawUI)
//   6. prediction path (dashed, faded-along-arc, wrap-aware) (drawUI)
//   7. absorbing entities (fading circle + optional ring)    (drawUI)
//   8. selection ring (edit mode)                            (drawUI)
//
// V9.0b new shader programs (vs V9.0a):
//   - _progCircleFill: instanced filled disks with 1-px AA edge. Used for
//     hover/drag ghost fill and the absorbing-entity fading body.
//   - _progCircleRing: instanced ring strokes (solid or dashed) with AA on
//     both edges. Used for hover ghost outline (dashed 3-3), drag ghost
//     outline (solid), handle ring (solid), selection ring (dashed 4-4),
//     black-hole absorbing edge (solid).
//   - _progLineSeg: instanced line-segment quads (solid or dashed, with
//     per-instance arc-length offset for continuous dash phase across a
//     polyline). Used for the drag rubber band (dashed 5-4) and the
//     prediction path (dashed 6-6 with per-batch alpha fade).
//
// Trail FBO design:
//   - Two RGBA8 textures at viewport CSS-pixel resolution (no DPR upscale,
//     saves 4× memory; linear sampler on final blit upsamples to backing).
//   - Per frame:
//       a. Bind FBO_write.
//       b. Decay pass: sample FBO_read, output = vec4(rgb, max(0, a - uDec)).
//       c. Dot plot pass: instanced quads at each non-absorbing entity,
//          fragment computes alpha via solid-interior + 1-px AA edge ramp.
//          Blend equation gl.MAX (per-channel) emulates V8.1c "newer wins"
//          semantics. Per-channel mix at intersections is acceptable.
//       d. Swap read/write index.
//   - Composite pass: sample FBO_read, alpha-blend onto screen.
//
// Entity sprite design:
//   - sprite-cache.js continues to produce HTMLCanvasElement sprites.
//   - First time we see a unique canvas, lazily upload to a GL texture
//     (kept in _spriteTexMap). Sprite cache caps at 200 → ≤200 textures
//     live at steady state, ~16MB total VRAM upper bound.
//   - GPU-texture eviction: sprite-cache.js invokes the eviction listener
//     registered in initWebGL when its LRU drops a canvas; we call
//     gl.deleteTexture on the orphan and remove the _spriteTexMap entry,
//     so long-session texture turnover doesn't leak GPU/JS memory.
//   - Per frame: bucket non-absorbing entities by their sprite canvas;
//     build a per-instance VBO of [x, y, w, h, ox, oy, alpha]; issue one
//     instanced draw per bucket; if wrap mode is on and the entity
//     straddles an edge, additional mirror instances are added to the VBO
//     before the draw.
//
// Coordinate space:
//   - Render in CSS-pixel logical coords. Vertex shader transforms via a
//     uniform 4×4 ortho matrix that maps [0..W] × [0..H] → NDC.
//   - gl.viewport(0, 0, W*DPR, H*DPR) every frame; backing store is DPR-
//     scaled per setupCanvas in main.js.

import { state } from './state.js';
import { ensureEntitySprite, onSpriteEvicted } from './sprite-cache.js';
import { resolveDisplayColor } from './entities.js';
import {
  TRAIL_DECAY, TRAIL_BLIT, TRAIL_DOT, ENTITY,
  CIRCLE_FILL, CIRCLE_RING, LINE_SEG,
  EQUIPOTENTIAL, STREAMLINE, GRID_WARP, RUBBER_SHEET_FS, PARTICLE_FLOW,
  SCREEN_DENT,
} from './shaders.js';
import { computePotentialAt, computeForceDirAt } from './potential.js';
import { computeFieldLines } from './field-lines.js';

// ─── Module state ─────────────────────────────────────────────────
let _gl = null;
let _canvas = null;
let _disabled = false;          // true once init failed irrecoverably

// Cached viewport
let _vpW = 0;                   // CSS px
let _vpH = 0;                   // CSS px
let _dpr = 1;
const _orthoMat = new Float32Array(16);

// Shader programs (each is { prog, ...attribLocs, ...uniformLocs })
let _progTrailDecay = null;
let _progTrailDot = null;
let _progTrailBlit = null;
let _progEntity = null;
// V9.0b UI shaders:
let _progCircleFill = null;
let _progCircleRing = null;
let _progLineSeg = null;
// V9.1 field-visualization shaders:
let _progEquipotential = null;
let _progStreamline = null;
// V9.2 (2026-05-26): grid-warp field viz. Replaces _progEquipotential
// as the default; _progEquipotential is kept compiled for ?field=legacy.
let _progGridWarp = null;
// Punched fly-screen field — default. Drawn on the shared grid LINE mesh.
let _progScreenDent = null;
// V12 (2026-05-28): full-screen fragment-shader rubber-sheet renderer.
// Replaces the GRID_WARP indexed-LINE mesh for state.fieldStyle ===
// 'rubber-sheet' specifically. Modes '2d' and '3d' continue to use
// _progGridWarp unchanged. No mesh = no finite extent = no boundary
// visible at any warp magnitude (the original V200 skirt's hard limit).
// See shaders.js RUBBER_SHEET_FS for the full design rationale.
let _progRubberSheetFS = null;
let _bufGridVerts = null;
let _bufGridIndices = null;
let _gridVertexCount = 0;
let _gridIndexCount = 0;
let _gridCols = 0;
let _gridRows = 0;
let _gridCellPx = 8;   // grid spacing — uploaded as uCellPx for the anti-fold cap
// V11.2 painter's algorithm caches. Static segment midpoints + index
// pairs are precomputed once in _rebuildGridWarpVerts; per-frame the
// sag at each midpoint is bilinear-sampled from _sagPixels (CPU side),
// segment screen-y mid is computed, and segments are sorted back-to-
// front so foreground mesh lines overpaint the lines behind them.
let _gridSegCount = 0;                       // number of line segments
let _gridSegMidWorld = new Float32Array(0);  // (mx, my) per seg — static
let _gridIndexPairs  = new Uint32Array(0);   // (i0, i1) per seg — static
let _gridSortKeys    = new Float32Array(0);  // screen-y mid per seg — dynamic
let _gridSortPerm    = null;                 // Array<number> of seg indices, sorted
let _gridIndicesScratch = new Uint32Array(0); // reordered index buffer for upload

// V9.2 sparse particle-flow overlay (companion to grid warp). CPU
// advection of ~200 particles per frame; rendered as additive-blended
// soft point sprites for the "luminous dust" aesthetic.
let _progParticleFlow = null;
let _bufParticles = null;       // interleaved (x, y, age) Float32Array
const PARTICLE_COUNT = 240;
const _particleData = new Float32Array(PARTICLE_COUNT * 3);  // x,y,age × N
const _particleVel = new Float32Array(PARTICLE_COUNT * 2);   // vx,vy × N
const _particleLife = new Uint16Array(PARTICLE_COUNT);       // frames-remaining
let _particlesSeeded = false;

// V11 (2026-05-27): rubber-sheet sag texture, fixed-scale model.
// CPU computes sag(x,y) = |φ(x,y)| / G_vert directly — no per-frame
// normalization, no EMA, no clamp. The slope ∇h then equals g_field/G_vert,
// so a body on the slope feels the same horizontal force the 2D sim
// applies at that point. G_vert tracks state.G and state.epsilon so the
// visual depth at the reference body (mass = m_ref) equals D_TARGET_SAG_PX
// regardless of those sliders. 256×256 R32F texture, uploaded each frame
// when state.fieldStyle === 'rubber-sheet'. All entity / UI / trail
// shaders sample via uSagTex and add screen_y += sag * sin(45°).
const _SAG_TEX_W = 256;
const _SAG_TEX_H = 256;
const _D_TARGET_SAG_PX = 120.0;    // visual depth at the reference body
const _SAG_REF_MASS    = 100.0;    // reference mass (default slider value)
let _sagTexture = null;            // GL texture handle (R32F)
const _sagPixels = new Float32Array(_SAG_TEX_W * _SAG_TEX_H);
let _sagActive = false;            // true iff this frame uses sag — drives uSagMode
// V11.2: per-body quadratic coefficients (a, c, R²) for inside-radius
// C¹ interpolation. 3 floats per body, grown on demand. Float64 keeps
// the (3R²/2 + ε²)/(R²+ε²)^(3/2) ratio precise at small radii.
let _quadCache = new Float64Array(64 * 3);

// Buffers
let _bufFsQuad = null;          // fullscreen quad NDC (vec2 in [-1,+1])
let _bufUnitQuad = null;        // unit quad centered (vec2 in [-1,+1]) for trail dots
let _bufEntityCornerQuad = null; // entity per-vertex (vec2 in [0,1])
let _bufInstanceTrail = null;   // dynamic per-instance VBO for trail dots
let _bufInstanceEntity = null;  // dynamic per-instance VBO for entity sprites
// V9.0b UI buffers:
let _bufInstanceCircleFill = null; // dynamic per-instance VBO for filled disks
let _bufInstanceCircleRing = null; // dynamic per-instance VBO for ring strokes
let _bufInstanceLineSeg = null;    // dynamic per-instance VBO for line segments
// V9.1 streamline buffer (one instance per seed; rebuilt per frame).
let _bufInstanceStreamline = null;

// VAOs (cached attrib+vbo state)
let _vaoFsQuad = null;
let _vaoTrailDot = null;
let _vaoEntity = null;
let _vaoCircleFill = null;
let _vaoCircleRing = null;
let _vaoLineSeg = null;
let _vaoStreamline = null;

// Trail ping-pong FBOs/textures
let _fboA = null, _fboB = null;
let _texA = null, _texB = null;
let _fboW = 0, _fboH = 0;       // FBO size in CSS px
let _trailReadIdx = 0;          // 0 → A is read, B is write; 1 → B is read, A is write

// Sprite canvas (HTMLCanvasElement) → { tex, w, h, ox, oy }
const _spriteTexMap = new Map();

// Reusable scratch arrays sized once; grown as needed.
let _trailInstanceData = new Float32Array(0);  // (cx, cy, r, g, b) × N
let _entityInstanceData = new Float32Array(0); // (cx, cy, w, h, ox, oy, alpha) × N
// V9.0b UI scratch arrays. Each frame drawUI() pushes layer data into these
// then issues one instanced draw per shader. Layout per element:
//   circleFill: (cx, cy, radius, r, g, b, a)               × 7 floats
//   circleRing: (cx, cy, radius, r, g, b, a, lineW, dashOn, dashPeriod)
//                                                          × 10 floats
//   lineSeg:    (x0, y0, x1, y1, r, g, b, a, arcStart, lineW, dashOn,
//                dashPeriod)                               × 12 floats
let _circleFillData = new Float32Array(0);
let _circleFillCount = 0;
let _circleRingData = new Float32Array(0);
let _circleRingCount = 0;
let _lineSegData = new Float32Array(0);
let _lineSegCount = 0;

// V9.1 field-visualization state.
// MAX_FIELD_ENTITIES must match the `#define MAX_ENTITIES` in
// EQUIPOTENTIAL.FS — see shaders.js. Entities beyond this cap are dropped
// from the field calculation (the field becomes visually approximate, but
// the rendered entities themselves still draw normally via _drawEntities).
const MAX_FIELD_ENTITIES = 128;
// Packed entity data uploaded to EQUIPOTENTIAL.FS uEntities[] each frame.
// Layout per slot: (x, y, G·q·m, 0) = 4 floats (vec4 alignment).
const _fieldEntityData = new Float32Array(MAX_FIELD_ENTITIES * 4);
let _fieldEntityCount = 0;
// Grid of streamline seeds — recomputed on resizeRenderer. Each seed is
// (x, y) in CSS px; the force direction is computed per frame.
let _streamlineSeeds = new Float32Array(0); // (x, y) × N
let _streamlineSeedCount = 0;
// Per-instance VBO scratch: (seedX, seedY, dirX, dirY, alpha) = 5 floats.
let _streamlineInstanceData = new Float32Array(0);
// Wall-clock anchor for the synchronized pulse (ms). Pulse phase derives
// from (now - _pulseStartMs) % UI_PULSE_PERIOD_MS.
let _pulseStartMs = 0;
// V9.x: exponential moving averages for the two log-contour parameters.
// `uContourThreshold` is the |φ| of the outermost ring (largest visible
// distance from any emitter); `uLogK` is ln(k) where k is the inter-ring
// |φ| ratio. Both are recomputed each frame from the scene's mass
// extremes — without EMA they jitter slightly as entities move (sample
// points catch slightly different φ peaks frame to frame), sliding every
// contour line ±1 px. EMA with τ ≈ 12 frames (~200 ms) tracks gross
// scene changes but smooths per-frame noise. Snap to instant value when
// zero (first frame / after Clear Sandbox / scene became empty).
let _smoothedContourThreshold = 0;
let _smoothedLogK = 0;

// Shader source strings live in src/shaders.js. Each program exports a
// `{ VS, FS }` pair. See that file for per-program attribute layouts and
// shader algorithm notes. Edits to a shader's attribute set must be
// mirrored here in `_init*` (attrib pointer setup) and `_push*` (scratch-
// array layout) to keep the byte offsets consistent.

// ─── Init ─────────────────────────────────────────────────────────

export function initWebGL(canvas) {
  _canvas = canvas;
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
    premultipliedAlpha: true,
  });
  if (!gl) {
    _showError('WebGL 2 不可用：请使用支持 WebGL 2 的现代浏览器（Chrome / Firefox / Edge / Safari 15+）。');
    _disabled = true;
    return false;
  }
  _gl = gl;

  // Hook sprite-cache LRU eviction so we release the matching GL texture
  // + _spriteTexMap entry instead of letting them linger until context
  // loss / page refresh. Registered once per initWebGL — listener slot in
  // sprite-cache is single-valued so re-init (post-context-loss) replaces
  // any prior closure capturing a stale `_gl`.
  onSpriteEvicted((canvas) => {
    const info = _spriteTexMap.get(canvas);
    if (!info) return;
    if (_gl && info.tex) _gl.deleteTexture(info.tex);
    _spriteTexMap.delete(canvas);
  });

  // Listen for context-loss recovery so a GPU hiccup doesn't kill the canvas.
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    console.warn('[renderer-webgl] context lost');
    // Null _gl during the lost window so the sprite-eviction listener
    // (registered above) short-circuits on its `if (_gl && info.tex)` guard
    // instead of calling deleteTexture on a stale context object. After
    // contextrestored re-runs initWebGL, _gl is reassigned to the fresh
    // context, and a new listener closure replaces the stale one in
    // sprite-cache.js (its slot is single-valued).
    _gl = null;
    _disabled = true;
  });
  canvas.addEventListener('webglcontextrestored', () => {
    console.info('[renderer-webgl] context restored — rebuilding GL resources');
    _disabled = false;
    _spriteTexMap.clear();
    // After context loss, every GL handle (program, buffer, VAO, FBO,
    // texture) the previous context held is INVALID — the driver no longer
    // recognizes them. We must drop the JS references so _initPrograms /
    // _initBuffers / _initFbos start from a clean slate. Without this nil-
    // out, _initBuffers would call gl.bindVertexArray(_vaoTrailDot) where
    // _vaoTrailDot is a stale handle, triggering an INVALID_OPERATION error.
    _progTrailDecay = _progTrailBlit = _progTrailDot = _progEntity = null;
    _progCircleFill = _progCircleRing = _progLineSeg = null;
    _progEquipotential = _progStreamline = null;
    _progGridWarp = _progRubberSheetFS = _progParticleFlow = null;
    _progScreenDent = null;
    _bufFsQuad = _bufUnitQuad = _bufEntityCornerQuad = null;
    _bufInstanceTrail = _bufInstanceEntity = null;
    _bufInstanceCircleFill = _bufInstanceCircleRing = _bufInstanceLineSeg = null;
    _bufInstanceStreamline = null;
    _vaoFsQuad = _vaoTrailDot = _vaoEntity = null;
    _vaoCircleFill = _vaoCircleRing = _vaoLineSeg = null;
    _vaoStreamline = null;
    _fboA = _fboB = _texA = _texB = null;
    try {
      _initPrograms();
      _initBuffers();
      _initFbos(_vpW, _vpH);
      _rebuildStreamlineSeeds();
    } catch (err) {
      _showError('WebGL 上下文恢复失败：' + (err && err.message ? err.message : err));
      console.error('[renderer-webgl] restore failed', err);
      _disabled = true;
    }
  });

  try {
    _initPrograms();
    _initBuffers();
  } catch (err) {
    _showError('WebGL 初始化失败：' + (err && err.message ? err.message : err));
    console.error('[renderer-webgl] init failed', err);
    _disabled = true;
    return false;
  }

  return true;
}

function _showError(msg) {
  const div = document.getElementById('webgl-error');
  if (div) {
    div.textContent = msg;
    div.hidden = false;
  } else {
    console.error('[renderer-webgl]', msg);
  }
}

function _initPrograms() {
  const gl = _gl;
  _progTrailDecay = _makeProgram(TRAIL_DECAY.VS, TRAIL_DECAY.FS);
  _progTrailBlit  = _makeProgram(TRAIL_BLIT.VS,  TRAIL_BLIT.FS);
  _progTrailDot   = _makeProgram(TRAIL_DOT.VS,   TRAIL_DOT.FS);
  _progEntity     = _makeProgram(ENTITY.VS,      ENTITY.FS);

  _progTrailDecay.aPos  = gl.getAttribLocation(_progTrailDecay.prog, 'aPos');
  _progTrailDecay.uTex  = gl.getUniformLocation(_progTrailDecay.prog, 'uTex');
  _progTrailDecay.uDec  = gl.getUniformLocation(_progTrailDecay.prog, 'uDec');

  _progTrailBlit.aPos   = gl.getAttribLocation(_progTrailBlit.prog, 'aPos');
  _progTrailBlit.uTex   = gl.getUniformLocation(_progTrailBlit.prog, 'uTex');

  _progTrailDot.aCorner = gl.getAttribLocation(_progTrailDot.prog, 'aCorner');
  _progTrailDot.iCenter = gl.getAttribLocation(_progTrailDot.prog, 'iCenter');
  _progTrailDot.iColor  = gl.getAttribLocation(_progTrailDot.prog, 'iColor');
  _progTrailDot.uOrtho  = gl.getUniformLocation(_progTrailDot.prog, 'uOrtho');
  _progTrailDot.uRinner = gl.getUniformLocation(_progTrailDot.prog, 'uRinner');
  _progTrailDot.uRouter = gl.getUniformLocation(_progTrailDot.prog, 'uRouter');

  _progEntity.aCorner   = gl.getAttribLocation(_progEntity.prog, 'aCorner');
  _progEntity.iCenter   = gl.getAttribLocation(_progEntity.prog, 'iCenter');
  _progEntity.iSize     = gl.getAttribLocation(_progEntity.prog, 'iSize');
  _progEntity.iOffset   = gl.getAttribLocation(_progEntity.prog, 'iOffset');
  _progEntity.iAlpha    = gl.getAttribLocation(_progEntity.prog, 'iAlpha');
  _progEntity.uOrtho    = gl.getUniformLocation(_progEntity.prog, 'uOrtho');
  _progEntity.uTex      = gl.getUniformLocation(_progEntity.prog, 'uTex');

  // V9.0b UI programs
  _progCircleFill = _makeProgram(CIRCLE_FILL.VS, CIRCLE_FILL.FS);
  _progCircleRing = _makeProgram(CIRCLE_RING.VS, CIRCLE_RING.FS);
  _progLineSeg    = _makeProgram(LINE_SEG.VS,    LINE_SEG.FS);

  _progCircleFill.aCorner = gl.getAttribLocation(_progCircleFill.prog, 'aCorner');
  _progCircleFill.iCenter = gl.getAttribLocation(_progCircleFill.prog, 'iCenter');
  _progCircleFill.iRadius = gl.getAttribLocation(_progCircleFill.prog, 'iRadius');
  _progCircleFill.iColor  = gl.getAttribLocation(_progCircleFill.prog, 'iColor');
  _progCircleFill.uOrtho  = gl.getUniformLocation(_progCircleFill.prog, 'uOrtho');

  _progCircleRing.aCorner     = gl.getAttribLocation(_progCircleRing.prog, 'aCorner');
  _progCircleRing.iCenter     = gl.getAttribLocation(_progCircleRing.prog, 'iCenter');
  _progCircleRing.iRadius     = gl.getAttribLocation(_progCircleRing.prog, 'iRadius');
  _progCircleRing.iColor      = gl.getAttribLocation(_progCircleRing.prog, 'iColor');
  _progCircleRing.iLineW      = gl.getAttribLocation(_progCircleRing.prog, 'iLineW');
  _progCircleRing.iDashOn     = gl.getAttribLocation(_progCircleRing.prog, 'iDashOn');
  _progCircleRing.iDashPeriod = gl.getAttribLocation(_progCircleRing.prog, 'iDashPeriod');
  _progCircleRing.uOrtho      = gl.getUniformLocation(_progCircleRing.prog, 'uOrtho');

  _progLineSeg.aCorner     = gl.getAttribLocation(_progLineSeg.prog, 'aCorner');
  _progLineSeg.iP0         = gl.getAttribLocation(_progLineSeg.prog, 'iP0');
  _progLineSeg.iP1         = gl.getAttribLocation(_progLineSeg.prog, 'iP1');
  _progLineSeg.iColor      = gl.getAttribLocation(_progLineSeg.prog, 'iColor');
  _progLineSeg.iArcStart   = gl.getAttribLocation(_progLineSeg.prog, 'iArcStart');
  _progLineSeg.iLineW      = gl.getAttribLocation(_progLineSeg.prog, 'iLineW');
  _progLineSeg.iDashOn     = gl.getAttribLocation(_progLineSeg.prog, 'iDashOn');
  _progLineSeg.iDashPeriod = gl.getAttribLocation(_progLineSeg.prog, 'iDashPeriod');
  _progLineSeg.uOrtho      = gl.getUniformLocation(_progLineSeg.prog, 'uOrtho');

  // V9.1 field-visualization programs.
  _progEquipotential = _makeProgram(EQUIPOTENTIAL.VS, EQUIPOTENTIAL.FS);
  _progStreamline    = _makeProgram(STREAMLINE.VS,    STREAMLINE.FS);

  _progEquipotential.aPos              = gl.getAttribLocation(_progEquipotential.prog, 'aPos');
  _progEquipotential.uViewport         = gl.getUniformLocation(_progEquipotential.prog, 'uViewport');
  _progEquipotential.uEntities         = gl.getUniformLocation(_progEquipotential.prog, 'uEntities[0]');
  _progEquipotential.uEntityCount      = gl.getUniformLocation(_progEquipotential.prog, 'uEntityCount');
  _progEquipotential.uEpsilon          = gl.getUniformLocation(_progEquipotential.prog, 'uEpsilon');
  _progEquipotential.uLogK             = gl.getUniformLocation(_progEquipotential.prog, 'uLogK');
  _progEquipotential.uContourThreshold = gl.getUniformLocation(_progEquipotential.prog, 'uContourThreshold');
  _progEquipotential.uContourLineW     = gl.getUniformLocation(_progEquipotential.prog, 'uContourLineW');
  _progEquipotential.uDpr              = gl.getUniformLocation(_progEquipotential.prog, 'uDpr');
  _progEquipotential.uColor            = gl.getUniformLocation(_progEquipotential.prog, 'uColor');

  _progStreamline.aCorner       = gl.getAttribLocation(_progStreamline.prog, 'aCorner');
  _progStreamline.iSeed         = gl.getAttribLocation(_progStreamline.prog, 'iSeed');
  _progStreamline.iDir          = gl.getAttribLocation(_progStreamline.prog, 'iDir');
  _progStreamline.iAlpha        = gl.getAttribLocation(_progStreamline.prog, 'iAlpha');
  _progStreamline.uOrtho        = gl.getUniformLocation(_progStreamline.prog, 'uOrtho');
  _progStreamline.uLength       = gl.getUniformLocation(_progStreamline.prog, 'uLength');
  _progStreamline.uLineW        = gl.getUniformLocation(_progStreamline.prog, 'uLineW');
  _progStreamline.uPulseHead    = gl.getUniformLocation(_progStreamline.prog, 'uPulseHead');
  _progStreamline.uPulseTailFrac = gl.getUniformLocation(_progStreamline.prog, 'uPulseTailFrac');
  _progStreamline.uColor        = gl.getUniformLocation(_progStreamline.prog, 'uColor');

  // V9.2 grid-warp program (default field viz).
  _progGridWarp = _makeProgram(GRID_WARP.VS, GRID_WARP.FS);
  _progGridWarp.aPos           = gl.getAttribLocation(_progGridWarp.prog, 'aPos');
  _progGridWarp.uViewport      = gl.getUniformLocation(_progGridWarp.prog, 'uViewport');
  _progGridWarp.uEntities      = gl.getUniformLocation(_progGridWarp.prog, 'uEntities[0]');
  _progGridWarp.uEntityCount   = gl.getUniformLocation(_progGridWarp.prog, 'uEntityCount');
  _progGridWarp.uEpsilon       = gl.getUniformLocation(_progGridWarp.prog, 'uEpsilon');
  _progGridWarp.uDispScale     = gl.getUniformLocation(_progGridWarp.prog, 'uDispScale');
  _progGridWarp.uTiltY         = gl.getUniformLocation(_progGridWarp.prog, 'uTiltY');
  _progGridWarp.uMode          = gl.getUniformLocation(_progGridWarp.prog, 'uMode');
  _progGridWarp.uOrtho         = gl.getUniformLocation(_progGridWarp.prog, 'uOrtho');
  _progGridWarp.uColor         = gl.getUniformLocation(_progGridWarp.prog, 'uColor');
  _progGridWarp.uIntensityMin  = gl.getUniformLocation(_progGridWarp.prog, 'uIntensityMin');
  _progGridWarp.uIntensityMax  = gl.getUniformLocation(_progGridWarp.prog, 'uIntensityMax');
  _progGridWarp.uContrastFloor = gl.getUniformLocation(_progGridWarp.prog, 'uContrastFloor');
  _progGridWarp.uCellPx        = gl.getUniformLocation(_progGridWarp.prog, 'uCellPx');

  // Membrane field program (default field viz). Full-screen relief-lit
  // gravity sheet on the shared VS_FULLSCREEN quad (_vaoFsQuad); see
  // SCREEN_DENT in shaders.js for the technique.
  _progScreenDent = _makeProgram(SCREEN_DENT.VS, SCREEN_DENT.FS);
  _progScreenDent.uViewport      = gl.getUniformLocation(_progScreenDent.prog, 'uViewport');
  _progScreenDent.uEntities      = gl.getUniformLocation(_progScreenDent.prog, 'uEntities[0]');
  _progScreenDent.uEntityCount   = gl.getUniformLocation(_progScreenDent.prog, 'uEntityCount');
  _progScreenDent.uHeightK       = gl.getUniformLocation(_progScreenDent.prog, 'uHeightK');
  _progScreenDent.uCore2         = gl.getUniformLocation(_progScreenDent.prog, 'uCore2');
  _progScreenDent.uWarpGain      = gl.getUniformLocation(_progScreenDent.prog, 'uWarpGain');
  _progScreenDent.uSlope         = gl.getUniformLocation(_progScreenDent.prog, 'uSlope');
  _progScreenDent.uAmbient       = gl.getUniformLocation(_progScreenDent.prog, 'uAmbient');
  _progScreenDent.uCellPx        = gl.getUniformLocation(_progScreenDent.prog, 'uCellPx');
  _progScreenDent.uColor         = gl.getUniformLocation(_progScreenDent.prog, 'uColor');
  _progScreenDent.uOpacity       = gl.getUniformLocation(_progScreenDent.prog, 'uOpacity');

  // V12 (2026-05-28): full-screen FS rubber-sheet renderer. Shares the
  // VS_FULLSCREEN VAO (_vaoFsQuad) with the trail-decay/blit programs.
  // See shaders.js RUBBER_SHEET_FS for the chosen-technique rationale
  // and shader source. Used only for state.fieldStyle === 'rubber-sheet';
  // modes '2d'/'3d' continue to dispatch the mesh GRID_WARP program.
  _progRubberSheetFS = _makeProgram(RUBBER_SHEET_FS.VS, RUBBER_SHEET_FS.FS);
  _progRubberSheetFS.aPos           = gl.getAttribLocation(_progRubberSheetFS.prog, 'aPos');
  _progRubberSheetFS.uSagTex        = gl.getUniformLocation(_progRubberSheetFS.prog, 'uSagTex');
  _progRubberSheetFS.uViewport      = gl.getUniformLocation(_progRubberSheetFS.prog, 'uViewport');
  _progRubberSheetFS.uSagYFactor    = gl.getUniformLocation(_progRubberSheetFS.prog, 'uSagYFactor');
  _progRubberSheetFS.uSagWrap       = gl.getUniformLocation(_progRubberSheetFS.prog, 'uSagWrap');
  _progRubberSheetFS.uCellPx        = gl.getUniformLocation(_progRubberSheetFS.prog, 'uCellPx');
  _progRubberSheetFS.uColor         = gl.getUniformLocation(_progRubberSheetFS.prog, 'uColor');
  _progRubberSheetFS.uContrastFloor = gl.getUniformLocation(_progRubberSheetFS.prog, 'uContrastFloor');
  _progRubberSheetFS.uMaxSag        = gl.getUniformLocation(_progRubberSheetFS.prog, 'uMaxSag');

  // V9.2 particle-flow program (companion overlay).
  _progParticleFlow = _makeProgram(PARTICLE_FLOW.VS, PARTICLE_FLOW.FS);
  _progParticleFlow.aPos       = gl.getAttribLocation(_progParticleFlow.prog, 'aPos');
  _progParticleFlow.aAge       = gl.getAttribLocation(_progParticleFlow.prog, 'aAge');
  _progParticleFlow.uOrtho     = gl.getUniformLocation(_progParticleFlow.prog, 'uOrtho');
  _progParticleFlow.uColor     = gl.getUniformLocation(_progParticleFlow.prog, 'uColor');
  _progParticleFlow.uPointSize = gl.getUniformLocation(_progParticleFlow.prog, 'uPointSize');

  // V10 rubber-sheet: bind sag uniforms on every shader whose vertices
  // need to sink into gravity wells. GRID_WARP is included because the
  // V10 fix added a `uMode == 2` branch that samples the sag texture so
  // the mesh agrees with the bodies on well depth (the legacy uMode == 0
  // per-vertex φ path saturates the depth clamp and produces a flat
  // surface regardless of body count).
  // EQUIPOTENTIAL/STREAMLINE/TRAIL_DECAY/TRAIL_BLIT are field-only /
  // fullscreen passes that stay flat.
  //
  // NOTE: uSagTex/uSagMode/uSagViewport are declared in the VS only (via
  // SAG_VS_HELPER); WebGL2 spec keeps them resolvable as long as the
  // linked program references them anywhere, so getUniformLocation
  // returns valid non-null handles even though the FS never reads them.
  for (const p of [_progTrailDot, _progEntity, _progCircleFill, _progCircleRing, _progLineSeg, _progParticleFlow, _progGridWarp]) {
    p.uSagTex      = gl.getUniformLocation(p.prog, 'uSagTex');
    p.uSagMode     = gl.getUniformLocation(p.prog, 'uSagMode');
    p.uSagViewport = gl.getUniformLocation(p.prog, 'uSagViewport');
    p.uSagYFactor  = gl.getUniformLocation(p.prog, 'uSagYFactor');
    p.uSagWrap     = gl.getUniformLocation(p.prog, 'uSagWrap');
  }
}

// V10: bind sag texture to a texture unit and upload sag uniforms for
// the currently-bound program. Called by every draw path that uses the
// sag-aware shaders. Texture unit 1 reserved for sag (unit 0 used by
// the entity sprite texture binding in _drawEntities; safe to keep
// sag in 1 for the whole frame).
const _SAG_TEX_UNIT = 1;
const _DEG_TO_RAD = Math.PI / 180;
// Guarded cos(viewTilt): clamps to [30°, 90°] and substitutes the
// default 45° if state.viewTilt is non-finite. Prevents a NaN from
// any out-of-band mutation (URL preset, future serializer, …) from
// turning every sag uniform into NaN and silently producing a black
// or scrambled frame with no console error.
function _sagYFactor() {
  const t = state.viewTilt;
  const safe = (Number.isFinite(t) && t >= 30 && t <= 90) ? t : 45;
  return Math.cos(safe * _DEG_TO_RAD);
}
function _bindSagUniforms(prog) {
  if (!_gl) return;
  const gl = _gl;
  gl.activeTexture(gl.TEXTURE0 + _SAG_TEX_UNIT);
  gl.bindTexture(gl.TEXTURE_2D, _sagTexture);
  gl.uniform1i(prog.uSagTex, _SAG_TEX_UNIT);
  gl.uniform1f(prog.uSagMode, _sagActive ? 1.0 : 0.0);
  gl.uniform2f(prog.uSagViewport, _vpW, _vpH);
  // V11.1: uSagYFactor = cos(viewTilt) — projection factor for sag
  // onto screen-Y. viewTilt is in degrees, 30°..90° via UI slider.
  // 90° → 0 (top-down flat), 45° → 0.707 (default), 30° → 0.866.
  gl.uniform1f(prog.uSagYFactor, _sagYFactor());
  // V11.3: uSagWrap selects fract (wrap) vs clamp (bounded) UV sampling.
  // _packFieldEntities packs 9 ghosts ONLY in wrap mode, so the sag
  // texture is toroidally correct only in wrap mode — fract sampling
  // is then valid and gives continuous sag at world edges and across
  // the GRID_EXPAND_FRAC mesh skirt. In bounded mode the texture's
  // content outside the viewport is meaningless, so we clamp to the
  // edge row (no regression vs pre-V11.3 behavior).
  gl.uniform1f(prog.uSagWrap, state.boundaryMode === 'wrap' ? 1.0 : 0.0);
  gl.activeTexture(gl.TEXTURE0);   // restore default
}

function _initBuffers() {
  const gl = _gl;

  // Fullscreen NDC quad (2 triangles, 6 verts)
  _bufFsQuad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufFsQuad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1,  -1,  1,
    -1,  1,  1, -1,   1,  1,
  ]), gl.STATIC_DRAW);

  // Centered unit quad in [-1, +1] for trail dots
  _bufUnitQuad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufUnitQuad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1,  -1,  1,
    -1,  1,  1, -1,   1,  1,
  ]), gl.STATIC_DRAW);

  // Top-left-origin unit quad in [0, 1] for entity sprites
  _bufEntityCornerQuad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufEntityCornerQuad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    0, 0,  1, 0,  0, 1,
    0, 1,  1, 0,  1, 1,
  ]), gl.STATIC_DRAW);

  // Per-instance buffers (filled per frame)
  _bufInstanceTrail = gl.createBuffer();
  _bufInstanceEntity = gl.createBuffer();
  _bufInstanceCircleFill = gl.createBuffer();
  _bufInstanceCircleRing = gl.createBuffer();
  _bufInstanceLineSeg    = gl.createBuffer();
  _bufInstanceStreamline = gl.createBuffer();

  // Fullscreen quad VAO — shared by decay + blit passes. Both their shader
  // programs use the same fullscreen VS (TRAIL_DECAY.VS === TRAIL_BLIT.VS,
  // see shaders.js) with `aPos` at location 0 (enforced via
  // bindAttribLocation in _makeProgram), so a single VAO works for both.
  _vaoFsQuad = gl.createVertexArray();
  gl.bindVertexArray(_vaoFsQuad);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufFsQuad);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(0, 0);
  gl.bindVertexArray(null);

  // Trail dot VAO
  _vaoTrailDot = gl.createVertexArray();
  gl.bindVertexArray(_vaoTrailDot);

  gl.bindBuffer(gl.ARRAY_BUFFER, _bufUnitQuad);
  gl.enableVertexAttribArray(_progTrailDot.aCorner);
  gl.vertexAttribPointer(_progTrailDot.aCorner, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(_progTrailDot.aCorner, 0);

  // Per-instance layout: iCenter (vec2) + iColor (vec3) = 5 floats = 20 bytes stride
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceTrail);
  gl.enableVertexAttribArray(_progTrailDot.iCenter);
  gl.vertexAttribPointer(_progTrailDot.iCenter, 2, gl.FLOAT, false, 20, 0);
  gl.vertexAttribDivisor(_progTrailDot.iCenter, 1);
  gl.enableVertexAttribArray(_progTrailDot.iColor);
  gl.vertexAttribPointer(_progTrailDot.iColor, 3, gl.FLOAT, false, 20, 8);
  gl.vertexAttribDivisor(_progTrailDot.iColor, 1);

  gl.bindVertexArray(null);

  // Entity VAO
  _vaoEntity = gl.createVertexArray();
  gl.bindVertexArray(_vaoEntity);

  gl.bindBuffer(gl.ARRAY_BUFFER, _bufEntityCornerQuad);
  gl.enableVertexAttribArray(_progEntity.aCorner);
  gl.vertexAttribPointer(_progEntity.aCorner, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(_progEntity.aCorner, 0);

  // Per-instance layout: iCenter(2) + iSize(2) + iOffset(2) + iAlpha(1) = 7 floats = 28 bytes stride
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceEntity);
  gl.enableVertexAttribArray(_progEntity.iCenter);
  gl.vertexAttribPointer(_progEntity.iCenter, 2, gl.FLOAT, false, 28, 0);
  gl.vertexAttribDivisor(_progEntity.iCenter, 1);
  gl.enableVertexAttribArray(_progEntity.iSize);
  gl.vertexAttribPointer(_progEntity.iSize, 2, gl.FLOAT, false, 28, 8);
  gl.vertexAttribDivisor(_progEntity.iSize, 1);
  gl.enableVertexAttribArray(_progEntity.iOffset);
  gl.vertexAttribPointer(_progEntity.iOffset, 2, gl.FLOAT, false, 28, 16);
  gl.vertexAttribDivisor(_progEntity.iOffset, 1);
  gl.enableVertexAttribArray(_progEntity.iAlpha);
  gl.vertexAttribPointer(_progEntity.iAlpha, 1, gl.FLOAT, false, 28, 24);
  gl.vertexAttribDivisor(_progEntity.iAlpha, 1);

  gl.bindVertexArray(null);

  // V9.0b: circle-fill VAO (uses centered _bufUnitQuad for aCorner ∈ [-1,+1])
  // Per-instance: (cx, cy, radius, r, g, b, a) = 7 floats = 28 bytes
  _vaoCircleFill = gl.createVertexArray();
  gl.bindVertexArray(_vaoCircleFill);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufUnitQuad);
  gl.enableVertexAttribArray(_progCircleFill.aCorner);
  gl.vertexAttribPointer(_progCircleFill.aCorner, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(_progCircleFill.aCorner, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceCircleFill);
  gl.enableVertexAttribArray(_progCircleFill.iCenter);
  gl.vertexAttribPointer(_progCircleFill.iCenter, 2, gl.FLOAT, false, 28, 0);
  gl.vertexAttribDivisor(_progCircleFill.iCenter, 1);
  gl.enableVertexAttribArray(_progCircleFill.iRadius);
  gl.vertexAttribPointer(_progCircleFill.iRadius, 1, gl.FLOAT, false, 28, 8);
  gl.vertexAttribDivisor(_progCircleFill.iRadius, 1);
  gl.enableVertexAttribArray(_progCircleFill.iColor);
  gl.vertexAttribPointer(_progCircleFill.iColor, 4, gl.FLOAT, false, 28, 12);
  gl.vertexAttribDivisor(_progCircleFill.iColor, 1);
  gl.bindVertexArray(null);

  // V9.0b: circle-ring VAO (uses _bufUnitQuad)
  // Per-instance: (cx, cy, radius, r, g, b, a, lineW, dashOn, dashPeriod)
  // = 10 floats = 40 bytes
  _vaoCircleRing = gl.createVertexArray();
  gl.bindVertexArray(_vaoCircleRing);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufUnitQuad);
  gl.enableVertexAttribArray(_progCircleRing.aCorner);
  gl.vertexAttribPointer(_progCircleRing.aCorner, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(_progCircleRing.aCorner, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceCircleRing);
  gl.enableVertexAttribArray(_progCircleRing.iCenter);
  gl.vertexAttribPointer(_progCircleRing.iCenter, 2, gl.FLOAT, false, 40, 0);
  gl.vertexAttribDivisor(_progCircleRing.iCenter, 1);
  gl.enableVertexAttribArray(_progCircleRing.iRadius);
  gl.vertexAttribPointer(_progCircleRing.iRadius, 1, gl.FLOAT, false, 40, 8);
  gl.vertexAttribDivisor(_progCircleRing.iRadius, 1);
  gl.enableVertexAttribArray(_progCircleRing.iColor);
  gl.vertexAttribPointer(_progCircleRing.iColor, 4, gl.FLOAT, false, 40, 12);
  gl.vertexAttribDivisor(_progCircleRing.iColor, 1);
  gl.enableVertexAttribArray(_progCircleRing.iLineW);
  gl.vertexAttribPointer(_progCircleRing.iLineW, 1, gl.FLOAT, false, 40, 28);
  gl.vertexAttribDivisor(_progCircleRing.iLineW, 1);
  gl.enableVertexAttribArray(_progCircleRing.iDashOn);
  gl.vertexAttribPointer(_progCircleRing.iDashOn, 1, gl.FLOAT, false, 40, 32);
  gl.vertexAttribDivisor(_progCircleRing.iDashOn, 1);
  gl.enableVertexAttribArray(_progCircleRing.iDashPeriod);
  gl.vertexAttribPointer(_progCircleRing.iDashPeriod, 1, gl.FLOAT, false, 40, 36);
  gl.vertexAttribDivisor(_progCircleRing.iDashPeriod, 1);
  gl.bindVertexArray(null);

  // V9.0b: line-segment VAO (uses _bufEntityCornerQuad [0,1] for aCorner)
  // Per-instance: (x0, y0, x1, y1, r, g, b, a, arcStart, lineW, dashOn,
  // dashPeriod) = 12 floats = 48 bytes
  _vaoLineSeg = gl.createVertexArray();
  gl.bindVertexArray(_vaoLineSeg);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufEntityCornerQuad);
  gl.enableVertexAttribArray(_progLineSeg.aCorner);
  gl.vertexAttribPointer(_progLineSeg.aCorner, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(_progLineSeg.aCorner, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceLineSeg);
  gl.enableVertexAttribArray(_progLineSeg.iP0);
  gl.vertexAttribPointer(_progLineSeg.iP0, 2, gl.FLOAT, false, 48, 0);
  gl.vertexAttribDivisor(_progLineSeg.iP0, 1);
  gl.enableVertexAttribArray(_progLineSeg.iP1);
  gl.vertexAttribPointer(_progLineSeg.iP1, 2, gl.FLOAT, false, 48, 8);
  gl.vertexAttribDivisor(_progLineSeg.iP1, 1);
  gl.enableVertexAttribArray(_progLineSeg.iColor);
  gl.vertexAttribPointer(_progLineSeg.iColor, 4, gl.FLOAT, false, 48, 16);
  gl.vertexAttribDivisor(_progLineSeg.iColor, 1);
  gl.enableVertexAttribArray(_progLineSeg.iArcStart);
  gl.vertexAttribPointer(_progLineSeg.iArcStart, 1, gl.FLOAT, false, 48, 32);
  gl.vertexAttribDivisor(_progLineSeg.iArcStart, 1);
  gl.enableVertexAttribArray(_progLineSeg.iLineW);
  gl.vertexAttribPointer(_progLineSeg.iLineW, 1, gl.FLOAT, false, 48, 36);
  gl.vertexAttribDivisor(_progLineSeg.iLineW, 1);
  gl.enableVertexAttribArray(_progLineSeg.iDashOn);
  gl.vertexAttribPointer(_progLineSeg.iDashOn, 1, gl.FLOAT, false, 48, 40);
  gl.vertexAttribDivisor(_progLineSeg.iDashOn, 1);
  gl.enableVertexAttribArray(_progLineSeg.iDashPeriod);
  gl.vertexAttribPointer(_progLineSeg.iDashPeriod, 1, gl.FLOAT, false, 48, 44);
  gl.vertexAttribDivisor(_progLineSeg.iDashPeriod, 1);
  gl.bindVertexArray(null);

  // V9.1: streamline VAO. Per-vertex aCorner ∈ [0,1]² via _bufEntityCornerQuad
  // (x = along, y = perp). Per-instance: (seedX, seedY, dirX, dirY, alpha) =
  // 5 floats = 20 bytes stride.
  _vaoStreamline = gl.createVertexArray();
  gl.bindVertexArray(_vaoStreamline);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufEntityCornerQuad);
  gl.enableVertexAttribArray(_progStreamline.aCorner);
  gl.vertexAttribPointer(_progStreamline.aCorner, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(_progStreamline.aCorner, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceStreamline);
  gl.enableVertexAttribArray(_progStreamline.iSeed);
  gl.vertexAttribPointer(_progStreamline.iSeed, 2, gl.FLOAT, false, 20, 0);
  gl.vertexAttribDivisor(_progStreamline.iSeed, 1);
  gl.enableVertexAttribArray(_progStreamline.iDir);
  gl.vertexAttribPointer(_progStreamline.iDir, 2, gl.FLOAT, false, 20, 8);
  gl.vertexAttribDivisor(_progStreamline.iDir, 1);
  gl.enableVertexAttribArray(_progStreamline.iAlpha);
  gl.vertexAttribPointer(_progStreamline.iAlpha, 1, gl.FLOAT, false, 20, 16);
  gl.vertexAttribDivisor(_progStreamline.iAlpha, 1);
  gl.bindVertexArray(null);
}

function _initFbos(w, h) {
  const gl = _gl;
  if (_fboA) gl.deleteFramebuffer(_fboA);
  if (_fboB) gl.deleteFramebuffer(_fboB);
  if (_texA) gl.deleteTexture(_texA);
  if (_texB) gl.deleteTexture(_texB);
  _fboA = _fboB = _texA = _texB = null;
  _fboW = Math.max(1, w | 0);
  _fboH = Math.max(1, h | 0);

  for (let i = 0; i < 2; i++) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, _fboW, _fboH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('[renderer-webgl] FBO incomplete:', status);
    }
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (i === 0) { _fboA = fbo; _texA = tex; }
    else         { _fboB = fbo; _texB = tex; }
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  _trailReadIdx = 0;
}

// V11 rubber-sheet: one-time sag-texture creation. Allocated at fixed
// 256×256 resolution regardless of viewport; the shader samples in
// world-uv space (worldPx / viewport) so the lookup adapts automatically.
function _initSagTexture() {
  if (!_gl) return;
  if (_sagTexture) return;   // already created
  const gl = _gl;
  // OES_texture_float_linear is REQUIRED for LINEAR filter on R32F.
  // Without it the sampler silently clamps to NEAREST — visible as
  // blocky body positions when sag wells are narrow. Universal on
  // desktop WebGL 2 but worth the explicit getExtension call.
  gl.getExtension('OES_texture_float_linear');
  _sagTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, _sagTexture);
  // R32F: single-channel float. WebGL 2 core supports R32F as a sampled
  // texture format (the EXT_color_buffer_float extension is only required
  // if we wanted to RENDER to it). We only sample, so no extension guard.
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, _SAG_TEX_W, _SAG_TEX_H, 0, gl.RED, gl.FLOAT, null);
  // Linear filtering — gives smooth sag values between texels.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  // V11: REPEAT (not CLAMP_TO_EDGE) so wrap-mode ghost copies at world
  // X > vpW or Y > vpH sample the correct tile of the sag field. The
  // CPU pass already includes 9-ghost contributions when boundary mode
  // is 'wrap', so the texture is tileable by construction. In bounded
  // mode the visible viewport stays in UV [0,1] anyway, so REPEAT has
  // no visible effect there.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.bindTexture(gl.TEXTURE_2D, null);
}

// Per-frame: sample sag(x, y) at 256×256 grid points using the entity
// data already packed in _fieldEntityData (includes 9-ghost copies in
// wrap mode). Upload via texSubImage2D so the GPU shaders can sample.
//
// V11 (2026-05-27 — current): h(x,y) = |φ(x,y)| / G_vert. Fixed scale
// with G_vert calibrated live from state.G and state.epsilon so a
// reference body of mass=_SAG_REF_MASS produces a well of depth
// =_D_TARGET_SAG_PX. No normalization, no EMA, no clamp — slopes are
// a direct function of the simulated field strength. xy plane is
// preserved exactly (height is a pure vertical rescaling of φ).
//
// History (kept for reference):
//   - V10.0: constant scale × |φ| → saturated cap everywhere within
//     a body footprint (|φ|~hundreds, ×5 hit cap immediately).
//   - V10.1: per-frame |φ_max| EMA rescale → surface dynamic but
//     bodies bobbed as orbit moved across 128² sample boundaries.
//   - V10.2: anchored EMA to position-invariant baseline G·max|qm|/ε
//     to kill scale-induced bobbing; texel aliasing residual remained.
//   - V11: drop EMA entirely. Scale is f(state.G, state.epsilon) only.
//     Bumping texture to 256² halves residual texel-stride wobble.
function _updateSagTexture() {
  if (!_gl || !_sagTexture) return;
  const gl = _gl;
  const ents = _fieldEntityData;
  const cnt = _fieldEntityCount;
  const eps = Math.max(1, state.epsilon);   // guard ÷0 if ε slider is at floor
  const eps2 = eps * eps;
  const sx = _vpW / _SAG_TEX_W;
  const sy = _vpH / _SAG_TEX_H;
  if (cnt === 0) {
    _sagPixels.fill(0);
  } else {
    // G_vert: vertical-gravity coefficient calibrated so the reference
    // body (mass=_SAG_REF_MASS=100 at default ε) produces a well of
    // depth _D_TARGET_SAG_PX=120 CSS px. Derivation:
    //   sag at body center = |φ_self|/G_vert = (G·m_ref/ε)/G_vert
    //   want this = D_TARGET → G_vert = G·m_ref / (D_TARGET·ε)
    // Tracking state.G and state.epsilon live keeps the visual depth
    // calibrated through any slider change, without per-frame rescale.
    // The result: h(x,y) = |φ(x,y)|/G_vert — a fixed-scale rubber sheet
    // whose slope ∇h equals the simulated gravity g_field at every point.
    const G_vert = (state.G * _SAG_REF_MASS) / (_D_TARGET_SAG_PX * eps);
    const invG_vert = 1 / G_vert;
    // V11.2 (2026-05-27): inside-radius C¹ quadratic interpolation.
    //
    // Plummer φ(r) = -Gqm / sqrt(r² + ε²) has a steep gradient near
    // body centers (max slope around r≈ε). When the body sits at a
    // sub-texel position relative to the 256² sag grid, the bilinear-
    // interpolated sag at the body's exact location changes sharply
    // as the body crosses texel boundaries — visible as orbital
    // jitter / vibration.
    //
    // Physical reality: bodies are solid spheres of radius R. Other
    // bodies can never get closer than R, so the field strength
    // INSIDE R is unphysical anyway. Replace it with a smooth quadratic
    //   f(r) = -(a + c·r²)
    // matching three constraints:
    //   1.  f(R)  = φ(R)        (boundary value continuous)
    //   2.  f'(R) = φ'(R)       (boundary slope continuous, C¹)
    //   3.  f'(0) = 0           (smooth peak at center — radial RBF)
    // Solving:
    //   rr2  = R² + ε²
    //   rr32 = (R² + ε²)^(3/2)
    //   c    = -Gqm / (2·rr32)                        (signed, < 0 for attractor)
    //   a    = Gqm · (3R²/2 + ε²) / rr32              (signed)
    // Inside the body |φ| has zero gradient at r=0 → body's apparent
    // sag is O(Δr²)-invariant to sub-pixel position drift → no jitter.
    //
    // Precompute (a, c, R²) per body once; the hot pixel loop just
    // does one squared-distance compare per (pixel, body) pair.
    if (_quadCache.length < cnt * 3) {
      _quadCache = new Float64Array(Math.max(cnt * 3, _quadCache.length * 2));
    }
    for (let k = 0; k < cnt; k++) {
      const o = k * 4;
      const Gqm = ents[o + 2];
      const R = ents[o + 3];
      const R2 = R * R;
      const rr2 = R2 + eps2;
      const rr32 = rr2 * Math.sqrt(rr2);
      // R=0 (theoretical degenerate, doesn't occur today since
      // _packFieldEntities skips absorbing bodies whose radius shrinks
      // to 0): rr2 → eps2 > 0, so no div-by-zero. The quadratic branch
      // condition `r2raw < R²=0` is then always false → coefficients
      // computed but unused → falls through to Plummer for all pixels.
      // Conservative, correct.
      _quadCache[k * 3]     = Gqm * (1.5 * R2 + eps2) / rr32;   // a
      _quadCache[k * 3 + 1] = -Gqm / (2 * rr32);                 // c
      _quadCache[k * 3 + 2] = R2;                                // R²
    }
    let idx = 0;
    for (let j = 0; j < _SAG_TEX_H; j++) {
      const y = (j + 0.5) * sy;
      for (let i = 0; i < _SAG_TEX_W; i++) {
        const x = (i + 0.5) * sx;
        let phi = 0;
        for (let k = 0; k < cnt; k++) {
          const o = k * 4;
          const dx = x - ents[o];
          const dy = y - ents[o + 1];
          const r2raw = dx * dx + dy * dy;
          const qo = k * 3;
          const R2 = _quadCache[qo + 2];
          if (r2raw < R2) {
            // Inside body's collision radius → smooth quadratic.
            // f(r) = -(a + c·r²), where r² is unsoftened (raw).
            phi -= _quadCache[qo] + _quadCache[qo + 1] * r2raw;
          } else {
            // Outside → standard Plummer (with ε² softening).
            phi += -ents[o + 2] / Math.sqrt(r2raw + eps2);
          }
        }
        const phiAbs = phi < 0 ? -phi : phi;
        _sagPixels[idx++] = phiAbs * invG_vert;
      }
    }
  }
  gl.bindTexture(gl.TEXTURE_2D, _sagTexture);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, _SAG_TEX_W, _SAG_TEX_H, gl.RED, gl.FLOAT, _sagPixels);
}

export function resizeRenderer(w, h, dpr) {
  if (_disabled || !_gl) return;
  _vpW = Math.max(1, w | 0);
  _vpH = Math.max(1, h | 0);
  _dpr = dpr || 1;
  _updateOrthoMat();
  _initFbos(_vpW, _vpH);
  _initSagTexture();
  _rebuildStreamlineSeeds();
  _rebuildGridWarpVerts();
}

// V9.2: build a regular grid of vertices covering the viewport, plus an
// index buffer of line-segment endpoints connecting each vertex to its
// right and down neighbors. Resolution targets ~5000 vertices = a good
// trade between visual density and per-frame vertex cost.
// Called on resize and on first init.
//
// V9.3 borderless: grid extends GRID_EXPAND_FRAC past the viewport on
// every side so the unwarped grid edge sits off-screen. The shader's
// vEdgeFade smoothstep on the ORIGINAL aPos fades alpha to 0 between
// the visible viewport edge and the rebuilt grid edge, so the user
// never sees a rectangular boundary — the field reads as an infinite
// warped plane fading into the background.
function _rebuildGridWarpVerts() {
  if (!_gl || _vpW <= 0 || _vpH <= 0) return;
  const gl = _gl;
  // Aim for ~80 columns at typical aspect; rows scale to viewport.
  const TARGET_COLS = 80;
  const cellPx = Math.max(8, _vpW / TARGET_COLS);
  // Extend grid past viewport on all sides — fade handled in shader.
  // 0.15 → grid is 30% wider+taller; ~1.69× total vertex count.
  // Vertex shader's vEdgeFade smoothstep(0.85, 1.15) matches this.
  const GRID_EXPAND_FRAC = 0.15;
  const extW = _vpW * (1 + 2 * GRID_EXPAND_FRAC);
  const extH = _vpH * (1 + 2 * GRID_EXPAND_FRAC);
  const offX = -_vpW * GRID_EXPAND_FRAC;
  const offY = -_vpH * GRID_EXPAND_FRAC;
  const cols = Math.max(8, Math.round(extW / cellPx) + 1);
  const rows = Math.max(8, Math.round(extH / cellPx) + 1);
  _gridCols = cols;
  _gridRows = rows;
  _gridCellPx = cellPx;   // shader uses this for the anti-fold cap (uCellPx)
  const vertCount = cols * rows;
  _gridVertexCount = vertCount;
  // Vertex positions in CSS-px. Span [offX, offX+extW] × [offY, offY+extH].
  // Visible viewport is [0, _vpW] × [0, _vpH] — vertices outside that
  // window fade to alpha 0 in the shader (vEdgeFade).
  const verts = new Float32Array(vertCount * 2);
  let idx = 0;
  for (let r = 0; r < rows; r++) {
    const y = offY + (r / (rows - 1)) * extH;
    for (let c = 0; c < cols; c++) {
      const x = offX + (c / (cols - 1)) * extW;
      verts[idx++] = x;
      verts[idx++] = y;
    }
  }
  // Index buffer: for each cell, add a horizontal segment (this vert →
  // right neighbor) and a vertical segment (this vert → down neighbor).
  // Last column gets no horizontal; last row gets no vertical.
  const numHorizontal = (cols - 1) * rows;
  const numVertical = cols * (rows - 1);
  const segCount = numHorizontal + numVertical;
  const idxCount = segCount * 2;
  _gridIndexCount = idxCount;
  _gridSegCount   = segCount;
  // Use Uint32 — index counts can exceed 65535 (e.g. 80×60 = 4800 verts,
  // ~9500 line endpoints).
  const indices = new Uint32Array(idxCount);
  // V11.2: parallel caches for painter's algorithm.
  //   _gridIndexPairs   — (i0, i1) per segment, immutable, original order.
  //   _gridSegMidWorld  — (mx, my) world-space midpoint, immutable.
  //   _gridSortKeys     — per-frame screen-y midpoint (sort key).
  //   _gridIndicesScratch — reorder destination uploaded each frame.
  _gridIndexPairs     = new Uint32Array(segCount * 2);
  _gridSegMidWorld    = new Float32Array(segCount * 2);
  _gridSortKeys       = new Float32Array(segCount);
  _gridIndicesScratch = new Uint32Array(idxCount);
  _gridSortPerm       = new Array(segCount);
  for (let s = 0; s < segCount; s++) _gridSortPerm[s] = s;
  let ii = 0;
  let segIdx = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = r * cols + c;
      const vx = verts[v * 2];
      const vy = verts[v * 2 + 1];
      if (c < cols - 1) {
        const vr = v + 1;
        indices[ii++] = v;  indices[ii++] = vr;
        _gridIndexPairs[segIdx * 2]     = v;
        _gridIndexPairs[segIdx * 2 + 1] = vr;
        _gridSegMidWorld[segIdx * 2]     = (vx + verts[vr * 2]) * 0.5;
        _gridSegMidWorld[segIdx * 2 + 1] = (vy + verts[vr * 2 + 1]) * 0.5;
        segIdx++;
      }
      if (r < rows - 1) {
        const vd = v + cols;
        indices[ii++] = v;  indices[ii++] = vd;
        _gridIndexPairs[segIdx * 2]     = v;
        _gridIndexPairs[segIdx * 2 + 1] = vd;
        _gridSegMidWorld[segIdx * 2]     = (vx + verts[vd * 2]) * 0.5;
        _gridSegMidWorld[segIdx * 2 + 1] = (vy + verts[vd * 2 + 1]) * 0.5;
        segIdx++;
      }
    }
  }
  if (!_bufGridVerts) _bufGridVerts = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufGridVerts);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  if (!_bufGridIndices) _bufGridIndices = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, _bufGridIndices);
  // V11.2: DYNAMIC_DRAW because the painter's-algorithm sort re-uploads
  // a reordered version of this index buffer per frame when fieldStyle
  // is 'rubber-sheet'. For other field styles the buffer stays at its
  // original contents (no per-frame upload) — still safe with DYNAMIC.
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
}

// V9.2 particle-flow: CPU-side advection + additive-blended point sprites.
// Particles drift along the gravitational force field, recycled when they
// either fall into a body (their nearest entity is closer than that
// entity's radius) or exceed their lifetime. Recycle spawns them at a
// random viewport edge with a small inward velocity, so they look like
// dust streaming in from the periphery.
//
// CPU cost at PARTICLE_COUNT=240 entities=N: 240 × N gravity evals per
// frame = ~3k ops at N=10. Trivial. GPU cost: one draw call with N point
// sprites — also negligible.
const PARTICLE_LIFE_FRAMES = 240;
const PARTICLE_COLOR = [0.78, 0.88, 1.0, 0.55];
const PARTICLE_POINT_SIZE = 5.0;
let _particleSeedRng = (() => {
  let s = 0x12345678 >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
})();

function _seedParticle(i) {
  const rng = _particleSeedRng;
  // Spawn at a random edge of the viewport with mild inward velocity so
  // they look like ambient dust streaming in.
  const edge = (rng() * 4) | 0;
  let x, y, vx, vy;
  if (edge === 0)      { x = 0;        y = rng() * _vpH; vx = +0.4;        vy = (rng() - 0.5) * 0.4; }
  else if (edge === 1) { x = _vpW;     y = rng() * _vpH; vx = -0.4;        vy = (rng() - 0.5) * 0.4; }
  else if (edge === 2) { x = rng() * _vpW; y = 0;        vx = (rng() - 0.5) * 0.4; vy = +0.4; }
  else                 { x = rng() * _vpW; y = _vpH;     vx = (rng() - 0.5) * 0.4; vy = -0.4; }
  _particleData[i * 3]     = x;
  _particleData[i * 3 + 1] = y;
  _particleData[i * 3 + 2] = 0;
  _particleVel[i * 2]      = vx;
  _particleVel[i * 2 + 1]  = vy;
  _particleLife[i] = (rng() * PARTICLE_LIFE_FRAMES) | 0;
}

function _ensureParticlesSeeded() {
  if (_particlesSeeded || _vpW <= 0 || _vpH <= 0) return;
  for (let i = 0; i < PARTICLE_COUNT; i++) _seedParticle(i);
  _particlesSeeded = true;
}

// Compute force on a single point from entity array (uses _fieldEntityData
// already packed by drawField, which includes wrap ghosts when active).
function _forceAtPoint(x, y) {
  let gx = 0, gy = 0;
  const eps2 = state.epsilon * state.epsilon;
  for (let i = 0; i < _fieldEntityCount; i++) {
    const o = i * 4;
    const ex = _fieldEntityData[o];
    const ey = _fieldEntityData[o + 1];
    const Gqm = _fieldEntityData[o + 2];
    const dx = x - ex;
    const dy = y - ey;
    const r2 = dx * dx + dy * dy + eps2;
    const invR3 = 1 / (r2 * Math.sqrt(r2));
    gx += -Gqm * dx * invR3;
    gy += -Gqm * dy * invR3;
  }
  return [gx, gy];
}

function _updateParticles() {
  _ensureParticlesSeeded();
  const W = _vpW, H = _vpH;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    let age = _particleData[i * 3 + 2];
    age += 1 / PARTICLE_LIFE_FRAMES;
    if (age >= 1 || _particleLife[i] <= 0) {
      _seedParticle(i);
      continue;
    }
    _particleLife[i]--;
    const x = _particleData[i * 3];
    const y = _particleData[i * 3 + 1];
    let vx = _particleVel[i * 2];
    let vy = _particleVel[i * 2 + 1];
    const [fx, fy] = _forceAtPoint(x, y);
    // Scale force so particles drift visibly but not insanely fast.
    const fScale = 6.0;
    vx += fx * fScale;
    vy += fy * fScale;
    // Speed cap so close approaches don't blow up.
    const speed = Math.hypot(vx, vy);
    if (speed > 6) { const k = 6 / speed; vx *= k; vy *= k; }
    const nx = x + vx;
    const ny = y + vy;
    // Recycle if outside viewport or inside a body.
    let recycle = false;
    if (nx < -8 || nx > W + 8 || ny < -8 || ny > H + 8) recycle = true;
    else {
      for (let j = 0, n = state.entities.length; j < n; j++) {
        const e = state.entities[j];
        if (e.absorbing) continue;
        const dr = Math.hypot(nx - e.x, ny - e.y);
        if (dr < e.radius + 2) { recycle = true; break; }
      }
    }
    if (recycle) {
      _seedParticle(i);
    } else {
      _particleData[i * 3]     = nx;
      _particleData[i * 3 + 1] = ny;
      _particleData[i * 3 + 2] = age;
      _particleVel[i * 2]      = vx;
      _particleVel[i * 2 + 1]  = vy;
    }
  }
}

function _drawParticleFlow() {
  if (!_progParticleFlow || _fieldEntityCount === 0) return;
  const gl = _gl;
  _updateParticles();
  if (!_bufParticles) _bufParticles = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufParticles);
  gl.bufferData(gl.ARRAY_BUFFER, _particleData, gl.DYNAMIC_DRAW);

  gl.useProgram(_progParticleFlow.prog);
  gl.uniformMatrix4fv(_progParticleFlow.uOrtho, false, _orthoMat);
  gl.uniform4f(_progParticleFlow.uColor,
    PARTICLE_COLOR[0], PARTICLE_COLOR[1], PARTICLE_COLOR[2], PARTICLE_COLOR[3]);
  gl.uniform1f(_progParticleFlow.uPointSize, PARTICLE_POINT_SIZE * _dpr);
  // V10.3: particle flow is companion-only to the '2d' field style, so
  // _sagActive is always false here and _bindSagUniforms sets uSagMode=0
  // naturally.
  _bindSagUniforms(_progParticleFlow);

  // Additive blending for the luminous dust look.
  const prevBlendFunc = [gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA];
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

  // Interleaved (x, y, age) stride = 12 bytes.
  gl.enableVertexAttribArray(_progParticleFlow.aPos);
  gl.vertexAttribPointer(_progParticleFlow.aPos, 2, gl.FLOAT, false, 12, 0);
  gl.enableVertexAttribArray(_progParticleFlow.aAge);
  gl.vertexAttribPointer(_progParticleFlow.aAge, 1, gl.FLOAT, false, 12, 8);
  gl.drawArrays(gl.POINTS, 0, PARTICLE_COUNT);
  gl.disableVertexAttribArray(_progParticleFlow.aPos);
  gl.disableVertexAttribArray(_progParticleFlow.aAge);

  // Restore blend func for any subsequent passes.
  gl.blendFunc(prevBlendFunc[0], prevBlendFunc[1]);
}

// V9.2: grid-warp draw. Selects between 3D oblique and 2D in-plane via
// state.fieldStyle ('3d' default | '2d'). Uses the entity array packed
// into _fieldEntityData by drawField().
const GRID_WARP_COLOR_3D = [120 / 255, 170 / 255, 230 / 255, 0.55];
const GRID_WARP_COLOR_2D = [150 / 255, 200 / 255, 250 / 255, 0.55];

// V9.6 relative-shading: CPU mirror of GRID_WARP.VS intensity math,
// sampled across the viewport to derive the per-frame MIN and MAX
// of vIntensity. Both are passed as uniforms (uIntensityMin /
// uIntensityMax); the fragment shader smoothsteps between them so
// the flattest visible vertex always renders fully bright and the
// deepest well fully dim — regardless of absolute warp strength or
// nonzero baseline warp in crowded scenes.
//
// CONTRACT (analogous to the Plummer-softening oracle in CLAUDE.md):
// if you change the per-vertex intensity formula in GRID_WARP.VS,
// you MUST mirror it here or shading will mismatch.
const _GRIDWARP_OVERSHOOT_FRAC = 0.4;   // mirrors VS
const _GRIDWARP_L_ATAN = 50.0;          // mirrors VS
const _GRIDWARP_DEPTH_CAP = 250.0;      // mirrors VS clamp in 3D mode
// EMA state for the smoothstep bounds (uIntensityMin/uIntensityMax).
// Slow rise/fall prevents brightness flash when a body is deleted or
// moves off-screen and the raw range collapses in one frame.
// Sentinel -1 = uninitialized → first frame seeds with raw values
// (no EMA warm-up dimness for the first ~30 frames).
let _intensityMinEMA = -1;
let _intensityMaxEMA = -1;
let _intensityModePrev = -1;   // detect 2D↔3D switch → reset EMA seed
// Scratch returned by _computeFieldIntensityRange — reused each
// frame to avoid GC churn from allocating a fresh {min, max}.
const _intensityRange = { min: 0, max: 0 };
// V11.4 (2026-05-28): cache the peak screen-Y offset added by the sag
// projection on this frame, so _drawEntities can widen the wrap-mode
// ghost-spawn threshold accordingly. With rubber-sheet ON, a body
// sprite renders at screen_y = world_y + sag·yFactor, so a wrap-mirror
// ghost is visible on canvas for a band of width (sprite._ox + _maxSagY)
// from each Y-edge — wider than the original sprite-only threshold.
// Computed in prepareFrameRenderer after _updateSagTexture; 0 when
// _sagActive is false (any non-rubber-sheet field style or field off).
let _maxSagY = 0;
// V12 (2026-05-28): same scan, kept in RAW (pre-yFactor) form for the
// RUBBER_SHEET_FS depth-fade normalization. The FS multiplies by yFactor
// internally for projection; we need the raw sag amplitude to scale the
// brightness ramp. Set to 0 when sag is off (same lifecycle as _maxSagY).
let _maxSagRaw = 0;
function _computeFieldIntensityRange(mode, dispScale, epsilon) {
  if (_fieldEntityCount === 0) {
    _intensityRange.min = 0;
    _intensityRange.max = 0;
    return _intensityRange;
  }
  // V11 rubber-sheet (mode 2): sag-texture pixels encode absolute
  // CSS-px sag = |φ|/G_vert (no clamp, no normalization). The mesh VS
  // outputs vIntensity = sag · cos(viewTilt), so the range is the
  // shared cos(viewTilt) factor × { min, max } of _sagPixels (see
  // SAG_VS_HELPER in shaders.js and _sagYFactor() above). Scanning
  // the 256×256 buffer is ~65k px / frame, hot in L1, sub-ms. Gives a
  // scene-relative brightness ramp that moves with the deepest live
  // well. NB: the `_fieldEntityCount === 0` early-return above is
  // load-bearing for this branch — without it, the scan would run on
  // a zero-filled buffer and produce a max=0 range, breaking the
  // smoothstep degenerate guard in _drawGridWarp.
  if (mode === 2) {
    // V11.1: brightness ramp tracks the visible projected depth, via
    // the SAME guarded helper the uniform-bind uses — so any view-tilt
    // change is reflected here in lockstep. At viewTilt=90° this
    // collapses to 0 and the smoothstep degenerate guard in
    // _drawGridWarp takes over.
    const yFactor = _sagYFactor();
    let mn = Infinity, mx = 0;
    const N = _sagPixels.length;
    for (let i = 0; i < N; i++) {
      const v = _sagPixels[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    _intensityRange.min = (mn === Infinity ? 0 : mn) * yFactor;
    _intensityRange.max = mx * yFactor;
    return _intensityRange;
  }
  const ents = _fieldEntityData;
  const cnt = _fieldEntityCount;
  const eps2 = epsilon * epsilon;
  // 16x12 = 192 samples — fixed cost regardless of entity count.
  // Per-sample inner loop is N entities × ~12 ops, so total
  // ~2300·N ops/frame. At N=200 that's ~460K ops ≈ 2 ms; cheaper
  // than the GL draw itself.
  const SC = 16, SR = 12;
  // Same anti-fold cap as the shader. Without mirroring it here the
  // CPU min/max range overstates max → smoothstep underestimates t →
  // brightness looks too bright at the deepest wells (the GL has
  // already capped to maxDisp, but the renderer thinks max is higher).
  const maxDisp = _gridCellPx * 0.45;
  let minI = Infinity, maxI = 0;
  for (let r = 0; r < SR; r++) {
    const y = (r / (SR - 1)) * _vpH;
    for (let c = 0; c < SC; c++) {
      const x = (c / (SC - 1)) * _vpW;
      let intensity = 0;
      if (mode === 0) {
        let phi = 0;
        for (let i = 0; i < cnt; i++) {
          const o = i * 4;
          const dx = x - ents[o];
          const dy = y - ents[o + 1];
          const r2 = dx * dx + dy * dy + eps2;
          phi += -ents[o + 2] / Math.sqrt(r2);
        }
        intensity = Math.min(_GRIDWARP_DEPTH_CAP, Math.max(0, -phi * dispScale));
        if (intensity > maxDisp) intensity = maxDisp;
      } else {
        // 2D — mirrors GRID_WARP.VS 2D branch verbatim. Variable
        // names track the VS: r = softened distance to body, raw =
        // unbounded magnitude, bounded = OVERSHOOT_FRAC-clamped
        // signed magnitude, disp2D = accumulated displacement vector.
        let disp2Dx = 0, disp2Dy = 0;
        for (let i = 0; i < cnt; i++) {
          const o = i * 4;
          const dx = x - ents[o];
          const dy = y - ents[o + 1];
          const r2 = dx * dx + dy * dy + eps2;
          const invR = 1 / Math.sqrt(r2);
          const r = 1 / invR;
          const raw = ents[o + 2] * dispScale * invR * invR;
          const maxAllowed = r * _GRIDWARP_OVERSHOOT_FRAC;
          const aRaw = raw < 0 ? -raw : raw;
          const clampedMag = aRaw < maxAllowed ? aRaw : maxAllowed;
          const bounded = raw < 0 ? -clampedMag : clampedMag;
          disp2Dx += -dx * invR * bounded;
          disp2Dy += -dy * invR * bounded;
        }
        const mag = Math.sqrt(disp2Dx * disp2Dx + disp2Dy * disp2Dy);
        intensity = _GRIDWARP_L_ATAN * Math.atan(mag / _GRIDWARP_L_ATAN);
        if (intensity > maxDisp) intensity = maxDisp;
      }
      if (intensity < minI) minI = intensity;
      if (intensity > maxI) maxI = intensity;
    }
  }
  _intensityRange.min = minI === Infinity ? 0 : minI;
  _intensityRange.max = maxI;
  return _intensityRange;
}

// V11.2 painter's algorithm for the rubber-sheet mesh.
// Goal: lines further into the scene (back) draw FIRST so foreground
// lines overpaint them where they cross in screen space. Without
// this, the static index order produces a uniform blend that loses
// the front/back occlusion cue the user expects in oblique 3D.
//
// Approach: each segment has a static world-space midpoint (cached
// in _gridSegMidWorld). Per frame, we bilinear-sample the CPU-side
// _sagPixels at that midpoint, compute screen-y = world-y +
// sag·cos(viewTilt), sort segment indices ascending (back = smaller
// screen-y first), reorder _gridIndexPairs into _gridIndicesScratch,
// and re-upload.
//
// Bilinear sampling on CPU mirrors what the GPU does via the R32F
// texture's LINEAR filter, so the painter's sort uses the same
// projected position the GPU will render at.
//
// Cost: O(S log S) sort with S ≈ 10k segments → ~1-2 ms. Only runs
// in rubber-sheet mode; other modes use the static buffer.
function _sortGridIndicesPainter() {
  if (!_gl) return false;
  if (_gridSegCount === 0) return false;
  const seg = _gridSegCount;
  const mid = _gridSegMidWorld;
  const keys = _gridSortKeys;
  const pixels = _sagPixels;
  const W = _vpW;
  const H = _vpH;
  if (W <= 0 || H <= 0) return false;
  const yFactor = _sagYFactor();
  const TW = _SAG_TEX_W;
  const TH = _SAG_TEX_H;
  // Bilinear-sample the sag texture at each segment midpoint and
  // compute screen-y. Off-viewport midpoints (negative or > viewport)
  // sample via REPEAT (mod-then-clamp arithmetic mirrors gl.REPEAT).
  for (let s = 0; s < seg; s++) {
    const mx = mid[s * 2];
    const my = mid[s * 2 + 1];
    // World→tex coords. REPEAT: u = fract(mx/W) × TW.
    // u, v are fract(…) ∈ [0,1) by construction, so fx ∈ [0,TW) and
    // ix0 = fx|0 lands in [0, TW-1] — no extra clamp needed.
    let u = (mx / W) - Math.floor(mx / W);   // [0, 1)
    let v = (my / H) - Math.floor(my / H);
    const fx = u * TW;
    const fy = v * TH;
    const ix0 = fx | 0;
    const iy0 = fy | 0;
    const ix1 = (ix0 + 1) % TW;     // REPEAT wrap to column 0 at right edge
    const iy1 = (iy0 + 1) % TH;
    const tx = fx - ix0;
    const ty = fy - iy0;
    const p00 = pixels[iy0 * TW + ix0];
    const p10 = pixels[iy0 * TW + ix1];
    const p01 = pixels[iy1 * TW + ix0];
    const p11 = pixels[iy1 * TW + ix1];
    const sag = (p00 * (1 - tx) + p10 * tx) * (1 - ty)
              + (p01 * (1 - tx) + p11 * tx) * ty;
    keys[s] = my + sag * yFactor;
  }
  // Sort segment indices ascending by screen-y midpoint (back first).
  // _gridSortPerm holds the segment indices; we sort the array in
  // place by comparator reading from keys.
  const perm = _gridSortPerm;
  perm.sort((a, b) => keys[a] - keys[b]);
  // Materialize the reordered index pairs into the scratch buffer.
  const src = _gridIndexPairs;
  const dst = _gridIndicesScratch;
  for (let s = 0; s < seg; s++) {
    const srcSeg = perm[s];
    dst[s * 2]     = src[srcSeg * 2];
    dst[s * 2 + 1] = src[srcSeg * 2 + 1];
  }
  return true;
}

// Membrane field — full-screen relief-lit grayscale gravity sheet. See
// SCREEN_DENT in shaders.js. Height field + 45° point-light hillshade +
// single-valued grid pinch, composited at the 膜透明度 alpha.
const _MEMBRANE_CORE_FRAC = 0.08;    // well core radius = CORE_FRAC · min(vp); larger = wider
const _MEMBRANE_SLOPE_K   = 2.2;     // ∇h→normal scale (× core); larger = stronger relief
const _MEMBRANE_WARP_K    = 0.8;     // grid-pinch gain (×, kept below fold)
const _MEMBRANE_COLOR     = [0.82, 0.86, 0.94];   // cool grayscale membrane tint
function _drawScreenDent() {
  const gl = _gl;
  if (!_progScreenDent) return;

  // Heaviest body's |G·q·m| → per-frame normalizers so relief + pinch
  // adapt to G / mass without a manual gain. Core scales with the canvas
  // (the "按 canvas 距离尺度缩放" request) so wells stay proportionate.
  let maxWeight = 0;
  for (let i = 0; i < _fieldEntityCount; i++) {
    const w = Math.abs(_fieldEntityData[i * 4 + 2]);
    if (w > maxWeight) maxWeight = w;
  }
  const core = Math.max(8, Math.min(_vpW, _vpH) * _MEMBRANE_CORE_FRAC);
  const core2 = core * core;
  // heightK folds amplitude + per-frame normalization: h_i = |w_i|·heightK
  //   / (r²+core²) with heightK = core²/maxWeight peaks at 1 per body.
  const heightK = maxWeight > 0 ? core2 / maxWeight : 0;
  const warpGain = maxWeight > 0 ? (_MEMBRANE_WARP_K * core2) / maxWeight : 0;
  // contrast slider → ambient floor (shadow depth). High contrast = low
  // ambient = deep shadows; contrast 0 = flat lighting.
  const ambient = 0.55 - 0.45 * Math.min(1, Math.max(0, state.fieldContrast));

  gl.useProgram(_progScreenDent.prog);
  gl.uniform2f(_progScreenDent.uViewport, _vpW, _vpH);
  gl.uniform4fv(_progScreenDent.uEntities, _fieldEntityData, 0, _fieldEntityCount * 4);
  gl.uniform1i(_progScreenDent.uEntityCount, _fieldEntityCount);
  gl.uniform1f(_progScreenDent.uHeightK, heightK);
  gl.uniform1f(_progScreenDent.uCore2, core2);
  gl.uniform1f(_progScreenDent.uWarpGain, warpGain);
  gl.uniform1f(_progScreenDent.uSlope, core * _MEMBRANE_SLOPE_K);
  gl.uniform1f(_progScreenDent.uAmbient, ambient);
  gl.uniform1f(_progScreenDent.uCellPx, Math.max(8, state.fieldLineSpacing));
  gl.uniform4f(_progScreenDent.uColor,
    _MEMBRANE_COLOR[0], _MEMBRANE_COLOR[1], _MEMBRANE_COLOR[2], 1.0);
  gl.uniform1f(_progScreenDent.uOpacity, Math.min(1, Math.max(0, state.membraneOpacity)));

  // Shared fullscreen-quad VAO (VS_FULLSCREEN), same as _drawRubberSheetFS.
  gl.bindVertexArray(_vaoFsQuad);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}

function _drawGridWarp() {
  const gl = _gl;
  // V12 (2026-05-28): rubber-sheet now uses a full-screen FS pass
  // (_drawRubberSheetFS) instead of the indexed-LINE mesh. Dispatch
  // here BEFORE the mesh-validity guard below so an uninitialized
  // mesh doesn't block the FS path. Modes '2d'/'3d' still use the
  // mesh path unchanged.
  if (state.fieldStyle === 'rubber-sheet') {
    _drawRubberSheetFS();
    return;
  }
  if (!_progGridWarp || !_bufGridVerts || !_bufGridIndices) return;
  if (_gridVertexCount === 0 || _gridIndexCount === 0) return;
  // mode dispatch:
  //   0 = legacy 3D oblique (per-vertex φ, saturates at depth clamp)
  //   1 = 2D in-plane warp
  //   (mode == 2 / rubber-sheet now handled by _drawRubberSheetFS above)
  const mode =
      (state.fieldStyle === '2d') ? 1
    : 0;
  // Displacement scale tuned so the 2D in-plane warp matches the
  // mockup_2d.py reference visually: cap=28 px saturates within
  // ~300px of a mass=200 body at G=80, leaving clean far-field.
  // 3D mode stays mild (5.0) to avoid the iconic-but-overwhelming
  // global-bowl effect — but acknowledged as a known v1 limitation
  // since user picked 2D+particles direction. Mode 2 explicitly sets
  // 0.0 — the VS early-returns before reading uDispScale, but sending
  // 0 documents the intent and prevents a future VS edit that drops
  // the early-return from silently inheriting the 2D 175.0 value
  // (which would massively over-scale the legacy 3D oblique path).
  const dispScale = (mode === 0) ? 5.0 : (mode === 2) ? 0.0 : 175.0;
  // V10 rubber-sheet: tiltY is encoded inside the sag texture (we baked
  // sin(45°) into sagProject already), so mode 2 doesn't read uTiltY at
  // all. Legacy mode 0 stays at 1.0; the rubber-sheet legacy code-path
  // that re-used mode=0 with a 0.7071 override is now unreachable.
  const TILT_Y_DEFAULT = 1.0;
  const tiltY = TILT_Y_DEFAULT;
  // V9.6: dynamic per-frame min+max → true scene-relative shading.
  // Anchors smoothstep to [scene min, scene max] so the flattest
  // visible vertex always renders fully bright relative to the
  // deepest well. Critical in crowded scenes where every vertex has
  // nonzero baseline warp — without min anchoring the dynamic range
  // collapses and the whole scene looks uniformly dim.
  // EMA-smoothed (α=0.15, ~7-frame half-life) prevents flicker when
  // a body is deleted / moves off-screen and raw range jumps.
  // Sample window is viewport only; off-viewport vertices in the
  // 15% grid extension inherit the same scale (acceptable since
  // they're clipped).
  const range = _computeFieldIntensityRange(mode, dispScale, state.epsilon);
  // Reset EMA when fieldStyle changes — 2D and 3D produce intensities
  // in different physical units (warp magnitude vs sag depth), so the
  // previous EMA would mismap brightness for ~7 frames post-switch.
  if (mode !== _intensityModePrev) {
    _intensityMinEMA = -1;
    _intensityMaxEMA = -1;
    _intensityModePrev = mode;
  }
  if (_intensityMinEMA < 0) {
    _intensityMinEMA = range.min;   // first-frame seed avoids cold-start dimness
    _intensityMaxEMA = range.max;
  } else {
    _intensityMinEMA = _intensityMinEMA * 0.85 + range.min * 0.15;
    _intensityMaxEMA = _intensityMaxEMA * 0.85 + range.max * 0.15;
  }
  // Guarantee max - min >= 0.5 to keep smoothstep non-degenerate
  // (empty / near-uniform scenes → no division blow-up).
  let intensityMin = _intensityMinEMA;
  let intensityMax = Math.max(_intensityMaxEMA, intensityMin + 0.5);
  // Mode 2 uses the same 3D ramp color as legacy 3D — the visual is a
  // wireframe rubber sheet either way.
  const color = (mode === 1) ? GRID_WARP_COLOR_2D : GRID_WARP_COLOR_3D;

  gl.useProgram(_progGridWarp.prog);
  gl.uniform2f(_progGridWarp.uViewport, _vpW, _vpH);
  gl.uniform4fv(_progGridWarp.uEntities, _fieldEntityData, 0, _fieldEntityCount * 4);
  gl.uniform1i(_progGridWarp.uEntityCount, _fieldEntityCount);
  gl.uniform1f(_progGridWarp.uEpsilon, state.epsilon);
  gl.uniform1f(_progGridWarp.uDispScale, dispScale);
  gl.uniform1f(_progGridWarp.uTiltY, tiltY);
  gl.uniform1i(_progGridWarp.uMode, mode);
  gl.uniformMatrix4fv(_progGridWarp.uOrtho, false, _orthoMat);
  gl.uniform4f(_progGridWarp.uColor, color[0], color[1], color[2], color[3]);
  // V10 rubber-sheet: bind the per-frame sag texture so the VS branch
  // for uMode == 2 can sample sag(world_x, world_y). Calling unconditionally
  // is cheap (one texture rebind + 2 uniform sets) and means the mesh
  // never sees stale sag uniforms regardless of which mode runs first.
  _bindSagUniforms(_progGridWarp);
  gl.uniform1f(_progGridWarp.uIntensityMin, intensityMin);
  gl.uniform1f(_progGridWarp.uIntensityMax, intensityMax);
  // contrast slider: state.fieldContrast 0..1, floor = 1 - contrast
  // (0 contrast → floor = 1 → no dimming; 1 contrast → floor = 0 → max).
  gl.uniform1f(_progGridWarp.uContrastFloor, 1 - state.fieldContrast);
  // anti-fold cap: max displacement < 0.5·cellPx → adjacent vertices
  // can never swap order → grid never self-intersects.
  gl.uniform1f(_progGridWarp.uCellPx, _gridCellPx);

  gl.bindBuffer(gl.ARRAY_BUFFER, _bufGridVerts);
  gl.enableVertexAttribArray(_progGridWarp.aPos);
  gl.vertexAttribPointer(_progGridWarp.aPos, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, _bufGridIndices);
  // V11.2: painter's algorithm — only rubber-sheet mode (mode==2) has
  // a real notion of front/back, so it's the only mode that re-uploads
  // a per-frame sorted index buffer. Modes 0/1 use the build-time
  // order (the buffer still holds those indices since the sort path
  // never writes for non-rubber-sheet modes).
  if (mode === 2 && _sagActive && _sortGridIndicesPainter()) {
    gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, _gridIndicesScratch);
  }
  gl.drawElements(gl.LINES, _gridIndexCount, gl.UNSIGNED_INT, 0);
  gl.disableVertexAttribArray(_progGridWarp.aPos);
}

// V12 (2026-05-28): rubber-sheet renderer via full-screen FS pass.
// User pivoted away from V11's indexed-LINE mesh after determining the
// 15% viewport-expansion "skirt" couldn't keep up with heavy warps —
// dense scenes stretched the mesh enough that the finite extent re-
// emerged at the viewport edge.
//
// Approach: a single full-screen quad runs the RUBBER_SHEET_FS program.
// Each fragment treats its screen-px position as world-x, samples the
// sag texture there, and subtracts sag·yFactor from screen-y to obtain
// its world-y (single-step forward-warp). Grid coverage is then
// computed in world space via fwidth-AA against fract(uvW - 0.5). No
// mesh, no skirt — there is no finite extent that could leak. In wrap
// mode the sag texture is toroidal (9-ghost pack); fract(uv) sampling
// continues across the seam without discontinuity.
//
// Cost: 1 sag texture fetch + ~10 ALU ops per fragment. Discard on
// non-line fragments drops most of the work. ~0.5 ms on iGPU at
// 1440×900. Cheaper than V11 mesh pipeline (~1.5 ms with painter sort).
//
// Dispatched from _drawGridWarp's top branch when state.fieldStyle ===
// 'rubber-sheet'. Modes '2d' and '3d' still use _drawGridWarp's mesh.
const RUBBER_SHEET_COLOR = GRID_WARP_COLOR_3D;
function _drawRubberSheetFS() {
  if (!_gl || !_progRubberSheetFS) return;
  if (_vpW <= 0 || _vpH <= 0) return;
  const gl = _gl;

  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  gl.useProgram(_progRubberSheetFS.prog);
  gl.uniform2f(_progRubberSheetFS.uViewport, _vpW, _vpH);
  // Bind the per-frame sag texture (R32F, REPEAT-wrap). The FS samples
  // it via sagUV() which selects fract (wrap) or clamp (bounded) per
  // state.boundaryMode — same convention SAG_VS_HELPER uses for bodies/
  // trails so visual consistency is automatic.
  gl.activeTexture(gl.TEXTURE0 + _SAG_TEX_UNIT);
  gl.bindTexture(gl.TEXTURE_2D, _sagTexture);
  gl.uniform1i(_progRubberSheetFS.uSagTex, _SAG_TEX_UNIT);
  gl.uniform1f(_progRubberSheetFS.uSagYFactor, _sagYFactor());
  gl.uniform1f(_progRubberSheetFS.uSagWrap, state.boundaryMode === 'wrap' ? 1.0 : 0.0);
  // Cell spacing comes from the same source as the V11 mesh (user-
  // controlled state.fieldLineSpacing slider, range 12-80 px). Using
  // it directly here keeps slider behavior visually consistent across
  // mesh ↔ FS path.
  gl.uniform1f(_progRubberSheetFS.uCellPx, Math.max(8, state.fieldLineSpacing));
  gl.uniform4f(_progRubberSheetFS.uColor,
    RUBBER_SHEET_COLOR[0], RUBBER_SHEET_COLOR[1],
    RUBBER_SHEET_COLOR[2], RUBBER_SHEET_COLOR[3]);
  // Contrast slider: state.fieldContrast 0..1, floor = 1 - contrast.
  // Same semantic as the V11 mesh path — preserved across the swap.
  gl.uniform1f(_progRubberSheetFS.uContrastFloor, 1 - state.fieldContrast);
  // Per-frame peak sag (raw, pre-yFactor) — drives the depth-fade
  // dynamic range. _maxSagRaw is set in prepareFrameRenderer next to
  // the V11.4 _maxSagY (same scan, two outputs). 0 when sag is off,
  // which makes the FS's depthFactor degenerate to 0 → brightness 1.
  gl.uniform1f(_progRubberSheetFS.uMaxSag, _maxSagRaw);
  gl.activeTexture(gl.TEXTURE0);   // restore default

  // Use the shared fullscreen-quad VAO (location-0 attrib pointer at
  // _bufFsQuad). RUBBER_SHEET_FS.VS === VS_FULLSCREEN, so the same
  // VAO that decay/blit use works here unchanged.
  gl.bindVertexArray(_vaoFsQuad);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);

  gl.disable(gl.BLEND);
}

// V9.9 curvilinear mode: render gravitational geodesics as polylines.
// computeFieldLines (CPU integrator) returns polylines that already
// handle wrap-mode splits + body absorption. Each polyline becomes
// N-1 line segments fed into the existing LINE_SEG instanced pipeline.
//
// Cost: ~N_bodies × raysPerBody × ~80 avg steps × N_entities per ∇φ
// eval. At 10 bodies × 16 rays × 80 × 10 = ~130 K ops / frame ≈ 1 ms
// CPU. GL upload + draw is O(total segments) ≈ ~5 K instances —
// negligible vs the GRID_WARP fragment work it replaces.
function _drawFieldLines() {
  if (!_progLineSeg || !_vaoLineSeg || !_bufInstanceLineSeg) return;
  if (state.entities.length === 0) return;
  const wrap = state.boundaryMode === 'wrap';
  const viewport = { width: _vpW, height: _vpH };
  const polylines = computeFieldLines(
    state.entities, viewport, state.fieldLineSpacing,
    state.G, state.epsilon, wrap,
  );
  if (polylines.length === 0) return;

  // Convert polylines into LINE_SEG instances. Reuse _lineSegData
  // and reset _lineSegCount when done so drawUI starts from a
  // clean buffer (drawUI runs after drawField).
  _lineSegCount = 0;
  // Same cyan family as the BH event-horizon ring, slightly cooler
  // and with low alpha so multiple overlapping lines stack pleasantly.
  const FL_R = 150 / 255, FL_G = 195 / 255, FL_B = 240 / 255, FL_A = 0.45;
  for (let p = 0; p < polylines.length; p++) {
    const line = polylines[p];
    const len = line.length;
    if (len < 4) continue;   // need at least 2 vertices for one segment
    let arc = 0;
    for (let i = 0; i < len - 2; i += 2) {
      const x0 = line[i],     y0 = line[i + 1];
      const x1 = line[i + 2], y1 = line[i + 3];
      _pushLineSeg(x0, y0, x1, y1, FL_R, FL_G, FL_B, FL_A, arc, 1.5, 0.0, 0.0);
      const dx = x1 - x0, dy = y1 - y0;
      arc += Math.sqrt(dx * dx + dy * dy);
    }
  }
  if (_lineSegCount === 0) return;

  const gl = _gl;
  gl.useProgram(_progLineSeg.prog);
  gl.uniformMatrix4fv(_progLineSeg.uOrtho, false, _orthoMat);
  // V10.3: field lines are curvilinear-mode only; rubber-sheet uses
  // GRID_WARP mesh instead — so _sagActive is false here and the
  // shader's sagProject is a passthrough via _bindSagUniforms.
  _bindSagUniforms(_progLineSeg);
  gl.bindVertexArray(_vaoLineSeg);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceLineSeg);
  gl.bufferData(gl.ARRAY_BUFFER, _lineSegData.subarray(0, _lineSegCount * 12), gl.DYNAMIC_DRAW);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, _lineSegCount);
  // Reset so drawUI's subsequent rebuild starts from zero.
  _lineSegCount = 0;
}

// V9.1: build a (cols × rows) grid of streamline seed positions covering
// the viewport with margins. Target seed count is ~96 (per user choice);
// we resolve to the integer (cols, rows) closest to that target that
// matches the viewport aspect ratio. Recomputed only on viewport change.
function _rebuildStreamlineSeeds() {
  // V9.2: density quartered (96 → 24). The grid-warp now carries the
  // primary "where does the field push" signal; streamlines are kept
  // as sparse decorative pulses to maintain motion.
  const targetCount = 24;
  const aspect = _vpW / Math.max(1, _vpH);
  // cols * rows ≈ targetCount; cols / rows ≈ aspect → cols = sqrt(target*aspect).
  let cols = Math.max(2, Math.round(Math.sqrt(targetCount * aspect)));
  let rows = Math.max(2, Math.round(targetCount / cols));
  const total = cols * rows;
  _streamlineSeedCount = total;
  if (_streamlineSeeds.length < total * 2) {
    _streamlineSeeds = new Float32Array(total * 2);
  }
  // Inset by half a cell so seeds sit at cell centers, not on the edge.
  const cellW = _vpW / cols;
  const cellH = _vpH / rows;
  let idx = 0;
  for (let r = 0; r < rows; r++) {
    const y = (r + 0.5) * cellH;
    for (let c = 0; c < cols; c++) {
      const x = (c + 0.5) * cellW;
      _streamlineSeeds[idx++] = x;
      _streamlineSeeds[idx++] = y;
    }
  }
}

function _updateOrthoMat() {
  // Maps CSS-px (x ∈ [0..W], y ∈ [0..H]) → NDC (-1..+1).
  // Y is flipped so y=0 is top (matches Canvas2D and entity coordinate convention).
  // Column-major mat4.
  const W = _vpW || 1, H = _vpH || 1;
  const m = _orthoMat;
  m[0] = 2 / W;  m[1] = 0;       m[2] = 0; m[3] = 0;
  m[4] = 0;      m[5] = -2 / H;  m[6] = 0; m[7] = 0;
  m[8] = 0;      m[9] = 0;       m[10] = 1; m[11] = 0;
  m[12] = -1;    m[13] = 1;      m[14] = 0; m[15] = 1;
}

// ─── Shader compilation helpers ───────────────────────────────────

function _makeProgram(vsSrc, fsSrc) {
  const gl = _gl;
  const vs = _compileShader(gl.VERTEX_SHADER, vsSrc);
  const fs = _compileShader(gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  // Force the primary per-vertex position attribute to location 0 in every
  // program, regardless of declaration order in the shader. This lets us
  // share `_vaoFsQuad` (whose location-0 attrib pointer addresses _bufFsQuad)
  // across the trail-decay and trail-blit programs without worrying that the
  // linker assigned `aPos` to a different location per program. Programs
  // that don't declare `aPos` / `aCorner` simply ignore the call (per spec).
  gl.bindAttribLocation(prog, 0, 'aPos');
  gl.bindAttribLocation(prog, 0, 'aCorner');
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    throw new Error('Program link failed: ' + log);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return { prog };
}

function _compileShader(type, src) {
  const gl = _gl;
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error('Shader compile failed: ' + log + '\n\nSource:\n' + src);
  }
  return sh;
}

// ─── Sprite GPU upload ────────────────────────────────────────────

function _ensureSpriteTexture(canvas) {
  let info = _spriteTexMap.get(canvas);
  if (info) return info;
  const gl = _gl;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  info = {
    tex,
    w: canvas.width,
    h: canvas.height,
    ox: canvas._ox || 0,
    oy: canvas._oy || 0,
  };
  _spriteTexMap.set(canvas, info);
  return info;
}

// ─── Color parsing ────────────────────────────────────────────────

const _colorRgbCache = new Map();
// Mid-gray fallback when an entity reaches the renderer without a color
// field. Prevents the entire runFrame() pipeline from throwing — see
// the long comment inside _colorToRgbNorm for the historical bug context.
const _COLOR_DEFAULT_RGB = [0.5, 0.5, 0.5];

function _colorToRgbNorm(c) {
  // Defensive: if an entity reaches the renderer without a color field
  // (e.g., constructed by external code that bypassed entities.createEntity),
  // fall back to mid-gray instead of throwing TypeError on charCodeAt.
  // Historical context: the crash was catastrophic — runFrame()'s catch
  // was rescheduling RAF without endFrame ever firing, so perf-mon /
  // state-dump showed "FPS 218" (raw RAF callback rate) while no actual
  // render or physics state-pull completed. Discovered 2026-05-25 while
  // diagnosing user-reported stalls.
  if (c == null) return _COLOR_DEFAULT_RGB;
  const cached = _colorRgbCache.get(c);
  if (cached) return cached;
  let r, g, b;
  if (c.charCodeAt(0) === 0x23 /* '#' */) {
    const hex = c.slice(1);
    if (hex.length === 6) {
      r = parseInt(hex.slice(0, 2), 16);
      g = parseInt(hex.slice(2, 4), 16);
      b = parseInt(hex.slice(4, 6), 16);
    } else if (hex.length === 3) {
      r = parseInt(hex[0] + hex[0], 16);
      g = parseInt(hex[1] + hex[1], 16);
      b = parseInt(hex[2] + hex[2], 16);
    } else {
      r = g = b = 128;
    }
  } else {
    const m = c.match(/hsl\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)%\s*,\s*(\d+(?:\.\d+)?)%\s*\)/);
    if (m) {
      [r, g, b] = _hslToRgb(+m[1], +m[2], +m[3]);
    } else {
      r = g = b = 128;
    }
  }
  const norm = [r / 255, g / 255, b / 255];
  _colorRgbCache.set(c, norm);
  return norm;
}

function _hslToRgb(h, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => {
    const v = l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return Math.round(255 * v);
  };
  return [f(0), f(8), f(4)];
}

// ─── Public: reset trail ──────────────────────────────────────────

export function resetTrailCanvas() {
  // Also resets the V9.1 contour-EMA state. Called from the "Clear
  // Sandbox" button → if we kept the EMAs across a scene clear, the next
  // (typically smaller) scene would rebuild its contour shape slowly
  // over ~12 frames, showing visibly wrong rings during that ramp.
  _smoothedContourThreshold = 0;
  _smoothedLogK = 0;
  if (_disabled || !_gl) return;
  if (!_fboA || !_fboB) return;
  const gl = _gl;
  gl.bindFramebuffer(gl.FRAMEBUFFER, _fboA);
  gl.viewport(0, 0, _fboW, _fboH);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, _fboB);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  _trailReadIdx = 0;
}

// ─── Public: update trail FBO (decay + dot plot) ──────────────────

export function updateTrailCanvas(simDeltaTime) {
  if (_disabled || !_gl) return;
  if (_vpW <= 0 || _vpH <= 0) return;
  if (!_fboA || !_fboB) return;
  const gl = _gl;

  // Slider 0-500 → lifetime in sim sec = slider / 50. 0 → instant wipe.
  const lifetime = state.trailLength / 50;
  if (lifetime <= 0) {
    resetTrailCanvas();
    return;
  }
  if (simDeltaTime <= 0) {
    // Paused — leave trail buffer alone.
    return;
  }

  // Alpha decrement in [0..1] space. Cap to 1 (instant wipe per pixel).
  const uDec = Math.min(1, simDeltaTime / lifetime);

  const readTex  = _trailReadIdx === 0 ? _texA : _texB;
  const writeFbo = _trailReadIdx === 0 ? _fboB : _fboA;

  // === Pass 1: decay (sample read tex, write decayed result into write fbo) ===
  gl.bindFramebuffer(gl.FRAMEBUFFER, writeFbo);
  gl.viewport(0, 0, _fboW, _fboH);
  gl.disable(gl.BLEND);
  gl.useProgram(_progTrailDecay.prog);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, readTex);
  gl.uniform1i(_progTrailDecay.uTex, 0);
  gl.uniform1f(_progTrailDecay.uDec, uDec);

  gl.bindVertexArray(_vaoFsQuad);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);

  // === Pass 2: dot plot (instanced quads, MAX blend) ===
  const ents = state.entities;
  const needed = ents.length * 5;
  if (_trailInstanceData.length < needed) {
    _trailInstanceData = new Float32Array(Math.max(needed, _trailInstanceData.length * 2));
  }
  let count = 0;
  for (let k = 0; k < ents.length; k++) {
    const e = ents[k];
    if (e.absorbing) continue;
    const rgb = _colorToRgbNorm(e.color);
    const o = count * 5;
    // V10.3: trail dot at raw world coords; the trail-dot VS's
    // sagProject samples the same sag texture as the mesh / body,
    // so the dot lands exactly under the body that drew it.
    // TODO(v11.5): trail dots emit no wrap-mirror copies, unlike body
    // sprites (see _drawEntities's wrap block). In rubber-sheet + wrap
    // mode the dot drawn just after a body wraps lands at
    // screen_y ≈ +sag·yFactor (up to ~84 px below the top edge) with
    // no mirror on the opposite side. Visually subtle because the
    // trail FBO is a phosphor-decay accumulator — a single missed dot
    // smears into neighbors within ~2 frames. Acceptable for V11.4.
    _trailInstanceData[o]     = e.x;
    _trailInstanceData[o + 1] = e.y;
    _trailInstanceData[o + 2] = rgb[0];
    _trailInstanceData[o + 3] = rgb[1];
    _trailInstanceData[o + 4] = rgb[2];
    count++;
  }

  if (count > 0) {
    const R = state.trailWidth * 0.5;
    const Ri = Math.max(0, R - 0.5);
    const Ro = R + 0.5;

    gl.useProgram(_progTrailDot.prog);
    gl.bindVertexArray(_vaoTrailDot);
    gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceTrail);
    gl.bufferData(gl.ARRAY_BUFFER, _trailInstanceData.subarray(0, count * 5), gl.DYNAMIC_DRAW);

    gl.uniformMatrix4fv(_progTrailDot.uOrtho, false, _orthoMat);
    gl.uniform1f(_progTrailDot.uRinner, Ri);
    gl.uniform1f(_progTrailDot.uRouter, Ro);
    // V10.3: trail dot iCenter is raw world; the shader's sagProject
    // does the projection from the shared sag texture.
    _bindSagUniforms(_progTrailDot);

    // MAX blend: dest = max(src, dest) per channel.
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.MAX);
    gl.blendFunc(gl.ONE, gl.ONE);

    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
  }

  // Reset GL state unconditionally (even when count===0) so downstream passes
  // see a clean slate. Without this, a zero-entity frame leaves whatever
  // equation the previous frame's dot-plot finished with — which would be
  // wrong if updateTrailCanvas is ever called outside the standard order.
  gl.bindVertexArray(null);
  gl.blendEquation(gl.FUNC_ADD);
  gl.disable(gl.BLEND);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  _trailReadIdx = 1 - _trailReadIdx;
}

// ─── Public: per-frame draw scene ─────────────────────────────────

export function drawScene() {
  if (_disabled || !_gl) return;
  if (_vpW <= 0 || _vpH <= 0) return;
  const gl = _gl;

  // 1. Clear screen with background color.
  const bg = _colorToRgbNorm(state.bgColor);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, Math.round(_vpW * _dpr), Math.round(_vpH * _dpr));
  gl.clearColor(bg[0], bg[1], bg[2], 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // 2. Blit trail FBO onto screen with straight-alpha blending.
  if (state.trailLength > 0 && _fboA && _fboB) {
    const readTex = _trailReadIdx === 0 ? _texA : _texB;
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(_progTrailBlit.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, readTex);
    gl.uniform1i(_progTrailBlit.uTex, 0);

    gl.bindVertexArray(_vaoFsQuad);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  // 3. Entity sprite instanced draw, bucketed by sprite canvas.
  _drawEntities();
}

function _drawEntities() {
  const gl = _gl;
  const wrap = state.boundaryMode === 'wrap';
  const W = _vpW;
  const H = _vpH;

  // Bucket entities by their sprite canvas. Skip absorbing entities — their
  // per-frame alpha varies so they're drawn by drawUI() via _progCircleFill.
  const buckets = new Map(); // canvas → flat array of (x, y, alpha) per instance
  const ents = state.entities;

  for (let i = 0; i < ents.length; i++) {
    const e = ents[i];
    if (e.absorbing) continue;
    const sprite = ensureEntitySprite(e);
    if (!sprite) continue;
    let arr = buckets.get(sprite);
    if (!arr) { arr = []; buckets.set(sprite, arr); }
    // V10.3: push raw world coords; the entity VS's sagProject samples
    // the same sag texture as the mesh so the body lands at the mesh's
    // visible well bottom by construction.
    arr.push(e.x, e.y, 1);

    if (wrap) {
      // Threshold uses sprite half-extent (sprite.width / 2), not entity
      // radius. The sprite has SPRITE_PADDING_PX (=8) of headroom on each
      // side for pinned-ring + AA — so its visible edge sits at
      // (entity radius + 8) from the entity center. Using bare e.radius
      // would cause the pinned-ring / AA edge to clip at the wrap boundary
      // even though the mirror copy hasn't been spawned yet. `sprite._ox`
      // (= sprite.width / 2) is exactly the half-extent we need.
      //
      // V11.4 (2026-05-28): split X and Y thresholds. With rubber-sheet
      // sag ON, a body sprite renders at screen_y = world_y + sag·yFactor
      // (see SAG_VS_HELPER in shaders.js). When the body sits near the
      // top edge, its sprite can extend up to `_maxSagY` (the frame's
      // peak projected offset) below the body center on screen. So the
      // wrap-mirror needs to spawn whenever world_y is within
      // (sprite._ox + _maxSagY) of the edge — wider than the X-axis,
      // which has no projection. _maxSagY = 0 when sag is off, restoring
      // the old single-`r` behavior for the non-rubber-sheet path.
      const rX = sprite._ox;
      const rY = sprite._ox + _maxSagY;
      const nearLeft   = e.x < rX;
      const nearRight  = e.x > W - rX;
      const nearTop    = e.y < rY;
      const nearBottom = e.y > H - rY;
      if (nearLeft)   arr.push(e.x + W, e.y, 1);
      if (nearRight)  arr.push(e.x - W, e.y, 1);
      if (nearTop)    arr.push(e.x, e.y + H, 1);
      if (nearBottom) arr.push(e.x, e.y - H, 1);
      if (nearLeft && nearTop)     arr.push(e.x + W, e.y + H, 1);
      if (nearLeft && nearBottom)  arr.push(e.x + W, e.y - H, 1);
      if (nearRight && nearTop)    arr.push(e.x - W, e.y + H, 1);
      if (nearRight && nearBottom) arr.push(e.x - W, e.y - H, 1);
    }
  }

  if (buckets.size === 0) return;

  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  gl.useProgram(_progEntity.prog);
  gl.uniformMatrix4fv(_progEntity.uOrtho, false, _orthoMat);
  // V10.3: entity VS's sagProject samples the same sag texture as the
  // mesh, so the body lands at the mesh's visible well bottom. No CPU
  // pre-bake; the texture binding + uSagMode comes from _bindSagUniforms.
  _bindSagUniforms(_progEntity);
  gl.bindVertexArray(_vaoEntity);

  for (const [sprite, posArr] of buckets) {
    const info = _ensureSpriteTexture(sprite);
    const inst = (posArr.length / 3) | 0;
    const needed = inst * 7;
    if (_entityInstanceData.length < needed) {
      _entityInstanceData = new Float32Array(Math.max(needed, _entityInstanceData.length * 2));
    }
    for (let k = 0; k < inst; k++) {
      const src = k * 3;
      const dst = k * 7;
      _entityInstanceData[dst]     = posArr[src];
      _entityInstanceData[dst + 1] = posArr[src + 1];
      _entityInstanceData[dst + 2] = info.w;
      _entityInstanceData[dst + 3] = info.h;
      _entityInstanceData[dst + 4] = info.ox;
      _entityInstanceData[dst + 5] = info.oy;
      _entityInstanceData[dst + 6] = posArr[src + 2];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceEntity);
    gl.bufferData(gl.ARRAY_BUFFER, _entityInstanceData.subarray(0, needed), gl.DYNAMIC_DRAW);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, info.tex);
    gl.uniform1i(_progEntity.uTex, 0);

    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, inst);
  }

  gl.bindVertexArray(null);
  // Match the cleanup discipline of updateTrailCanvas: leave BLEND in a
  // known-disabled state so any future pass that fails to re-enable it
  // (or a debug-time injected pass) starts from a deterministic baseline.
  gl.disable(gl.BLEND);
}

// ─── V9.0b UI overlay layers (WebGL) ──────────────────────────────
// drawUI() is called per frame from main.js after drawScene(). It populates
// the three UI scratch arrays from state.drag / state.hoverPos /
// state.selectedId / absorbing entities, then issues one instanced draw per
// shader program (or zero if a program has no work this frame).

// Visual constants — dash lengths / alpha / ring color for the UI overlay.
const UI_SELECT_RING_COLOR = [0x6b / 255, 0x8c / 255, 0xff / 255]; // #6b8cff
const UI_GHOST_FILL_ALPHA = 0.18;
const UI_PREDICTION_BATCHES = 8;
const UI_PREDICTION_DASH_ON = 6;     // px
const UI_PREDICTION_DASH_OFF = 6;
const UI_RUBBER_BAND_DASH_ON = 5;    // px
const UI_RUBBER_BAND_DASH_OFF = 4;
const UI_HOVER_DASH_ON = 3;          // px (V8.1c [3, 3])
const UI_HOVER_DASH_OFF = 3;
const UI_SELECT_DASH_ON = 4;         // px (V8.1c [4, 4])
const UI_SELECT_DASH_OFF = 4;

// V10.3: UI overlays push raw world coordinates. The UI shaders
// (CIRCLE_FILL / CIRCLE_RING / LINE_SEG) sample the shared sag
// texture via sagProject so hover ghost / drag preview / prediction
// line / selection ring / absorbing fallback all align with the
// mesh's visible depression at their world position.
function _pushCircleFill(cx, cy, radius, r, g, b, a) {
  const need = (_circleFillCount + 1) * 7;
  if (_circleFillData.length < need) {
    _circleFillData = _grow(_circleFillData, need);
  }
  const o = _circleFillCount * 7;
  _circleFillData[o]     = cx;
  _circleFillData[o + 1] = cy;
  _circleFillData[o + 2] = radius;
  _circleFillData[o + 3] = r;
  _circleFillData[o + 4] = g;
  _circleFillData[o + 5] = b;
  _circleFillData[o + 6] = a;
  _circleFillCount++;
}

function _pushCircleRing(cx, cy, radius, r, g, b, a, lineW, dashOn, dashPeriod) {
  const need = (_circleRingCount + 1) * 10;
  if (_circleRingData.length < need) {
    _circleRingData = _grow(_circleRingData, need);
  }
  const o = _circleRingCount * 10;
  _circleRingData[o]     = cx;
  _circleRingData[o + 1] = cy;
  _circleRingData[o + 2] = radius;
  _circleRingData[o + 3] = r;
  _circleRingData[o + 4] = g;
  _circleRingData[o + 5] = b;
  _circleRingData[o + 6] = a;
  _circleRingData[o + 7] = lineW;
  _circleRingData[o + 8] = dashOn;
  _circleRingData[o + 9] = dashPeriod;
  _circleRingCount++;
}

function _pushLineSeg(x0, y0, x1, y1, r, g, b, a, arcStart, lineW, dashOn, dashPeriod) {
  const need = (_lineSegCount + 1) * 12;
  if (_lineSegData.length < need) {
    _lineSegData = _grow(_lineSegData, need);
  }
  const o = _lineSegCount * 12;
  _lineSegData[o]      = x0;
  _lineSegData[o + 1]  = y0;
  _lineSegData[o + 2]  = x1;
  _lineSegData[o + 3]  = y1;
  _lineSegData[o + 4]  = r;
  _lineSegData[o + 5]  = g;
  _lineSegData[o + 6]  = b;
  _lineSegData[o + 7]  = a;
  _lineSegData[o + 8]  = arcStart;
  _lineSegData[o + 9]  = lineW;
  _lineSegData[o + 10] = dashOn;
  _lineSegData[o + 11] = dashPeriod;
  _lineSegCount++;
}

function _grow(arr, neededLen) {
  let newLen = arr.length || 64;
  while (newLen < neededLen) newLen *= 2;
  // CRITICAL: copy existing data, otherwise every entry pushed before the
  // resize point is zeroed out. The push functions write element N into
  // the array, then the next push for element N+1 may trigger _grow —
  // without next.set(arr) the previously-written element N becomes 0 and
  // its instance renders at (0,0) with zero radius, vanishing from the
  // output. Caught by V9.0b reviewer.
  const next = new Float32Array(newLen);
  next.set(arr);
  return next;
}

export function drawUI() {
  if (_disabled || !_gl) return;
  if (_vpW <= 0 || _vpH <= 0) return;
  _circleFillCount = 0;
  _circleRingCount = 0;
  _lineSegCount = 0;

  _buildHoverGhost();
  _buildDragPreview();   // includes prediction path
  _buildAbsorbing();
  _buildSelectionRing();

  if (_circleFillCount === 0 && _circleRingCount === 0 && _lineSegCount === 0) return;

  const gl = _gl;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, Math.round(_vpW * _dpr), Math.round(_vpH * _dpr));
  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // Draw filled circles first (under outlines / lines).
  // V10.3: each UI shader's sagProject samples the shared sag texture
  // so overlays align with the mesh well at their world position.
  if (_circleFillCount > 0) {
    gl.useProgram(_progCircleFill.prog);
    gl.uniformMatrix4fv(_progCircleFill.uOrtho, false, _orthoMat);
    _bindSagUniforms(_progCircleFill);
    gl.bindVertexArray(_vaoCircleFill);
    gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceCircleFill);
    gl.bufferData(gl.ARRAY_BUFFER, _circleFillData.subarray(0, _circleFillCount * 7), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, _circleFillCount);
  }

  // Lines (rubber band + prediction path) above fills.
  if (_lineSegCount > 0) {
    gl.useProgram(_progLineSeg.prog);
    gl.uniformMatrix4fv(_progLineSeg.uOrtho, false, _orthoMat);
    _bindSagUniforms(_progLineSeg);
    gl.bindVertexArray(_vaoLineSeg);
    gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceLineSeg);
    gl.bufferData(gl.ARRAY_BUFFER, _lineSegData.subarray(0, _lineSegCount * 12), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, _lineSegCount);
  }

  // Ring strokes on top (outline of ghosts, handle, selection ring, BH edge).
  if (_circleRingCount > 0) {
    gl.useProgram(_progCircleRing.prog);
    gl.uniformMatrix4fv(_progCircleRing.uOrtho, false, _orthoMat);
    _bindSagUniforms(_progCircleRing);
    gl.bindVertexArray(_vaoCircleRing);
    gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceCircleRing);
    gl.bufferData(gl.ARRAY_BUFFER, _circleRingData.subarray(0, _circleRingCount * 10), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, _circleRingCount);
  }

  gl.bindVertexArray(null);
}

// hover ghost: filled translucent disk + 1-px dashed ring outline.
function _buildHoverGhost() {
  if (state.isEditMode) return;
  if (state.drag) return;
  if (!state.hoverPos) return;
  const colorStr = resolveDisplayColor(
    state.pending.type, state.pending.charge,
    state.pending.type === 'planet' ? '#ffffff' : '#000000',
  );
  const c = _colorToRgbNorm(colorStr);
  const r = state.pending.radius;
  const { x, y } = state.hoverPos;
  _pushCircleFill(x, y, r, c[0], c[1], c[2], UI_GHOST_FILL_ALPHA);
  _pushCircleRing(
    x, y, r, c[0], c[1], c[2], 1.0,
    1.0, UI_HOVER_DASH_ON, UI_HOVER_DASH_ON + UI_HOVER_DASH_OFF,
  );
}

// drag preview: ghost fill + solid outline at placement; dashed rubber-
// band line to cursor; hollow handle ring at cursor; dashed faded-along-
// arc prediction polyline (wrap-aware).
function _buildDragPreview() {
  const drag = state.drag;
  if (!drag) return;
  const colorStr = resolveDisplayColor(
    state.pending.type, state.pending.charge, drag.previewBaseColor,
  );
  const c = _colorToRgbNorm(colorStr);
  const radius = state.pending.radius;

  // 1. Ghost fill at placement.
  _pushCircleFill(drag.startX, drag.startY, radius, c[0], c[1], c[2], UI_GHOST_FILL_ALPHA);
  // 2. Solid 1-px outline.
  _pushCircleRing(drag.startX, drag.startY, radius, c[0], c[1], c[2], 1.0, 1.0, 0.0, 0.0);

  // 3. Dashed rubber band to cursor.
  _pushLineSeg(
    drag.startX, drag.startY, drag.currentX, drag.currentY,
    c[0], c[1], c[2], 1.0, 0.0, 1.5,
    UI_RUBBER_BAND_DASH_ON, UI_RUBBER_BAND_DASH_ON + UI_RUBBER_BAND_DASH_OFF,
  );

  // 4. Hollow handle ring at cursor.
  const dx = drag.currentX - drag.startX;
  const dy = drag.currentY - drag.startY;
  const dragDist = Math.hypot(dx, dy);
  const handleR = Math.max(6, Math.min(14, dragDist * 0.08 + 6));
  _pushCircleRing(drag.currentX, drag.currentY, handleR, c[0], c[1], c[2], 1.0, 2.0, 0.0, 0.0);

  // 5. Prediction path: dashed polyline with per-batch alpha fade.
  const path = drag.predictionPath;
  if (!path || path.length < 2) return;
  const data = path.data;
  const n = path.length;
  const batchSize = Math.max(2, Math.ceil(n / UI_PREDICTION_BATCHES));
  const wrap = state.boundaryMode === 'wrap';
  const wrapX = wrap ? _vpW * 0.5 : Infinity;
  const wrapY = wrap ? _vpH * 0.5 : Infinity;
  const dashPeriod = UI_PREDICTION_DASH_ON + UI_PREDICTION_DASH_OFF;

  // Walk the path segment by segment. Accumulate arcStart (running cum
  // length) so dash phase is continuous across the polyline. Reset arc
  // on a wrap-jump (the segment is skipped — no draw — so phase reset
  // there is invisible). Within each batch, alpha = (1 - b/N) * 0.85.
  // prevX/prevY carry forward across batch boundaries (the path is one
  // continuous polyline visually; batches only stagger alpha).
  let arc = 0;
  let prevX = data[0];
  let prevY = data[1];
  for (let b = 0; b < UI_PREDICTION_BATCHES; b++) {
    const start = b * batchSize;
    if (start >= n - 1) break;
    const end = Math.min(n - 1, start + batchSize);
    const alpha = (1 - b / UI_PREDICTION_BATCHES) * 0.85;
    for (let i = start + 1; i <= end; i++) {
      const x = data[i * 2];
      const y = data[i * 2 + 1];
      const adx = Math.abs(x - prevX);
      const ady = Math.abs(y - prevY);
      if (adx > wrapX || ady > wrapY) {
        // wrap-jump: skip drawing this segment; reset phase.
        prevX = x; prevY = y;
        arc = 0;
        continue;
      }
      const segLen = Math.hypot(x - prevX, y - prevY);
      if (segLen > 0) {
        _pushLineSeg(
          prevX, prevY, x, y,
          c[0], c[1], c[2], alpha, arc, 1.5,
          UI_PREDICTION_DASH_ON, dashPeriod,
        );
        arc += segLen;
      }
      prevX = x; prevY = y;
    }
  }
}

// absorbing entities: per-frame variable alpha, can't be sprite-cached.
// Fading filled disk; optionally a thin outline for black holes.
function _buildAbsorbing() {
  const ents = state.entities;
  for (let i = 0; i < ents.length; i++) {
    const e = ents[i];
    if (!e.absorbing) continue;
    const t = Math.min(1, e.absorbing.elapsedSim / state.absorptionDuration);
    const fade = Math.max(0, 1 - t);
    if (fade <= 0) continue;
    const r = Math.max(0, e.radius);
    const col = _colorToRgbNorm(e.color);
    _pushCircleFill(e.x, e.y, r, col[0], col[1], col[2], fade);

    if (e.type === 'black_hole') {
      // V9.8: thicker, brighter event-horizon ring + halo so the
      // body stays visible against any background (matches the
      // sprite-cache BH style; the absorbing fallback bypasses
      // the cache because alpha is per-frame).
      let er, eg, eb, eaCore, eaHalo;
      if (e.charge === -1) {
        er = 20 / 255; eg = 30 / 255; eb = 45 / 255;
        eaCore = 0.95; eaHalo = 0.30;
      } else {
        er = 160 / 255; eg = 210 / 255; eb = 255 / 255;
        eaCore = 1.00; eaHalo = 0.30;
      }
      _pushCircleRing(e.x, e.y, r + 2, er, eg, eb, eaHalo * fade, 5.0, 0.0, 0.0);
      _pushCircleRing(e.x, e.y, r,     er, eg, eb, eaCore * fade, 2.5, 0.0, 0.0);
    }
  }
}

function _buildSelectionRing() {
  if (state.selectedId === null) return;
  const sel = state.entities.find(e => e.id === state.selectedId);
  if (!sel) return;
  _pushCircleRing(
    sel.x, sel.y, sel.radius + 6,
    UI_SELECT_RING_COLOR[0], UI_SELECT_RING_COLOR[1], UI_SELECT_RING_COLOR[2], 1.0,
    2.0, UI_SELECT_DASH_ON, UI_SELECT_DASH_ON + UI_SELECT_DASH_OFF,
  );
}

// ─── V9.1 field visualization (drawField) ─────────────────────────
// Two passes, both gated by state.showField:
//   1. Equipotential contour lines — fullscreen quad fragment shader
//      sums φ = Σ -G·q·m / sqrt(r²+ε²) per pixel and draws contour rings
//      where |∇φ| × derivative-aware-mask ≤ 0.5 px.
//   2. Pulsing streamlines — instanced quad per grid seed; each seed
//      points along the local force direction (computed on CPU via
//      computeForceDirAt); the shader animates a "light strip" sliding
//      from seed to tip in sync across all seeds.
//
// Performance: with showField OFF the entire function early-exits before
// any GL state changes — zero overhead. With showField ON the cost is:
//   - 1 fullscreen FS pass with O(viewport_px × N) per-pixel work, where
//     N is min(entity count, MAX_FIELD_ENTITIES).
//   - 1 instanced draw with up to ~96 instances + 96 × N CPU force evals.

const UI_PULSE_PERIOD_MS = 1500;     // total cycle period (flight + pause)
const UI_PULSE_FLIGHT_MS = 1300;     // active flight; remainder is pause
const UI_PULSE_TAIL_FRAC = 0.25;     // trail length as fraction of streamline
const UI_STREAMLINE_LENGTH_PX = 50;
const UI_STREAMLINE_WIDTH_PX = 3;
const UI_CONTOUR_NUM_BANDS = 12;     // target visible bands across dominant scale
const UI_CONTOUR_LINE_W_PX = 1.0;
// Light-blue-white pulse strip; lighter alpha so it doesn't dominate the
// scene at peak.
const UI_STREAMLINE_COLOR = [180 / 255, 210 / 255, 255 / 255, 0.65];

// Pack non-absorbing, non-neutral entities into _fieldEntityData with
// 9-ghost PBC copies in wrap mode. Extracted so it can run BEFORE
// drawSceneGL (needed by V10 sag-texture prep) without coupling to
// drawField's heavier setup.
function _packFieldEntities() {
  _fieldEntityCount = 0;
  const wrap = state.boundaryMode === 'wrap';
  const W = _vpW, H = _vpH;
  const ents = state.entities;
  outer: for (let i = 0; i < ents.length; i++) {
    const e = ents[i];
    if (e.absorbing) continue;
    const Gqm = state.G * e.charge * e.mass;
    if (Gqm === 0) continue;
    const nLo = wrap ? -1 : 0;
    const nHi = wrap ? 1 : 0;
    for (let oy = nLo; oy <= nHi; oy++) {
      for (let ox = nLo; ox <= nHi; ox++) {
        if (_fieldEntityCount >= MAX_FIELD_ENTITIES) break outer;
        const o = _fieldEntityCount * 4;
        _fieldEntityData[o]     = e.x + ox * W;
        _fieldEntityData[o + 1] = e.y + oy * H;
        _fieldEntityData[o + 2] = Gqm;
        // V11.2: slot 3 = body collision radius, consumed by
        // _updateSagTexture for inside-radius quadratic interpolation.
        // 0 was unused before; readers that only need (x,y,Gqm) ignore it.
        _fieldEntityData[o + 3] = e.radius;
        _fieldEntityCount++;
      }
    }
  }
}

// V10 rubber-sheet: per-frame renderer prep. Must run BEFORE drawSceneGL
// so the sag texture is ready when entities + trails + UI sample it.
// Called from main.js's runFrame.
export function prepareFrameRenderer() {
  if (_disabled || !_gl) return;
  if (_vpW <= 0 || _vpH <= 0) return;
  const rubberSheet = (state.showField && state.fieldStyle === 'rubber-sheet');
  if (rubberSheet) {
    _packFieldEntities();
    _updateSagTexture();
    _sagActive = true;
    // V11.4 (2026-05-28): scan _sagPixels for the peak sag value, then
    // multiply by the cos(viewTilt) factor to get the maximum screen-Y
    // offset the sag projection adds this frame. _drawEntities uses this
    // to widen the wrap-mode ghost-spawn threshold on the Y axis so a
    // body whose visible sprite sits at world_y + max_sag·yFactor still
    // gets a mirror copy spawned in time when it nears the top/bottom
    // edge. Without this, bodies disappear for ~0.4 s after wrapping in
    // rubber-sheet mode because the wrap teleport happens at world_y=0
    // while the sprite is still drawn at screen_y ≈ 84 px (sag tail) —
    // far outside the old `sprite._ox` (~24 px) ghost-spawn band.
    //
    // Cost: ~65 k pass over the L1-hot Float32Array, < 0.5 ms. This is
    // a *second* scan of _sagPixels (the first happens inside
    // _computeFieldIntensityRange's mode==2 branch during drawField),
    // because that call sits later in the frame than _drawEntities. The
    // two could be fused by either tracking max inline in
    // _updateSagTexture or running _computeFieldIntensityRange earlier,
    // but the win is sub-ms and not worth the ordering refactor today.
    let mx = 0;
    const N = _sagPixels.length;
    for (let i = 0; i < N; i++) {
      const v = _sagPixels[i];
      if (v > mx) mx = v;
    }
    _maxSagY = mx * _sagYFactor();
    _maxSagRaw = mx;
  } else {
    _sagActive = false;
    _maxSagY = 0;
    _maxSagRaw = 0;
  }
}

// V10.3 (2026-05-27): the V10.2 analytical Newton-sum per-body sag was
// pulled out per user direction — "投影前后的物体在2D平面上呈现相
// 近 … 通过对齐网格凹陷处和物体就行" (the body before/after projection
// looks similar in 2D; just align it with the mesh's depression).
// Bodies, trails, and UI overlays now sample the SAME 256×256 sag
// texture the GRID_WARP mesh uses (via sagProject in the VS). The
// body's sampled Y at its world position equals the mesh's bilinearly-
// interpolated well bottom at that same position by construction —
// they wobble together as the body crosses sample boundaries but stay
// visually locked. Trade-off: both have texture-aliasing wobble in
// unison (preferred); the previous analytical path eliminated wobble
// for bodies but left them visually disjoint from the mesh.

export function drawField() {
  if (_disabled || !_gl) return;
  if (!state.showField) return;
  if (_vpW <= 0 || _vpH <= 0) return;
  const gl = _gl;

  // Pack entity data into the GPU uniform array (same data used by
  // prepareFrameRenderer for the sag texture; safe to repack here as
  // it's idempotent given the same state).
  _packFieldEntities();
  // V12 (2026-05-28): rubber-sheet FS path renders a flat grid even when
  // the scene is empty (no bodies = no warp → straight grid lines). The
  // mesh paths (curvilinear/legacy/2d/3d) still early-out because their
  // CPU uniform uploads + N-body inner loops are wasted with no bodies.
  if (_fieldEntityCount === 0 &&
      state.fieldStyle !== 'rubber-sheet' && state.fieldStyle !== 'screen') return;

  // Common GL state for both passes.
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, Math.round(_vpW * _dpr), Math.round(_vpH * _dpr));
  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // V9.2 rewrite (2026-05-26): default field viz is now grid warp.
  // Legacy equipotential rings are kept gated behind state.fieldStyle
  // === 'legacy' for regression debugging; default is '2d' (user pick).
  // Particle-flow overlay companions the 2D grid for "luminous dust"
  // feel; in 3D and legacy modes it's omitted (those carry their own
  // visual character).
  // V9.9: 'curvilinear' default — equipotential rings + radial
  // geodesic field lines from each body. Fold-free by construction.
  // 'legacy' = rings only (no field lines). '2d'/'3d' = old GRID_WARP.
  if (state.fieldStyle === 'screen') {
    // Default: punched fly-screen dimples. Clean — no rings / no dust.
    _drawScreenDent();
  } else if (state.fieldStyle === 'curvilinear') {
    _drawEquipotential();
    _drawFieldLines();
  } else if (state.fieldStyle === 'legacy') {
    _drawEquipotential();
  } else if (state.fieldStyle === 'rubber-sheet') {
    // V10: rubber-sheet mesh (mode=2 in GRID_WARP) — VS samples the
    // per-frame normalized sag texture so the mesh dips into wells
    // in lockstep with bodies / trails / UI (all sharing the same
    // texture via _bindSagUniforms). Avoids the legacy mode=0 path's
    // per-vertex φ saturation that made the surface look flat.
    _drawGridWarp();
  } else {
    _drawGridWarp();
    if (state.fieldStyle === '2d') _drawParticleFlow();
  }
  _drawStreamlines();

  gl.disable(gl.BLEND);
}

// Sample points (in CSS-px, viewport-relative fractions) used by
// _estimatePhiRange to derive a contour-spacing matched to the actual
// φ-range visible on screen this frame. Corners + center + edge midpoints
// (9 samples) catch both far-field zero and any near-field extremes the
// user can see.
const _PHI_SAMPLE_FRACS = [
  [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
  [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
  [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
];

function _drawEquipotential() {
  const gl = _gl;
  // When wrap mode is active, CPU samplers must use the same 9-ghost PBC
  // sum the GPU shader does (via the ghost copies pre-packed into the
  // uniform array). Otherwise the φ range we compute here disagrees
  // with what the shader actually renders.
  const boundary = (state.boundaryMode === 'wrap')
    ? { width: _vpW, height: _vpH }
    : null;

  // ─── Stage 1: find φ extremes for threshold ─────────────────────
  // Sample at viewport reference points + each visible entity center.
  // The deepest extremes (= largest |φ|) live at entity centers due to
  // Plummer softening: |φ_max_at_body| = G·|q·m|/ε.
  let phiMin = +Infinity;
  let phiMax = -Infinity;
  for (let i = 0; i < _PHI_SAMPLE_FRACS.length; i++) {
    const sx = _PHI_SAMPLE_FRACS[i][0] * _vpW;
    const sy = _PHI_SAMPLE_FRACS[i][1] * _vpH;
    const phi = computePotentialAt(sx, sy, state.entities, state.G, state.epsilon, boundary);
    if (phi < phiMin) phiMin = phi;
    if (phi > phiMax) phiMax = phi;
  }
  for (let i = 0; i < state.entities.length; i++) {
    const e = state.entities[i];
    if (e.absorbing) continue;
    if (e.charge * e.mass === 0) continue;
    const phi = computePotentialAt(e.x, e.y, state.entities, state.G, state.epsilon, boundary);
    if (phi < phiMin) phiMin = phi;
    if (phi > phiMax) phiMax = phi;
  }
  const phiTop = Math.max(Math.abs(phiMin), Math.abs(phiMax));
  if (!isFinite(phiTop) || phiTop <= 1e-6) return;

  // ─── Stage 2: adaptive k from mass extremes ─────────────────────
  // k = ratio of |φ| between adjacent rings. Adaptive so that the
  // lightest emitter still gets RINGS_FOR_LIGHTEST visible rings AND
  // the heaviest gets NUM_BANDS - RINGS_FOR_LIGHTEST rings:
  //   k = (m_max / m_min)^(1 / (NUM_BANDS - RINGS_FOR_LIGHTEST))
  // When all emitters have similar |q·m|, default to a √2 ratio so
  // we get a familiar topographic feel without quirky aliasing.
  let qmMax = 0;
  let qmMin = Infinity;
  for (let i = 0; i < state.entities.length; i++) {
    const e = state.entities[i];
    if (e.absorbing) continue;
    const qm = Math.abs(e.charge * e.mass);
    if (qm <= 0) continue;
    if (qm > qmMax) qmMax = qm;
    if (qm < qmMin) qmMin = qm;
  }
  const RINGS_FOR_LIGHTEST = 3;
  const ratio = (qmMax > 0 && qmMin > 0 && qmMin < qmMax) ? qmMax / qmMin : 1;
  let k;
  if (ratio < 2) {
    k = Math.SQRT2;     // ≈ 1.414 — pleasant default for ~uniform masses
  } else {
    const exponent = 1 / Math.max(1, UI_CONTOUR_NUM_BANDS - RINGS_FOR_LIGHTEST);
    k = Math.pow(ratio, exponent);
  }
  // Clamp to a sane visual range (avoids k → 1 which would make rings
  // collapse onto each other, and k → ∞ which would put adjacent rings
  // factor-of-10 apart and skip masses entirely).
  k = Math.max(1.2, Math.min(2.5, k));

  // ─── Stage 3: threshold (outermost ring's |φ|) ──────────────────
  // Rings live at |φ| = threshold · k^n for n = 0..NUM_BANDS-1.
  // Outermost ring (n=0) at threshold; innermost (n=NUM_BANDS-1) at
  // threshold·k^(NUM_BANDS-1). We want the innermost to reach phiTop,
  // so threshold = phiTop / k^(NUM_BANDS-1).
  const instantThreshold = phiTop / Math.pow(k, UI_CONTOUR_NUM_BANDS - 1);
  const instantLogK = Math.log(k);

  // ─── Stage 4: EMA-smooth threshold and logK ─────────────────────
  if (_smoothedContourThreshold <= 1e-9) {
    _smoothedContourThreshold = instantThreshold;
    _smoothedLogK = instantLogK;
  } else {
    _smoothedContourThreshold = 0.92 * _smoothedContourThreshold + 0.08 * instantThreshold;
    _smoothedLogK             = 0.92 * _smoothedLogK             + 0.08 * instantLogK;
  }

  // ─── Stage 5: theme-aware color ─────────────────────────────────
  const bgRgb = _colorToRgbNorm(state.bgColor);
  const bgIsLight = (bgRgb[0] + bgRgb[1] + bgRgb[2]) > 1.5;
  const cR = bgIsLight ? 0.16 : 0.86;
  const cG = bgIsLight ? 0.16 : 0.86;
  const cB = bgIsLight ? 0.20 : 0.90;
  const cA = bgIsLight ? 0.45 : 0.35;

  // ─── Stage 6: draw ──────────────────────────────────────────────
  gl.useProgram(_progEquipotential.prog);
  gl.uniform2f(_progEquipotential.uViewport, _vpW, _vpH);
  gl.uniform4fv(_progEquipotential.uEntities, _fieldEntityData);
  gl.uniform1i(_progEquipotential.uEntityCount, _fieldEntityCount);
  gl.uniform1f(_progEquipotential.uEpsilon, state.epsilon);
  gl.uniform1f(_progEquipotential.uLogK, _smoothedLogK);
  gl.uniform1f(_progEquipotential.uContourThreshold, _smoothedContourThreshold);
  gl.uniform1f(_progEquipotential.uContourLineW, UI_CONTOUR_LINE_W_PX);
  gl.uniform1f(_progEquipotential.uDpr, _dpr);
  gl.uniform4f(_progEquipotential.uColor, cR, cG, cB, cA);
  gl.bindVertexArray(_vaoFsQuad);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
}

function _drawStreamlines() {
  const gl = _gl;
  if (_streamlineSeedCount === 0) return;

  // Synchronized pulse phase. uPulseHead ∈ [0, 1] during flight; we skip
  // the draw call entirely during the 0.2 s pause window between cycles.
  const now = performance.now();
  if (_pulseStartMs === 0) _pulseStartMs = now;
  const phaseMs = (now - _pulseStartMs) % UI_PULSE_PERIOD_MS;
  if (phaseMs >= UI_PULSE_FLIGHT_MS) return;  // pause window — no strips visible
  const pulseHead = phaseMs / UI_PULSE_FLIGHT_MS;

  // Build per-instance data: (seedX, seedY, dirX, dirY, alpha) × N.
  // alpha scales with log(force magnitude) so weak-field zones fade out
  // and don't add visual noise. Clamped to [0.15, 1.0].
  const N = _streamlineSeedCount;
  const need = N * 5;
  if (_streamlineInstanceData.length < need) {
    _streamlineInstanceData = new Float32Array(need);
  }
  const boundary = (state.boundaryMode === 'wrap')
    ? { width: _vpW, height: _vpH }
    : null;
  let liveCount = 0;
  for (let i = 0; i < N; i++) {
    const sx = _streamlineSeeds[i * 2];
    const sy = _streamlineSeeds[i * 2 + 1];
    const f = computeForceDirAt(sx, sy, state.entities, state.G, state.epsilon, boundary);
    if (f.mag === 0) continue;                // skip null-field seeds
    // Use log-scale to map force magnitude to brightness, then clamp.
    // mag near a heavy entity can be huge; far-field is tiny. Log compresses.
    const logMag = Math.log10(1 + f.mag);
    const alpha = Math.max(0.15, Math.min(1.0, logMag / 4.0));
    const o = liveCount * 5;
    _streamlineInstanceData[o]     = sx;
    _streamlineInstanceData[o + 1] = sy;
    _streamlineInstanceData[o + 2] = f.x;
    _streamlineInstanceData[o + 3] = f.y;
    _streamlineInstanceData[o + 4] = alpha;
    liveCount++;
  }
  if (liveCount === 0) return;

  gl.useProgram(_progStreamline.prog);
  gl.uniformMatrix4fv(_progStreamline.uOrtho, false, _orthoMat);
  gl.uniform1f(_progStreamline.uLength, UI_STREAMLINE_LENGTH_PX);
  gl.uniform1f(_progStreamline.uLineW, UI_STREAMLINE_WIDTH_PX);
  gl.uniform1f(_progStreamline.uPulseHead, pulseHead);
  gl.uniform1f(_progStreamline.uPulseTailFrac, UI_PULSE_TAIL_FRAC);
  gl.uniform4f(_progStreamline.uColor,
    UI_STREAMLINE_COLOR[0], UI_STREAMLINE_COLOR[1],
    UI_STREAMLINE_COLOR[2], UI_STREAMLINE_COLOR[3]);
  gl.bindVertexArray(_vaoStreamline);
  gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceStreamline);
  gl.bufferData(gl.ARRAY_BUFFER, _streamlineInstanceData.subarray(0, liveCount * 5), gl.DYNAMIC_DRAW);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, liveCount);
  gl.bindVertexArray(null);
}
