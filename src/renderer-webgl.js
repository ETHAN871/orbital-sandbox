// renderer-webgl.js — V9.0b WebGL 2 renderer (full pipeline).
//
// V9.0b is the final state of the "all rendering on GPU" migration. The
// Canvas2D overlay (`#overlay` canvas + renderer.js) is gone; every visible
// pixel is produced by a WebGL 2 shader.
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
//     (kept in _spriteTexMap). Sprite cache caps at 200 → ≤200 textures,
//     ~16MB total VRAM upper bound. No explicit GPU-texture eviction (the
//     sprite-cache.js LRU caps CPU-side canvases; stale GL textures linger
//     until context loss or page reload).
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
import { ensureEntitySprite } from './sprite-cache.js';
import { resolveDisplayColor } from './entities.js';
import {
  TRAIL_DECAY, TRAIL_BLIT, TRAIL_DOT, ENTITY,
  CIRCLE_FILL, CIRCLE_RING, LINE_SEG,
  EQUIPOTENTIAL, STREAMLINE,
} from './shaders.js';
import { computePotentialAt, computeForceDirAt } from './potential.js';

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
// V9.x: exponential moving average of the contour band spacing. The
// (φmax-φmin)/numBands estimate jitters frame-to-frame as entities move
// — even small shifts make every contour line slide one or two pixels
// every frame, producing a visible "wobble". Low-pass-filter the spacing
// with an EMA so it tracks gross changes in ~12 frames (~200 ms at 60Hz)
// but ignores per-frame noise. Reset to the latest sample when zero
// (first frame after init or scene clear) so we don't start at zero.
let _smoothedContourSpacing = 0;

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

  // Listen for context-loss recovery so a GPU hiccup doesn't kill the canvas.
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    console.warn('[renderer-webgl] context lost');
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

  _progEquipotential.aPos             = gl.getAttribLocation(_progEquipotential.prog, 'aPos');
  _progEquipotential.uViewport        = gl.getUniformLocation(_progEquipotential.prog, 'uViewport');
  _progEquipotential.uEntities        = gl.getUniformLocation(_progEquipotential.prog, 'uEntities[0]');
  _progEquipotential.uEntityCount     = gl.getUniformLocation(_progEquipotential.prog, 'uEntityCount');
  _progEquipotential.uEpsilon         = gl.getUniformLocation(_progEquipotential.prog, 'uEpsilon');
  _progEquipotential.uContourSpacing  = gl.getUniformLocation(_progEquipotential.prog, 'uContourSpacing');
  _progEquipotential.uContourLineW    = gl.getUniformLocation(_progEquipotential.prog, 'uContourLineW');
  _progEquipotential.uDpr             = gl.getUniformLocation(_progEquipotential.prog, 'uDpr');
  _progEquipotential.uColor           = gl.getUniformLocation(_progEquipotential.prog, 'uColor');

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

export function resizeRenderer(w, h, dpr) {
  if (_disabled || !_gl) return;
  _vpW = Math.max(1, w | 0);
  _vpH = Math.max(1, h | 0);
  _dpr = dpr || 1;
  _updateOrthoMat();
  _initFbos(_vpW, _vpH);
  _rebuildStreamlineSeeds();
}

// V9.1: build a (cols × rows) grid of streamline seed positions covering
// the viewport with margins. Target seed count is ~96 (per user choice);
// we resolve to the integer (cols, rows) closest to that target that
// matches the viewport aspect ratio. Recomputed only on viewport change.
function _rebuildStreamlineSeeds() {
  const targetCount = 96;
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

function _colorToRgbNorm(c) {
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
  // Also resets the V9.1 contour-spacing EMA. Called from the "Clear
  // Sandbox" button → if we kept the EMA across a scene clear, the next
  // (typically smaller) scene would rebuild its contour spacing slowly
  // over ~12 frames, showing visibly wrong spacing during that ramp.
  _smoothedContourSpacing = 0;
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
    arr.push(e.x, e.y, 1);

    if (wrap) {
      // Threshold uses sprite half-extent (sprite.width / 2), not entity
      // radius. The sprite has SPRITE_PADDING_PX (=8) of headroom on each
      // side for pinned-ring + AA — so its visible edge sits at
      // (entity radius + 8) from the entity center. Using bare e.radius
      // would cause the pinned-ring / AA edge to clip at the wrap boundary
      // even though the mirror copy hasn't been spawned yet. `sprite._ox`
      // (= sprite.width / 2) is exactly the half-extent we need.
      const r = sprite._ox;
      const nearLeft   = e.x < r;
      const nearRight  = e.x > W - r;
      const nearTop    = e.y < r;
      const nearBottom = e.y > H - r;
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

// Visual constants (matched to V8.1c values to keep visuals identical across the migration).
const UI_SELECT_RING_COLOR = [0x6b / 255, 0x8c / 255, 0xff / 255]; // #6b8cff
const UI_GHOST_FILL_ALPHA = 0.18;
const UI_PREDICTION_BATCHES = 8;
const UI_PREDICTION_DASH_ON = 6;     // px (V8.1c [6, 6])
const UI_PREDICTION_DASH_OFF = 6;
const UI_RUBBER_BAND_DASH_ON = 5;    // px (V8.1c [5, 4])
const UI_RUBBER_BAND_DASH_OFF = 4;
const UI_HOVER_DASH_ON = 3;          // px (V8.1c [3, 3])
const UI_HOVER_DASH_OFF = 3;
const UI_SELECT_DASH_ON = 4;         // px (V8.1c [4, 4])
const UI_SELECT_DASH_OFF = 4;

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
  if (_circleFillCount > 0) {
    gl.useProgram(_progCircleFill.prog);
    gl.uniformMatrix4fv(_progCircleFill.uOrtho, false, _orthoMat);
    gl.bindVertexArray(_vaoCircleFill);
    gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceCircleFill);
    gl.bufferData(gl.ARRAY_BUFFER, _circleFillData.subarray(0, _circleFillCount * 7), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, _circleFillCount);
  }

  // Lines (rubber band + prediction path) above fills.
  if (_lineSegCount > 0) {
    gl.useProgram(_progLineSeg.prog);
    gl.uniformMatrix4fv(_progLineSeg.uOrtho, false, _orthoMat);
    gl.bindVertexArray(_vaoLineSeg);
    gl.bindBuffer(gl.ARRAY_BUFFER, _bufInstanceLineSeg);
    gl.bufferData(gl.ARRAY_BUFFER, _lineSegData.subarray(0, _lineSegCount * 12), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, _lineSegCount);
  }

  // Ring strokes on top (outline of ghosts, handle, selection ring, BH edge).
  if (_circleRingCount > 0) {
    gl.useProgram(_progCircleRing.prog);
    gl.uniformMatrix4fv(_progCircleRing.uOrtho, false, _orthoMat);
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
      // V8.1c: charge -1 (white BH) → rgba(0,0,0,0.55); else rgba(120,180,255,0.75).
      let er, eg, eb, ea;
      if (e.charge === -1) {
        er = 0; eg = 0; eb = 0; ea = 0.55;
      } else {
        er = 120 / 255; eg = 180 / 255; eb = 255 / 255; ea = 0.75;
      }
      _pushCircleRing(e.x, e.y, r, er, eg, eb, ea * fade, 1.5, 0.0, 0.0);
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

export function drawField() {
  if (_disabled || !_gl) return;
  if (!state.showField) return;
  if (_vpW <= 0 || _vpH <= 0) return;
  const gl = _gl;

  // Pack entity data into the GPU uniform array. In normal (destroy)
  // boundary mode each non-absorbing, non-neutral entity contributes a
  // single entry. In wrap mode each such entity contributes its 9-ghost
  // PBC neighbourhood (itself + 8 mirror copies offset by ±W and ±H), so
  // the field continuously hands off across the wrap boundary instead of
  // discontinuously snapping when a body teleports. Matches the 9-ghost
  // PBC physics in physics.js — without this the streamlines would
  // disagree with where the bodies actually feel force.
  //
  // MAX_FIELD_ENTITIES caps total entries (entity × ghost). At 9× cost
  // per entity in wrap mode the budget is ~14 visible entities before
  // truncation; truncated ghosts just visually approximate but do not
  // crash. Acceptable for the typical scene; user can disable field viz
  // if they spawn 20+ bodies in wrap mode.
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
        _fieldEntityData[o + 3] = 0;
        _fieldEntityCount++;
      }
    }
  }
  if (_fieldEntityCount === 0) return;       // empty scene → nothing to draw

  // Common GL state for both passes.
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, Math.round(_vpW * _dpr), Math.round(_vpH * _dpr));
  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  _drawEquipotential();
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
  // uniform array). Otherwise the (φmax-φmin) range we compute here
  // disagrees with what the shader actually renders, and the contour
  // spacing falls out of sync at the wrap boundary.
  const boundary = (state.boundaryMode === 'wrap')
    ? { width: _vpW, height: _vpH }
    : null;

  // Adaptive contour spacing: sample φ at 9 viewport reference points
  // PLUS each visible entity's position (where φ-extremes always live —
  // near ε floor depth). Spacing = (φmax − φmin) / numBands puts roughly
  // numBands visible contour lines across the actual rendered φ-range,
  // regardless of how strong or weak the dominant entity is.
  let phiMin = +Infinity;
  let phiMax = -Infinity;
  for (let i = 0; i < _PHI_SAMPLE_FRACS.length; i++) {
    const sx = _PHI_SAMPLE_FRACS[i][0] * _vpW;
    const sy = _PHI_SAMPLE_FRACS[i][1] * _vpH;
    const phi = computePotentialAt(sx, sy, state.entities, state.G, state.epsilon, boundary);
    if (phi < phiMin) phiMin = phi;
    if (phi > phiMax) phiMax = phi;
  }
  // Sample at each VISIBLE entity (use state.entities, not the packed
  // ghost array) so the deepest wells / highest peaks of φ are caught.
  for (let i = 0; i < state.entities.length; i++) {
    const e = state.entities[i];
    if (e.absorbing) continue;
    if (e.charge * e.mass === 0) continue;
    const phi = computePotentialAt(e.x, e.y, state.entities, state.G, state.epsilon, boundary);
    if (phi < phiMin) phiMin = phi;
    if (phi > phiMax) phiMax = phi;
  }
  const phiRange = phiMax - phiMin;
  // If all samples collapse to the same value (no field variation visible),
  // skip the pass entirely — there's nothing to draw.
  if (!isFinite(phiRange) || phiRange <= 1e-6) return;
  const instantSpacing = phiRange / UI_CONTOUR_NUM_BANDS;
  // EMA-smooth the spacing across frames to eliminate per-frame jitter.
  // α=0.08 gives a 1/α ≈ 12-frame time constant (≈ 200ms at 60Hz). The
  // first frame after init / a long pause snaps to the instant value to
  // avoid a slow ramp-up from zero.
  if (_smoothedContourSpacing <= 1e-6) {
    _smoothedContourSpacing = instantSpacing;
  } else {
    _smoothedContourSpacing = 0.92 * _smoothedContourSpacing + 0.08 * instantSpacing;
  }
  const spacing = _smoothedContourSpacing;

  // Theme-aware contour color: light gray on dark bg, dark gray on light bg.
  const bgRgb = _colorToRgbNorm(state.bgColor);
  const bgIsLight = (bgRgb[0] + bgRgb[1] + bgRgb[2]) > 1.5;
  const cR = bgIsLight ? 0.16 : 0.86;
  const cG = bgIsLight ? 0.16 : 0.86;
  const cB = bgIsLight ? 0.20 : 0.90;
  const cA = bgIsLight ? 0.45 : 0.35;

  gl.useProgram(_progEquipotential.prog);
  gl.uniform2f(_progEquipotential.uViewport, _vpW, _vpH);
  // _fieldEntityData is preallocated to MAX_FIELD_ENTITIES * 4 floats. We
  // could pass a subarray of just `_fieldEntityCount * 4` here, but the GLSL
  // `if (i >= uEntityCount) break;` stops reads at uEntityCount, so the
  // tail bytes are inert. Uploading the full buffer is one less subarray
  // allocation per frame and the bandwidth (~2 KB) is negligible.
  gl.uniform4fv(_progEquipotential.uEntities, _fieldEntityData);
  gl.uniform1i(_progEquipotential.uEntityCount, _fieldEntityCount);
  gl.uniform1f(_progEquipotential.uEpsilon, state.epsilon);
  gl.uniform1f(_progEquipotential.uContourSpacing, spacing);
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
