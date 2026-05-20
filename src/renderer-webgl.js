// renderer-webgl.js — V9.0a WebGL 2 hot-path renderer.
//
// Replaces the Canvas2D background, trail-FBO, and entity-sprite rendering
// from renderer.js. UI lines (selection ring, drag preview, hover ghost,
// prediction line, absorbing fallback) remain on the overlay Canvas2D in
// V9.0a — they will migrate to WebGL in V9.0b.
//
// Layer order (back to front):
//   1. clear with state.bgColor
//   2. trail FBO blit (composited with straight-alpha blend)
//   3. entity sprite instanced draw (with 9-ghost wrap mirrors)
//   --- WebGL ends here; Canvas2D overlay layers above ---
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
//     ~16MB total VRAM upper bound. No explicit eviction in V9.0a (TODO V9.0b).
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

// Buffers
let _bufFsQuad = null;          // fullscreen quad NDC (vec2 in [-1,+1])
let _bufUnitQuad = null;        // unit quad centered (vec2 in [-1,+1]) for trail dots
let _bufEntityCornerQuad = null; // entity per-vertex (vec2 in [0,1])
let _bufInstanceTrail = null;   // dynamic per-instance VBO for trail dots
let _bufInstanceEntity = null;  // dynamic per-instance VBO for entity sprites

// VAOs (cached attrib+vbo state)
let _vaoFsQuad = null;
let _vaoTrailDot = null;
let _vaoEntity = null;

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

// ─── Shader source ────────────────────────────────────────────────

const VS_FULLSCREEN = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;            // [-1,1] → [0,1]
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FS_TRAIL_DECAY = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform float uDec;                  // alpha decrement, [0..1]
out vec4 outColor;
void main() {
  vec4 c = texture(uTex, vUv);
  float a = max(0.0, c.a - uDec);
  outColor = vec4(c.rgb, a);
}`;

const FS_TRAIL_BLIT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
out vec4 outColor;
void main() {
  outColor = texture(uTex, vUv);
}`;

// Per-instance attribs: iCenter (vec2, px), iColor (vec3, 0..1).
// Per-vertex: aCorner ∈ {-1,+1}² unit quad. Scaled by uRouter (= R + 0.5 px).
const VS_TRAIL_DOT = `#version 300 es
in vec2 aCorner;
in vec2 iCenter;
in vec3 iColor;
uniform mat4 uOrtho;
uniform float uRouter;               // outer radius in px (R + 0.5)
out vec2 vLocal;                     // offset from center, px
out vec3 vColor;
void main() {
  vec2 worldPx = iCenter + aCorner * uRouter;
  vLocal = aCorner * uRouter;
  vColor = iColor;
  gl_Position = uOrtho * vec4(worldPx, 0.0, 1.0);
}`;

const FS_TRAIL_DOT = `#version 300 es
precision highp float;
in vec2 vLocal;
in vec3 vColor;
uniform float uRinner;               // R - 0.5
uniform float uRouter;               // R + 0.5
out vec4 outColor;
void main() {
  float d = length(vLocal);
  if (d > uRouter) discard;
  float a = (d <= uRinner) ? 1.0 : (uRouter - d);
  outColor = vec4(vColor, a);
}`;

// Per-instance: iCenter (vec2 px), iSize (vec2 px = sprite w,h), iOffset
// (vec2 px = sprite ox, oy from top-left of sprite to entity center),
// iAlpha (float; always 1 in V9.0a).
// Per-vertex: aCorner ∈ {0,1}² unit quad (0=top-left, 1=bottom-right).
const VS_ENTITY = `#version 300 es
in vec2 aCorner;
in vec2 iCenter;
in vec2 iSize;
in vec2 iOffset;
in float iAlpha;
uniform mat4 uOrtho;
out vec2 vUv;
out float vAlpha;
void main() {
  vec2 spritePx = vec2(aCorner.x * iSize.x, aCorner.y * iSize.y);
  vec2 worldPx = iCenter - iOffset + spritePx;
  vUv = aCorner;
  vAlpha = iAlpha;
  gl_Position = uOrtho * vec4(worldPx, 0.0, 1.0);
}`;

const FS_ENTITY = `#version 300 es
precision highp float;
in vec2 vUv;
in float vAlpha;
uniform sampler2D uTex;
out vec4 outColor;
void main() {
  vec4 t = texture(uTex, vUv);
  outColor = vec4(t.rgb, t.a * vAlpha);
}`;

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
    _bufFsQuad = _bufUnitQuad = _bufEntityCornerQuad = null;
    _bufInstanceTrail = _bufInstanceEntity = null;
    _vaoFsQuad = _vaoTrailDot = _vaoEntity = null;
    _fboA = _fboB = _texA = _texB = null;
    try {
      _initPrograms();
      _initBuffers();
      _initFbos(_vpW, _vpH);
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
  _progTrailDecay = _makeProgram(VS_FULLSCREEN, FS_TRAIL_DECAY);
  _progTrailBlit  = _makeProgram(VS_FULLSCREEN, FS_TRAIL_BLIT);
  _progTrailDot   = _makeProgram(VS_TRAIL_DOT,  FS_TRAIL_DOT);
  _progEntity     = _makeProgram(VS_ENTITY,     FS_ENTITY);

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

  // Fullscreen quad VAO — shared by decay + blit passes. Both their shader
  // programs use VS_FULLSCREEN with `aPos` at location 0 (enforced via
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

// ─── Color parsing (matches V8.1c renderer.js) ────────────────────

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

  // Bucket entities by their sprite canvas. Skip absorbing entities (they
  // render on the overlay 2D canvas via renderer.js).
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
}
