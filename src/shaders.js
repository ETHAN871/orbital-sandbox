// shaders.js — GLSL ES 3.00 source strings for all WebGL 2 programs.
//
// Extracted from renderer-webgl.js post-V9.0b (per architect audit; the
// inline shader block was crowding 270+ lines into the renderer module and
// V9.1 will add at least 2 more programs). Keeping them as ES module
// string exports preserves the project's zero-build invariant — no fetch,
// no bundler, just `import { ... } from './shaders.js'`.
//
// Convention: each program exports a `{ VS, FS }` pair. Programs that
// share a vertex shader (the two fullscreen-quad programs — trail decay
// and trail blit — both use VS_FULLSCREEN) re-export the same string.
//
// Per-program block comments describe the per-instance attribute layout,
// the per-vertex `aCorner` convention, and any notable shader logic. Edits
// to a shader's attribute set MUST be reflected in the corresponding
// `_init*` and `_push*` sites in renderer-webgl.js (byte offsets, attrib
// pointer setup, scratch-array layout).

// ─── Shared vertex shader for fullscreen quads ────────────────────
// Used by both the trail-decay and trail-blit programs. aPos is a unit
// NDC quad in [-1, +1]²; vUv = aPos*0.5+0.5 yields [0, 1] for texture
// sampling.
const VS_FULLSCREEN = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;            // [-1,1] → [0,1]
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

// ─── Trail FBO: decay pass ────────────────────────────────────────
// Linear alpha decrement per frame: a_new = max(0, a_old - uDec).
// RGB carried through unchanged. Render target = the "write" trail FBO;
// source = the "read" FBO texture sampled at vUv.
export const TRAIL_DECAY = {
  VS: VS_FULLSCREEN,
  FS: `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform float uDec;                  // alpha decrement, [0..1]
out vec4 outColor;
void main() {
  vec4 c = texture(uTex, vUv);
  float a = max(0.0, c.a - uDec);
  outColor = vec4(c.rgb, a);
}`,
};

// ─── Trail FBO: blit pass ─────────────────────────────────────────
// Samples the post-decay+plot FBO texture and writes to the default
// framebuffer. Standard SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending lifts
// straight-alpha trail pixels over the cleared bg.
export const TRAIL_BLIT = {
  VS: VS_FULLSCREEN,
  FS: `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
out vec4 outColor;
void main() {
  outColor = texture(uTex, vUv);
}`,
};

// ─── Trail dot plot ────────────────────────────────────────────────
// Per-instance attribs: iCenter (vec2 px), iColor (vec3, 0..1).
// Per-vertex: aCorner ∈ {-1,+1}² centered unit quad. Scaled by uRouter
// (= R + 0.5 px). FS uses MAX blend equation (set up by caller) so
// successive dots leave the brighter alpha per channel.
export const TRAIL_DOT = {
  VS: `#version 300 es
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
}`,
  FS: `#version 300 es
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
}`,
};

// ─── Entity sprite (instanced quad with texture) ──────────────────
// Per-instance: iCenter (vec2 px), iSize (vec2 px = sprite w,h), iOffset
// (vec2 px = sprite ox, oy from top-left of sprite to entity center),
// iAlpha (float; always 1.0 — absorbing entities are filtered out
// upstream in _drawEntities and rendered via the filled-circle UI shader).
// Per-vertex: aCorner ∈ {0,1}² unit quad (0=top-left, 1=bottom-right).
export const ENTITY = {
  VS: `#version 300 es
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
}`,
  FS: `#version 300 es
precision highp float;
in vec2 vUv;
in float vAlpha;
uniform sampler2D uTex;
out vec4 outColor;
void main() {
  vec4 t = texture(uTex, vUv);
  outColor = vec4(t.rgb, t.a * vAlpha);
}`,
};

// ─── V9.0b UI shaders ─────────────────────────────────────────────
// Common pattern: per-instance attribs carry primitive params + color;
// per-vertex attrib aCorner is the unit quad in [-1,+1]² (filled circle,
// ring) or [0,1]² (line seg). VS scales the corner to the primitive's
// bounding extent in px and transforms via uOrtho; FS uses the per-pixel
// local coords (passed as a varying) to compute alpha and discard outside.

// --- Filled circle (hover ghost fill, drag ghost fill, absorbing body) ---
// Per-instance: iCenter(vec2 px), iRadius(float px), iColor(vec4).
// AA edge ramp = 1 px (same formula as the trail-dot shader above).
export const CIRCLE_FILL = {
  VS: `#version 300 es
in vec2 aCorner;
in vec2 iCenter;
in float iRadius;
in vec4 iColor;
uniform mat4 uOrtho;
out vec2 vLocal;
out vec4 vColor;
out float vRadius;
void main() {
  float pad = iRadius + 1.0;          // +1 px for AA ramp
  vec2 worldPx = iCenter + aCorner * pad;
  vLocal = aCorner * pad;
  vColor = iColor;
  vRadius = iRadius;
  gl_Position = uOrtho * vec4(worldPx, 0.0, 1.0);
}`,
  FS: `#version 300 es
precision highp float;
in vec2 vLocal;
in vec4 vColor;
in float vRadius;
out vec4 outColor;
void main() {
  float d = length(vLocal);
  if (d > vRadius + 0.5) discard;
  // Solid interior + 1-px AA ramp at the edge (alpha 1 → 0 across 1 px).
  float a = (d <= vRadius - 0.5) ? 1.0 : (vRadius + 0.5 - d);
  outColor = vec4(vColor.rgb, vColor.a * a);
}`,
};

// --- Ring stroke (hover outline, drag outline, handle, selection, BH edge) ---
// Per-instance: iCenter, iRadius, iColor, iLineW (px, full stroke width),
// iDashOn (px; 0 → solid), iDashPeriod (px; 0 → solid).
// FS draws |d - iRadius| < iLineW/2 with 1-px AA on both edges. If
// iDashPeriod > 0, masks by arc-length: arc = (atan2(y,x) wrapped to
// [0, 2π) by adding 2π if negative) × iRadius, range [0, 2π·R); visible
// if mod(arc, iDashPeriod) < iDashOn. The +2π wrap puts the dash seam
// at the +X axis (3 o'clock) to match V8.1c Canvas2D's `arc(c,c,r,0,2π)`
// + setLineDash start angle.
export const CIRCLE_RING = {
  VS: `#version 300 es
in vec2 aCorner;
in vec2 iCenter;
in float iRadius;
in vec4 iColor;
in float iLineW;
in float iDashOn;
in float iDashPeriod;
uniform mat4 uOrtho;
out vec2 vLocal;
out vec4 vColor;
out float vRadius;
out float vLineW;
out float vDashOn;
out float vDashPeriod;
void main() {
  float pad = iRadius + iLineW * 0.5 + 1.0;   // +1 px AA ramp
  vec2 worldPx = iCenter + aCorner * pad;
  vLocal = aCorner * pad;
  vColor = iColor;
  vRadius = iRadius;
  vLineW = iLineW;
  vDashOn = iDashOn;
  vDashPeriod = iDashPeriod;
  gl_Position = uOrtho * vec4(worldPx, 0.0, 1.0);
}`,
  FS: `#version 300 es
precision highp float;
#define PI 3.14159265359
in vec2 vLocal;
in vec4 vColor;
in float vRadius;
in float vLineW;
in float vDashOn;
in float vDashPeriod;
out vec4 outColor;
void main() {
  float d = length(vLocal);
  float half = vLineW * 0.5;
  float distToRing = abs(d - vRadius);
  if (distToRing > half + 0.5) discard;
  // 1-px AA ramp on each edge of the ring.
  float a = (distToRing <= half - 0.5) ? 1.0 : (half + 0.5 - distToRing);
  if (vDashPeriod > 0.0) {
    // Arc-length measured from the +X axis (3 o'clock), increasing in the
    // direction atan2 grows (clockwise in screen space since Y is flipped
    // by uOrtho). This matches V8.1c Canvas2D arc(cx, cy, r, 0, 2π) +
    // setLineDash, which began the dash sequence at angle=0. Without the
    // +2π wrap, atan2's (-π, π] range would put the seam at 9 o'clock.
    float ang = atan(vLocal.y, vLocal.x);
    if (ang < 0.0) ang += 2.0 * PI;
    float arc = ang * vRadius;
    float phase = mod(arc, vDashPeriod);
    if (phase >= vDashOn) discard;
  }
  outColor = vec4(vColor.rgb, vColor.a * a);
}`,
};

// --- Line segment (rubber band, prediction path) ---
// Each segment = one instance. Per-vertex aCorner ∈ {0,1}² where x is
// "along" (0=start, 1=end) and y is "perp" (0=−lineW/2, 1=+lineW/2).
// Per-instance: iP0, iP1 (vec2 px), iColor, iArcStart (px; cumulative
// arc-length for dash phase continuity across a polyline), iLineW (px),
// iDashOn, iDashPeriod (0 → solid).
export const LINE_SEG = {
  VS: `#version 300 es
in vec2 aCorner;
in vec2 iP0;
in vec2 iP1;
in vec4 iColor;
in float iArcStart;
in float iLineW;
in float iDashOn;
in float iDashPeriod;
uniform mat4 uOrtho;
out vec4 vColor;
out float vT;             // perp param (-half..+half px)
out float vArc;           // arc-length at this fragment, px
out float vLineW;
out float vDashOn;
out float vDashPeriod;
void main() {
  vec2 dir = iP1 - iP0;
  float segLen = length(dir);
  vec2 along = (segLen > 0.0) ? dir / segLen : vec2(1.0, 0.0);
  vec2 perp = vec2(-along.y, along.x);
  // Pad +0.5 px in the perpendicular direction so the FS AA ramp at the
  // long edges of the quad isn't clipped. The along-axis extent stays
  // exactly [iP0, iP1] — for a polyline that's correct because consecutive
  // segments share endpoints; standalone segments (rubber band) have no
  // visible end-caps but the dash mask hides truncation.
  float s = aCorner.x;
  float halfW = iLineW * 0.5 + 0.5;
  vec2 worldPx = iP0
               + along * (s * segLen)
               + perp  * ((aCorner.y - 0.5) * 2.0 * halfW);
  vT = (aCorner.y - 0.5) * 2.0 * halfW;
  vArc = iArcStart + s * segLen;
  vColor = iColor;
  vLineW = iLineW;
  vDashOn = iDashOn;
  vDashPeriod = iDashPeriod;
  gl_Position = uOrtho * vec4(worldPx, 0.0, 1.0);
}`,
  FS: `#version 300 es
precision highp float;
in vec4 vColor;
in float vT;
in float vArc;
in float vLineW;
in float vDashOn;
in float vDashPeriod;
out vec4 outColor;
void main() {
  float half = vLineW * 0.5;
  float absT = abs(vT);
  if (absT > half + 0.5) discard;
  float a = (absT <= half - 0.5) ? 1.0 : (half + 0.5 - absT);
  if (vDashPeriod > 0.0) {
    float phase = mod(vArc, vDashPeriod);
    if (phase >= vDashOn) discard;
  }
  outColor = vec4(vColor.rgb, vColor.a * a);
}`,
};
