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

// ─── V10 rubber-sheet oblique projection helper ───────────────────
// Inlined into the VS of every shader whose world-space vertices need
// to "sink into the gravity well" when rubber-sheet mode is active:
// TRAIL_DOT, ENTITY, CIRCLE_FILL, CIRCLE_RING, LINE_SEG, PARTICLE_FLOW.
//
// Sampling: uSagTex is a 256×256 R32F texture pre-baked by the CPU
// each frame with sag(world_x, world_y) in CSS-px. uSagMode gates the
// effect (0 = flat / passthrough, 1 = apply oblique). uSagViewport
// is the CSS-px viewport (W, H) used to map worldPx → uv.
//
// Projection: screen_y = world_y + sag * cos(viewTilt). Renderer
// supplies uSagYFactor = cos(state.viewTilt * π / 180), 0..1.
//   viewTilt = 45° → factor ≈ 0.7071 (classic oblique, V11 default)
//   viewTilt = 90° → factor = 0       (top-down: sag invisible, flat)
//   viewTilt = 30° → factor ≈ 0.866   (strong oblique)
// No global y-axis compression — flat regions render unchanged; only
// the gravity wells pull bodies + UI + trails downward visually.
// V11.3 (2026-05-28): mode-aware UV. uSagWrap selects between:
//   uSagWrap = 1 → fract(uv)  (wrap-boundary: 9-ghost pack in
//                              _packFieldEntities makes the texture
//                              toroidally correct, so fract sampling
//                              + gl.REPEAT on the texture gives
//                              continuous sag across world edges and
//                              the GRID_EXPAND_FRAC mesh skirt).
//   uSagWrap = 0 → clamp(uv)  (bounded: world truly ends at the
//                              viewport. CPU bake has no ghosts, so
//                              fract would mirror in-viewport content
//                              into the skirt → phantom wells. Clamp
//                              freezes the skirt at edge-row sag —
//                              today's behavior, no regression).
// The previous unconditional clamp killed wrap-aware sampling for
// every ghost copy (e.g. body mirror at worldPx.y = H+0.5) and for
// every mesh vertex outside the viewport in wrap mode — symptom: 1-3
// px leap as bodies cross the seam.
const SAG_VS_HELPER = `
uniform sampler2D uSagTex;
uniform float uSagMode;
uniform float uSagYFactor;
uniform float uSagWrap;
uniform vec2 uSagViewport;
vec2 sagProject(vec2 worldPx) {
  if (uSagMode < 0.5) return worldPx;
  vec2 raw = worldPx / uSagViewport;
  vec2 uv = (uSagWrap > 0.5) ? fract(raw) : clamp(raw, 0.0, 1.0);
  float sag = texture(uSagTex, uv).r;
  return vec2(worldPx.x, worldPx.y + sag * uSagYFactor);
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
${SAG_VS_HELPER}
void main() {
  vec2 worldPx = iCenter + aCorner * uRouter;
  vLocal = aCorner * uRouter;
  vColor = iColor;
  gl_Position = uOrtho * vec4(sagProject(worldPx), 0.0, 1.0);
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
${SAG_VS_HELPER}
void main() {
  vec2 spritePx = vec2(aCorner.x * iSize.x, aCorner.y * iSize.y);
  // Project the entity CENTER through sag, then add sprite offset so
  // the sprite shape stays unwarped (just shifted to the well bottom).
  vec2 worldPx = sagProject(iCenter) - iOffset + spritePx;
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
${SAG_VS_HELPER}
void main() {
  float pad = iRadius + 1.0;          // +1 px for AA ramp
  // Project center through sag; corner offset stays unwarped so the
  // circle remains a circle on screen, just shifted to the well bottom.
  vec2 worldPx = sagProject(iCenter) + aCorner * pad;
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
${SAG_VS_HELPER}
void main() {
  float pad = iRadius + iLineW * 0.5 + 1.0;   // +1 px AA ramp
  // Project center; corner offset stays unwarped (circular ring on screen).
  vec2 worldPx = sagProject(iCenter) + aCorner * pad;
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
  // NOTE: do NOT name this variable 'half' — that is a GLSL ES 3.00
  // reserved word (future-use for half-precision floats) and shaders
  // fail to compile. Likewise, no backticks in shader comments either —
  // they would terminate the surrounding JS template literal.
  float halfW = vLineW * 0.5;
  float distToRing = abs(d - vRadius);
  if (distToRing > halfW + 0.5) discard;
  // 1-px AA ramp on each edge of the ring.
  float a = (distToRing <= halfW - 0.5) ? 1.0 : (halfW + 0.5 - distToRing);
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
${SAG_VS_HELPER}
void main() {
  // Project both endpoints so polyline endpoints sit at projected positions
  // (e.g. prediction line follows the body into its well). Segment geometry
  // is then built from projected endpoints — line is straight on screen
  // between A_projected and B_projected.
  vec2 p0 = sagProject(iP0);
  vec2 p1 = sagProject(iP1);
  vec2 dir = p1 - p0;
  float segLen = length(dir);
  vec2 along = (segLen > 0.0) ? dir / segLen : vec2(1.0, 0.0);
  vec2 perp = vec2(-along.y, along.x);
  float s = aCorner.x;
  float halfW = iLineW * 0.5 + 0.5;
  vec2 worldPx = p0
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
  // 'half' is a GLSL ES 3.00 reserved word — use halfW. See FS_CIRCLE_RING above.
  float halfW = vLineW * 0.5;
  float absT = abs(vT);
  if (absT > halfW + 0.5) discard;
  float a = (absT <= halfW - 0.5) ? 1.0 : (halfW + 0.5 - absT);
  if (vDashPeriod > 0.0) {
    float phase = mod(vArc, vDashPeriod);
    if (phase >= vDashOn) discard;
  }
  outColor = vec4(vColor.rgb, vColor.a * a);
}`,
};

// ─── V9.1 field visualization shaders ─────────────────────────────
// Two programs, both gated by `state.showField` so they incur zero GPU
// cost when the field overlay is hidden.

// --- Equipotential contour lines (V9.x: log-spaced) ---------------
// Fragment shader sums φ = Σ_i -G·q·m / sqrt(r² + ε²) per pixel over up to
// MAX_ENTITIES uniform-array entries (pre-multiplied G·q·m packed into
// entity.z by the caller so the shader only carries one float multiply
// per entity per pixel).
//
// Contour lines are placed at |φ| = uContourThreshold · k^n for integer
// n ∈ [0, NUM_BANDS). k = exp(uLogK) is set per frame by the CPU from
// the scene's mass extremes so the lightest emitter still shows ≥3 outer
// rings and the heaviest gets the full ring set — regardless of how
// dominant any single mass is.
//
// Pixel-distance to nearest ring derives the line mask from a derivative
// of log|φ| against frame-space, scaled to CSS pixels by uDpr:
//   n = log(|φ|/threshold) / log(k)
//   dSteps = |mod(n + 0.5, 1) - 0.5|             # distance to nearest integer
//   pxPerStep = |φ| · log(k) / |∇φ|              # fragments per unit n
//   dPxCss = dSteps · pxPerStep / uDpr           # CSS-px from contour
// For a 1/r isolated body this gives pxPerStep ∝ r — rings naturally
// space out linearly with distance, matching topographic-map intuition
// (steeper gradient → tighter rings).
//
// Per-instance: none (one fullscreen quad).
// Per-vertex: aPos (NDC), shared with VS_FULLSCREEN.
// Uniforms:
//   uViewport         vec2  CSS-px viewport (W, H) for vUv → world conversion
//   uEntities         vec4[MAX_ENTITIES]  per-entity (x, y, G·q·m, _unused_)
//   uEntityCount      int   real count (≤ MAX_ENTITIES); loop early-exit
//   uEpsilon          float softening length for Plummer
//   uLogK             float ln(k); k = inter-ring |φ| ratio (per-frame)
//   uContourThreshold float |φ| at the outermost (n=0) ring (per-frame)
//   uContourLineW     float target line width in CSS-px (typically 1.0)
//   uDpr              float device pixel ratio for CSS-px line scaling
//   uColor            vec4  contour color (theme-dependent: light on
//                           dark bg, dark on light bg)
//
// MAX_ENTITIES = 128 sits well under the WebGL 2 minimum-spec
// MAX_FRAGMENT_UNIFORM_VECTORS = 224; with one vec4 per entity that's
// 128 uniform vectors. If a user spawns >128 entities the renderer will
// cap the array at 128 with no error (just visually approximate field).
export const EQUIPOTENTIAL = {
  VS: `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`,
  FS: `#version 300 es
precision highp float;
#define MAX_ENTITIES 128
in vec2 vUv;
uniform vec2 uViewport;
uniform vec4 uEntities[MAX_ENTITIES];
uniform int uEntityCount;
uniform float uEpsilon;
// V9.x — logarithmic contour spacing. Rings live at |φ| = threshold·k^n
// for integer n ∈ [0, NUM_BANDS). This gives small-mass entities a few
// outer rings (n near 0, low |φ|) and high-mass entities many inner
// rings (n near NUM_BANDS, high |φ|) regardless of the global mass
// ratio — so a small body next to a huge one still shows topographic
// detail. uLogK = ln(k); uContourThreshold is the |φ| at the outermost
// (n=0) ring. Both set by CPU each frame from the scene's mass extremes;
// see _drawEquipotential in renderer-webgl.js.
uniform float uLogK;
uniform float uContourThreshold;
uniform float uContourLineW;
uniform float uDpr;                  // device pixel ratio: 1 CSS-px = DPR fragments
uniform vec4 uColor;
out vec4 outColor;

float computePhi(vec2 p) {
  float phi = 0.0;
  // Plummer softening: safeR = sqrt(r² + ε²). Smooth and bounded at
  // r=0; gradient is well-defined everywhere → no contour-line kinks
  // at body centers. Matches physics.js / potential.js exactly so the
  // force the body feels and the field the viewer sees agree.
  float eps2 = uEpsilon * uEpsilon;
  for (int i = 0; i < MAX_ENTITIES; i++) {
    if (i >= uEntityCount) break;
    vec4 e = uEntities[i];
    float dx = p.x - e.x;
    float dy = p.y - e.y;
    float safeR = sqrt(dx * dx + dy * dy + eps2);
    phi += -e.z / safeR;
  }
  return phi;
}

void main() {
  // vUv comes from VS_FULLSCREEN's aPos*0.5+0.5 mapping. aPos is in NDC where
  // the default framebuffer has y=+1 at the top. CSS-px entity coords have
  // y=0 at the top instead. Flip vUv.y so p below is in the same CSS-px
  // space the entity positions were uploaded in. (No backticks in this
  // comment — they would terminate the surrounding JS template literal.)
  vec2 p = vec2(vUv.x * uViewport.x, (1.0 - vUv.y) * uViewport.y);
  float phi = computePhi(p);
  float absPhi = abs(phi);
  // Skip far-field (below outermost ring) and degenerate (no field at all).
  if (absPhi < uContourThreshold) discard;
  if (uLogK <= 0.0) discard;
  // Log-spaced ring index: n=0 at outermost ring (|φ|=threshold), grows
  // toward inner (deep-well) rings. dSteps = distance in step-units to
  // nearest integer ring. abs(mod(...) - 0.5) gives a triangle wave in
  // [0, 0.5] peaking at half-step (midway between rings).
  float n = log(absPhi / uContourThreshold) / uLogK;
  float halfStep = 0.5;
  float dSteps = abs(mod(n + halfStep, 1.0) - halfStep);
  // Gradient magnitude in fragments (dFdx/dFdy step = 1 fragment).
  float gx = dFdx(phi);
  float gy = dFdy(phi);
  float gradMag = sqrt(gx * gx + gy * gy);
  if (gradMag < 1e-6) discard;
  // Pixel-width of one full log-step: derived from d/dr[ln|φ|] = |∇φ|/|φ|,
  // so one step (Δlog|φ| = logK) spans pxPerStep = |φ|·logK/|∇φ| fragments.
  // For a single 1/r body this gives pxPerStep ∝ r — far-field rings are
  // naturally sparse in screen space, near-body rings tighter. The
  // logarithmic mapping is what makes light bodies still get visible
  // outer rings near their weak |φ| neighbourhood.
  float pxPerStep = absPhi * uLogK / gradMag;
  float dPxFrag = dSteps * pxPerStep;
  float dPxCss  = dPxFrag / max(uDpr, 1.0);
  float halfW = uContourLineW * 0.5;
  if (dPxCss > halfW + 0.5) discard;
  float alpha = (dPxCss <= halfW - 0.5) ? 1.0 : (halfW + 0.5 - dPxCss);
  outColor = vec4(uColor.rgb, uColor.a * alpha);
}`,
};

// --- Pulsing streamlines ("灯带" light strips) --------------------
// Each streamline is one short straight segment in the direction of the
// local force at its grid-seed origin. The CPU computes the seed force
// direction once per frame (via computeForceDirAt in potential.js) and
// pushes per-instance (seedX, seedY, dirX, dirY). The shader animates a
// "head" of bright color sliding from along=0 to along=1 over a shared
// uniform `uPulseHead ∈ [0..1]`, with a trailing fade of length
// `uPulseTailFrac` so the strip looks like a comet of light.
//
// Per-vertex aCorner ∈ {0,1}²: x = along (0=seed, 1=tip), y = perp.
// Per-instance: iSeed (vec2 px), iDir (vec2 unit), iAlpha (float).
// Uniforms:
//   uOrtho           mat4   CSS-px → NDC matrix (shared with other UI)
//   uLength          float  streamline length in px (~50)
//   uLineW           float  half-width × 2 in px (~3)
//   uPulseHead       float  head position [0..1] along the strip
//   uPulseTailFrac   float  trail length as fraction of strip (~0.25)
//   uColor           vec4   light-strip base color (cool blue-white)
//
// iAlpha modulates per-instance brightness — used to fade low-magnitude
// field zones so the field intensity is also visually encoded.
export const STREAMLINE = {
  VS: `#version 300 es
in vec2 aCorner;
in vec2 iSeed;
in vec2 iDir;
in float iAlpha;
uniform mat4 uOrtho;
uniform float uLength;
uniform float uLineW;
out vec2 vAlongPerp;
out float vAlpha;
void main() {
  vec2 along = iDir;
  vec2 perp = vec2(-along.y, along.x);
  float s = aCorner.x;               // 0..1 along the strip
  float halfW = uLineW * 0.5 + 0.5;  // +0.5 px for AA ramp
  float t = (aCorner.y - 0.5) * 2.0 * halfW;
  vec2 worldPx = iSeed + along * (s * uLength) + perp * t;
  vAlongPerp = vec2(s, t);
  vAlpha = iAlpha;
  gl_Position = uOrtho * vec4(worldPx, 0.0, 1.0);
}`,
  FS: `#version 300 es
precision highp float;
in vec2 vAlongPerp;
in float vAlpha;
uniform float uLineW;
uniform float uPulseHead;
uniform float uPulseTailFrac;
uniform vec4 uColor;
out vec4 outColor;
void main() {
  float along = vAlongPerp.x;      // 0..1 from seed to tip
  float perp = vAlongPerp.y;
  float halfW = uLineW * 0.5;
  float absT = abs(perp);
  if (absT > halfW + 0.5) discard;
  float aaPerp = (absT <= halfW - 0.5) ? 1.0 : (halfW + 0.5 - absT);
  // Light strip occupies (uPulseHead - tailFrac) .. uPulseHead along [0..1].
  // Clamp tailStart to 0 so during the first uPulseTailFrac of the cycle the
  // strip emerges *growing* from the seed point rather than rendering its
  // full length immediately (which would look like a glitch flash on every
  // pulse-cycle reset).
  float tailStart = max(0.0, uPulseHead - uPulseTailFrac);
  if (along > uPulseHead) discard;
  if (along < tailStart) discard;
  // Brightness ramps from 0 at tailStart to 1 at uPulseHead (head bright).
  // Use the *actual* strip length (uPulseHead - tailStart) so the gradient
  // remains correctly normalized even when the start was clamped to 0.
  float stripLen = max(uPulseHead - tailStart, 1e-6);
  float aTrail = (along - tailStart) / stripLen;
  outColor = vec4(uColor.rgb, uColor.a * aaPerp * aTrail * vAlpha);
}`,
};

// --- Grid warp (spacetime fabric) --------------------------------------
// New field viz (2026-05-26 rewrite). Each frame draws a regular grid of
// vertices. Per-vertex the GPU computes the gravitational potential
// contribution from up to MAX_ENTITIES masses and either:
//
//   - 3D mode (uMode == 0): vertex Z = scale × Σ -G·q·m / sqrt(r² + ε²),
//     projected to 2D via oblique projection (x, y + Z·tiltY). Looks
//     like the classic rubber-sheet / Interstellar gravity-well drawing.
//
//   - 2D mode (uMode == 1): vertex displaced in the XY plane along
//     the local force direction ∇φ by an amount proportional to |∇φ|.
//     Grid lines bend toward masses without leaving the plane — flat
//     and clean, matches the 2D top-down camera.
//
// Cost: vertex-shader work only, no fragment-shader N-body sum. With a
// 80×60 grid that's 4800 vertices × (≤ 128 masses × 5 ops) ≈ 3 M vertex
// ops, but vertex throughput is 30-50× higher than fragment throughput
// on most GPUs, and we replaced 630 k fragments × 128 masses (~80 M ops)
// from the old EQUIPOTENTIAL fullscreen pass. Net ~20-30× speedup.
//
// uColor.a is modulated per-fragment by a "intensity" estimate: edges
// in high-warp regions are slightly more opaque, far-field edges fade.
// This is what gives the visual rhythm of "deeper wells = denser ink".
export const GRID_WARP = {
  VS: `#version 300 es
#define MAX_ENTITIES 128
in vec2 aPos;          // grid vertex in CSS px (untransformed)
uniform vec2 uViewport;
uniform vec4 uEntities[MAX_ENTITIES];   // (x, y, G·q·m, 0)
uniform int uEntityCount;
uniform float uEpsilon;
uniform float uDispScale;     // amplitude of warp (px)
uniform float uTiltY;         // 3D: how much Z projects into screen-Y (0..1)
uniform int uMode;            // 0 = 3D oblique (per-vertex φ), 1 = 2D in-plane, 2 = rubber-sheet (sag texture)
uniform mat4 uOrtho;          // CSS-px → NDC
uniform float uCellPx;        // grid cell spacing — caps displacement so adjacent vertices can't swap
out float vIntensity;         // displacement magnitude — drives brightness in FS
${SAG_VS_HELPER}
void main() {
  // V10 rubber-sheet branch — uMode == 2.
  // Reads the same per-frame normalized sag texture used by bodies,
  // trails, UI, and the prediction line. Per-vertex φ calculation
  // in the legacy uMode == 0 path saturates the depth cap (φ values
  // are O(hundreds), depth = -φ × dispScale hits the 250-px clamp
  // for every vertex within a body's footprint → mesh looks flat
  // regardless of how many bodies are placed). Sharing the sag
  // texture path lets the mesh, bodies, and UI all dip into wells
  // together with a single normalization pass owned by the CPU.
  if (uMode == 2) {
    vec2 wp = sagProject(aPos);
    // intensity drives the FS height-shading. For rubber-sheet mode
    // we want the FS ramp to expose the same depth signal the user
    // sees in the projection, so feed it the screen-Y delta (= sag).
    vIntensity = max(0.0, wp.y - aPos.y);
    gl_Position = uOrtho * vec4(wp, 0.0, 1.0);
    return;
  }
  float eps2 = uEpsilon * uEpsilon;
  vec2 p = aPos;
  // (V9.4) No alpha edge-fade — the prior viewport-distance smoothstep
  // read as "edge darkening" instead of an infinite-plane illusion.
  // The grid still extends past the viewport (see
  // _rebuildGridWarpVerts) so the unwarped grid corners stay off
  // screen; brightness is driven purely by displacement magnitude
  // (vIntensity) in the fragment shader — flat regions full bright,
  // deep wells dim (mimics ambient overhead lighting on a rubber
  // sheet sagging into a gravity well).
  float phi = 0.0;             // Σ φ for 3D mode
  // 2D mode: build displacement with PER-BODY anti-overshoot bound.
  // Each entity's pull on a vertex is capped at OVERSHOOT_FRAC × r_i
  // where r_i is THIS vertex's distance to THIS body. A single body
  // can therefore never move a vertex more than OVERSHOOT_FRAC × r_i
  // toward itself → no crossing through body centres → no triangular
  // rays at body positions and no adjacent-vertex order swaps (folds).
  // OVERSHOOT_FRAC = 0.4: 40% of the way per body. With multi-body
  // sums this can briefly exceed any single r_i, hence the final
  // global cap below.
  vec2 disp2D = vec2(0.0);
  for (int i = 0; i < MAX_ENTITIES; i++) {
    if (i >= uEntityCount) break;
    vec4 e = uEntities[i];
    vec2 d = p - vec2(e.x, e.y);   // points AWAY from body
    float r2 = d.x * d.x + d.y * d.y + eps2;
    float invR = inversesqrt(r2);
    if (uMode == 0) {
      phi += -e.z * invR;
    } else {
      // raw signed magnitude (m/r² scaled by uDispScale).
      float r = 1.0 / invR;
      float raw = e.z * uDispScale * invR * invR;
      // Anti-overshoot on magnitude only (preserves direction +
      // attractor/repulsor sign).
      float OVERSHOOT_FRAC = 0.4;
      float maxAllowed = r * OVERSHOOT_FRAC;
      float bounded = sign(raw) * min(abs(raw), maxAllowed);
      // Apply along the unit vector TOWARD the body (= -d / r).
      disp2D += -d * invR * bounded;
    }
  }
  vec2 worldPx;
  float intensity;
  // Anti-fold cap: max displacement magnitude per vertex.
  // For two adjacent vertices spacing = cellPx, if both can be
  // displaced by up to D, worst case (opposite directions) → final
  // separation = cellPx - 2D. For no swap (no fold), need 2D <
  // cellPx → D < cellPx/2. Use 0.45 for safety margin (diagonal
  // adjacency, atan smoothing variance). Without this cap, dense
  // clusters produce visible grid folding / "wireframe sphere"
  // illusion as adjacent vertices cross each other.
  float maxDisp = uCellPx * 0.45;
  if (uMode == 0) {
    // 3D oblique: phi is negative for attractors → -phi is positive
    // depth. Project Z into Y as a "tilt down" — depressions appear
    // below the body, matching the rubber-sheet mental model.
    float depth = clamp(-phi * uDispScale, 0.0, 250.0);
    depth = min(depth, maxDisp);
    worldPx = vec2(p.x, p.y + depth * uTiltY);
    intensity = depth;
  } else {
    // 2D in-plane: atan-style soft saturation as backstop for the
    // already-bounded per-body sum (multi-body pile-ups).
    //
    // sat(m) = L * atan(m / L)
    //   • slope at m=0 is 1 → small sums pass through 1:1 (single
    //     body still warps exactly as bounded above, no premature
    //     compression).
    //   • as m→∞, sat → L·π/2 ≈ 78.5 px with L=50 — but then the
    //     anti-fold cap below clamps to ≤ 0.45·cellPx, so the
    //     effective max scales with grid density. Big clusters
    //     produce the deepest visible warp the grid topology can
    //     hold without lines crossing.
    //   • approach to the limit is smooth (like atan→π/2) instead
    //     of the 1/x falloff of the rational form.
    float mag = length(disp2D);
    float L = 50.0;
    float satMag = L * atan(mag / L);
    satMag = min(satMag, maxDisp);
    float factor = (mag > 0.0001) ? satMag / mag : 1.0;
    worldPx = p + disp2D * factor;
    intensity = satMag;
  }
  vIntensity = intensity;
  gl_Position = uOrtho * vec4(worldPx, 0.0, 1.0);
}`,
  FS: `#version 300 es
precision highp float;
in float vIntensity;
uniform vec4 uColor;
uniform float uIntensityMin;
uniform float uIntensityMax;
uniform float uContrastFloor;  // dimmest brightness for the height ramp
out vec4 outColor;
void main() {
  // Relative height-based shading (V9.6): smoothstep maps each
  // vertex's intensity from the per-FRAME [min, max] range into
  // [0, 1] then into brightness [1.0 .. 0.25]. Crucially uses
  // MIN (not 0) as the lower bound: in scenes with many bodies
  // every vertex has nonzero baseline warp, so the "flattest"
  // point is at intensity = min > 0. Mapping from 0 instead of
  // min collapses the dynamic range and the whole scene looks
  // uniformly dim. With min anchoring, the flattest visible spot
  // always renders fully bright relative to the deepest well —
  // contrast between bodies remains visible even in crowded scenes.
  // Renderer guarantees max - min >= 0.5 so smoothstep degenerate
  // case is avoided.
  // brightness: 1.0 at flat regions (t=0), uContrastFloor at deep
  // wells (t=1). Renderer maps the slider value: dim = 1 - contrast,
  // so contrast=0 → dim=1.0 (flat shading, no contrast), contrast=1
  // → dim=0.0 (deep wells go fully dark).
  float t = smoothstep(uIntensityMin, uIntensityMax, vIntensity);
  float brightness = mix(1.0, uContrastFloor, t);
  outColor = vec4(uColor.rgb * brightness, uColor.a);
}`,
};

// ─── Membrane field — relief-lit grayscale gravity sheet ──────────
// Full-screen FS that renders the gravity field as the 2D top-down
// projection of a rubber-sheet membrane (à la the BabylonJS spacetime
// sandbox), lit by a 45°-elevation point light. Replaces the earlier
// lens warp, which folded into caustic "flowers" (its UV remap crossed
// the single-valued threshold near clusters). Three cues, all centered
// on each body's 2D position (no oblique offset):
//
//   1. Height field h(x,y) = Σ |w_i|·core²/(r_i²+core²): the sag depth,
//      ∝ |G·q·m|, fast 1/r² falloff scaled to the canvas (uCore2). Its
//      analytic gradient ∇h is summed in the same loop.
//   2. Relief lighting — surface normal N = normalize(∇h·uSlope, 1) of
//      the dipping sheet (z = −h); matte Lambert diffuse against a 45°
//      upper-left point light → grayscale. This is the PRIMARY depth
//      cue (hillshade): light-facing slope bright, far slope shaded,
//      flat field uniform mid-gray. uAmbient (from the contrast slider)
//      sets shadow depth.
//   3. Grid pinch — lines converge INTO the wells (funnel projected to
//      2D). Per-body pull is clamped (PMAX) and gain kept modest so the
//      map stays single-valued: NO caustic folds, far field stays a
//      straight grid.
//
// The whole membrane is output at alpha = uOpacity (the 膜透明度 slider)
// so it composites as a semi-transparent sheet over the scene.
export const SCREEN_DENT = {
  VS: VS_FULLSCREEN,                   // vUv ∈ [0,1]² covers the viewport
  FS: `#version 300 es
precision highp float;
#define MAX_ENTITIES 128
in vec2 vUv;
uniform vec2 uViewport;
uniform vec4 uEntities[MAX_ENTITIES];   // (x, y, G·q·m, radius)
uniform int uEntityCount;
uniform float uHeightK;              // sag amplitude (= core²/maxWeight, normalized)
uniform float uCore2;                // softening² (px²): well core size
uniform float uWarpGain;             // grid-pinch gain (kept below fold)
uniform float uSlope;                // ∇h→normal scale (relief strength)
uniform float uAmbient;              // base z-ambient relief floor (fixed)
uniform float uContrast;             // 45° highlight/shadow group strength (slider)
uniform float uWrap;                 // 1 = wrap: use minimum-image (toroidal, wrap-invariant)
uniform vec2 uCell;                  // grid spacing per-axis (px); toroidal in wrap mode
uniform vec4 uColor;                 // membrane base tint (rgb)
uniform float uOpacity;              // whole-membrane alpha (0..1)
out vec4 outColor;

const float WARP_DEPTH = 3.0;        // inextensible pinch saturation (× core) — deep but fold-free
const float TAU = 6.28318530718;     // 2π — continuous periodic warp (anti wrap seam-tear)
const float LOD_DB = 0.5;            // octave deadband — suppress mid-field LOD flicker (断线)
const float REFINE_THRESHOLD = 1.0;  // height below this → no heavy-body refine bias
const float REFINE_GAIN = 0.9;       // height above threshold → finer octaves (bias)
const float MAX_BIAS_OCT = 1.5;      // cap on heavy-body refine bias (≤ ~2.8× base density)
const float FILL_SHADE = 0.2;        // membrane fill shading weight (0 = no color block; lines carry it)
const float LINE_DARK_MIN = 0.3;     // line darkening in bright regions (light lines)
const float LINE_DARK_MAX = 0.85;    // line darkening cap in deep regions (clamp — never solid black)
const float AO_STRENGTH = 1.3;       // depth→ambient-occlusion rate (deeper well = darker)
const float AO_FLOOR = 0.12;         // min ambient at great depth (indirect bounce, not black)

// Analytically box-filtered grid (Inigo Quilez, "filterable procedurals").
// Instead of POINT-sampling the line pattern and smoothing by the derivative
// (which ALIASES — moiré — once the warped grid is finer than the footprint,
// e.g. the high-curvature pinch near a deep well, made worse at half-res), it
// INTEGRATES the line indicator analytically over the pixel footprint w = the
// explicit derivative. An over-dense grid then converges to its average tone
// (≈2·LW) — a smooth grey — instead of aliasing into false closed loops.
// Returns line coverage: 1 on a grid line, 0 in the cell interior. LW = line
// width as a fraction of the cell. Derivative passed explicitly (continuous
// base coord × octave scale): fwidth() of the scaled coord spikes at LOD seams.
float gridAt(vec2 uvw, vec2 d) {
  const float LW = 0.06;
  const float WIDEN = 2.0;                        // footprint safety factor: fwidth
  // underestimates the true screen pixel span (linear est. in curved regions +
  // half-res FBO upscaled 2× to screen) → widen so the box filter never
  // under-integrates and aliases. Over-filtering just fades to grey (clean).
  vec2 w = max(d * WIDEN, vec2(1e-6));            // footprint, cell units / fragment
  vec2 a = uvw + 0.5 * w;
  vec2 b = uvw - 0.5 * w;
  // ∫ of the line indicator (ink on the first LW of each cell) over [b, a]:
  // box-averaged coverage per axis → LW (the mean) when finer than the footprint.
  vec2 cov = (floor(a) * LW + min(fract(a), vec2(LW))
            - floor(b) * LW - min(fract(b), vec2(LW))) / w;
  return 1.0 - (1.0 - cov.x) * (1.0 - cov.y);     // union: on an x-line OR a y-line
}

void main() {
  // CSS-px frag position (y down from top) — matches uEntities coords.
  vec2 p = vec2(vUv.x * uViewport.x, (1.0 - vUv.y) * uViewport.y);
  vec2 grad = vec2(0.0);             // ∇h (height-field gradient)
  vec2 warp = vec2(0.0);             // grid-pinch displacement (inextensible-style)
  float h = 0.0;                     // absolute height field (∝ mass), drives refinement
  float bodyMask = 0.0;              // 1 where a body covers this fragment (hole)
  for (int i = 0; i < MAX_ENTITIES; i++) {
    if (i >= uEntityCount) break;
    vec4 e = uEntities[i];
    vec2 di = p - e.xy;              // points AWAY from the body
    // Wrap: minimum-image — each fragment sees the body's NEAREST image. This
    // makes the field toroidally exact AND wrap-invariant: a body at y vs y+H
    // yields the identical field, so crossing the seam no longer shifts the
    // grid (the 9-ghost packing was asymmetric → grid phase jumped on wrap).
    if (uWrap > 0.5) di -= uViewport * floor(di / uViewport + 0.5);
    float r2 = dot(di, di);
    float inv = 1.0 / (r2 + uCore2);
    float w = abs(e.z);             // field strength ∝ |G·q·m|·embed
    // ∂/∂p [ |w|·core²/(r²+core²) ] = |w|·core²·(-2·di)/(r²+core²)².
    grad += (w * uHeightK) * (-2.0) * di * (inv * inv);
    h    += (w * uHeightK) * inv;   // absolute height (h=1 at a REF_MASS body's center)
    // INEXTENSIBLE-STYLE inward pull. Magnitude |d| = D·(1 − c²/(r²+c²)): 0 at
    // the body, rising MONOTONICALLY (dd/dr ≥ 0 ⇒ the radial map r→r+d has slope
    // ≥ 1, NEVER inverts ⇒ no fold per body) and saturating at D far out. The
    // grid gets DENSER through the well wall (deep pinch, bottom not flat) yet
    // can't fold — the property di·f lacked (its |d|=r·f decreases outward ⇒
    // slope < 0 ⇒ fold). D saturates with mass (a heavy body dents only so far),
    // so summing many monotone inward pulls stays fold-resistant.
    float pull0 = uWarpGain * w / uCore2;                 // core pull (mass strength)
    float D = WARP_DEPTH * sqrt(uCore2) * pull0 / (1.0 + pull0);
    float dmag = D * (1.0 - uCore2 * inv);                // 0 → D, monotone
    if (uWrap > 0.5) {
      vec2 t = 0.5 + 0.5 * cos(TAU * di / uViewport);     // seam taper (C¹)
      dmag *= t.x * t.y;
    }
    warp += (di / max(sqrt(r2), 1e-3)) * dmag;            // unit-radial × magnitude
    // Hole: drop the membrane where a body sprite covers it (e.w = radius).
    // Relaxing wells (a removed body springing back) pack radius 0 → no hole.
    if (e.w > 0.5) bodyMask = max(bodyMask, 1.0 - smoothstep(e.w * 0.8, e.w * 1.1, sqrt(r2)));
  }
  // Two DIFFUSE lights superimposed (no specular): a +z ambient and a 45°
  // upper-left light. The 45° light's weight is the contrast slider;
  // half-Lambert keeps its terminator soft and its shadow shallow.
  vec3 N = normalize(vec3(grad * uSlope, 1.0));
  const float C = 0.70710678;
  vec3 L = normalize(vec3(-C * C, C * C, C));        // 45° elev, az 135°
  float dir = dot(N, L) * 0.5 + 0.5;                 // 45° directional (half-Lambert)
  float k = clamp(uContrast, 0.0, 1.0);
  float baseLit = (N.z + k * dir) / (1.0 + k);       // z-ambient slope + 45° sun fill
  // Depth ambient occlusion: a point deeper in a well sees less of the sky /
  // sun (cavity occlusion) → darker. Rational sky-visibility falloff with a
  // floor (indirect bounce keeps it off pure black). Unified with the N.z
  // slope term: slope = which way the surface faces, AO = how buried it is —
  // both attenuate the incoming light, so AO multiplies the lit result.
  float ao = AO_FLOOR + (1.0 - AO_FLOOR) / (1.0 + h * AO_STRENGTH);
  float gray = mix(uAmbient, 1.0, baseLit) * ao;
  // Adaptive refinement by DYADIC SUBDIVISION: keep the base grid fixed and
  // fade in midpoint lines (×2, ×4 spacing) where the field deepens — i.e.
  // a heavy body splits existing cells rather than reflowing the whole grid.
  // refine ∝ height field h (heavier/closer body → more levels), continuous
  // so adding a body fades the finer lines in smoothly (no "refresh").
  // Screen-adaptive grid LOD. Keep the on-screen line density ~constant: the
  // warp-compressed well auto-COARSENS ("expands outward") instead of mushing,
  // while stretched areas subdivide. lodC = octaves of compression (warped vs
  // flat coord derivatives); a heavy body biases the level finer (bounded) so
  // masses still get extra detail without over-densifying.
  vec2 uvW = (p + warp) / uCell;              // warped (drawn) grid coord
  vec2 uvF = p / uCell;                        // flat reference (base density)
  vec2 dW = fwidth(uvW);                       // continuous base derivatives
  vec2 dF = fwidth(uvF);
  float fwW = max(max(dW.x, dW.y), 1e-6);
  float fwF = max(max(dF.x, dF.y), 1e-6);
  float lodC = log2(fwW / fwF);                // ≥0 where the warp compresses
  float bias = clamp((h - REFINE_THRESHOLD) * REFINE_GAIN, 0.0, MAX_BIAS_OCT);
  float lam = lodC - bias;                     // net octave (− = finer than base)
  // Soft deadband: fwidth-derived lodC is per-quad NOISY; in the flat mid-field
  // it jitters across octave boundaries → spurious half-faded subdivision lines
  // that read as short broken ticks / tearing (断线). Shrink lam toward 0 so
  // |lam| < LOD_DB renders the pure base grid (no flicker), while strong LOD
  // near wells (|lam| > LOD_DB) still engages.
  lam = lam - clamp(lam, -LOD_DB, LOD_DB);
  float n0 = floor(lam);
  float fr = lam - n0;
  // Blend the two bracketing octaves (smooth coarsen/refine, no pop). Pass the
  // derivative EXPLICITLY (continuous dW × octave scale) — never fwidth() of
  // the scaled coord, which spikes at seams and prints stray line fragments.
  float s0 = exp2(-n0);
  float s1 = exp2(-(n0 + 1.0));
  float line = mix(gridAt(uvW * s0, dW * s0), gridAt(uvW * s1, dW * s1), fr);
  // Region brightness is carried by the LINES (darkness + count), not a filled
  // color block: the fill is near-uniform (FILL_SHADE≈0); each line darkens
  // where the lighting gray is dark, and the depth bias also packs MORE
  // lines there (denser = darker). On-screen density is constant under the LOD
  // except for the bias, so line count meaningfully encodes depth here. The
  // per-line darkening is clamped (LINE_DARK_MAX) so deep lines never go solid.
  float fill = mix(1.0, gray, FILL_SHADE);
  float lineDark = clamp(mix(LINE_DARK_MIN, LINE_DARK_MAX, 1.0 - gray), 0.0, LINE_DARK_MAX);
  vec3 rgb = uColor.rgb * fill * (1.0 - line * lineDark);
  // Carve the hole: membrane goes transparent where a body covers it.
  outColor = vec4(rgb, uOpacity * (1.0 - bodyMask));
}`,
};

// ─── V12 rubber-sheet — full-screen fragment-shader pass ──────────
// User pivoted away from the indexed-LINE mesh approach (V11) because
// the 15% viewport-expansion "skirt" remained a finite-extent band-aid:
// under heavy mass / many bodies the mesh-skirt stretched enough that
// the viewport edge re-emerged. After 4 parallel research agents + 3
// architect/critic rounds, the chosen path is PATH B: a single full-
// screen quad whose fragment shader computes world coords per pixel
// via single-step forward-warp:
//   screenPx = vUv * viewport
//   sag      = sample(uSagTex, screenPx / viewport)
//   worldPx  = (screenPx.x, screenPx.y - sag · yFactor)
// and draws grid lines in world space via Made-by-Evan's fwidth-AA
// pattern. Boundary-free by construction (no mesh = no skirt = no
// finite extent). In wrap mode the sag texture is toroidal (9-ghost
// pack in _packFieldEntities), so fract(uv) sampling at any screen
// position gives a valid sag value. In bounded mode clamp at the
// viewport edge — sag freezes at edge-row value, no visible artifact.
//
// Visual gestalt trade-off (R1's known issue): lines bend AROUND
// wells rather than diving INTO them. The forward-warp is single-
// valued by construction; an inverse warp would be multi-valued at
// deep wells (folding) and any inverse-style approach (Newton, LUT,
// hybrid) flickers at the fold rim. User explicitly accepted this
// trade-off after seeing the option list. Lines drape over the rim
// of wells like a true rubber sheet would — physically defensible.
//
// Depth fade: dimmer where sag is deeper, used as a soft proxy for
// painter-style occlusion. uContrastFloor controls how dark the
// deepest wells render (slider semantic: 0 = no dimming, 1 = max).
//
// Used ONLY for state.fieldStyle === 'rubber-sheet'. Modes '2d' and
// '3d' continue to use the mesh-based GRID_WARP program unchanged.
export const RUBBER_SHEET_FS = {
  VS: VS_FULLSCREEN,                   // vUv ∈ [0,1]² covers the viewport
  FS: `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSagTex;             // 256² R32F, REPEAT-wrap
uniform vec2  uViewport;               // (vpW, vpH) in CSS-px
uniform float uSagYFactor;             // cos(viewTilt) ∈ [0, 0.866]
uniform float uSagWrap;                // 1.0 wrap (fract uv), 0.0 bounded (clamp uv)
uniform float uCellPx;                 // grid cell spacing in world-space CSS-px
uniform vec4  uColor;                  // RGBA grid color
uniform float uContrastFloor;          // 1 - state.fieldContrast, depth-fade floor
uniform float uMaxSag;                 // max(_sagPixels) (raw, pre-yFactor) — for normalization
out vec4 outColor;

vec2 sagUV(vec2 worldPx) {
  vec2 raw = worldPx / uViewport;
  // +1.0 before fract guards against negative raw (e.g. world_y < 0
  // in the inverse-warp output below). fract is C0-continuous because
  // the sag texture is REPEAT-wrap with toroidal 9-ghost content in
  // wrap mode. In bounded mode just clamp to [0, 1] — no ghosts,
  // edge-row sag is the only physically meaningful boundary value.
  return (uSagWrap > 0.5) ? fract(raw + 1.0) : clamp(raw, 0.0, 1.0);
}

float sagAt(vec2 worldPx) {
  return texture(uSagTex, sagUV(worldPx)).r;
}

void main() {
  // Each fragment's "screen" position in CSS-px (the canvas pixel it
  // covers). vUv is [0,1] over the viewport, so vUv*viewport = px.
  vec2 screenPx = vUv * uViewport;

  // Single-step forward-warp inverse. The forward projection is
  //   screen_y = world_y + sag(world_x, world_y) · yFactor
  // The single-step approximation samples sag at the screen position
  // (not the true world position) — sub-pixel accurate when sag is
  // shallow, slightly off near deep-well centers. The visual gestalt
  // (lines bend around wells) is the documented trade-off. Choosing
  // this over Newton iteration because Newton diverges where sag is
  // multi-valued (fold-over near deep wells), which is the exact
  // visual region we want to render correctly.
  float sag = sagAt(screenPx);
  vec2 worldPx = vec2(screenPx.x, screenPx.y - sag * uSagYFactor);

  // Grid line coverage via Made-by-Evan / Inigo Quilez fwidth-AA.
  //   uvW = worldPx / cellPx → integer values at grid lines
  //   abs(fract(uvW - 0.5) - 0.5) is sawtooth peaking at 0.5 mid-cell,
  //   zero at grid lines (uvW = integer).
  //   Dividing by fwidth gives screen-px distance to nearest line.
  //   1 - clamp(min(d.x, d.y), 0, 1) is line coverage: 1 at line,
  //   linearly falling to 0 within one fragment-quad of derivative.
  vec2 uvW = worldPx / uCellPx;
  vec2 fw = max(fwidth(uvW), vec2(1e-6));
  vec2 g = abs(fract(uvW - 0.5) - 0.5) / fw;
  float lineDist = min(g.x, g.y);
  // Line coverage with smooth AA across ~2 backing-px. smoothstep
  // (0.5, 1.5, lineDist) gives full coverage where lineDist ≤ 0.5
  // fragment-units, smooth fade to 0 at lineDist = 1.5. Result: a
  // 1-fragment-bright core + 1-fragment AA tail on each side =
  // visually ~2-3 backing-px wide line. Wider lines than the
  // canonical 1-fragment Made-by-Evan pattern, but more legible on
  // hi-DPR displays where 1 backing-px is sub-pixel-visible.
  float line = 1.0 - smoothstep(0.5, 1.5, lineDist);

  // Early discard for non-line fragments. Saves ~95% of the depth
  // fade math for typical line widths. Without this, the alpha-blend
  // path still has to do per-fragment work even where line ≈ 0.
  if (line < 0.01) discard;

  // Depth fade: replaces V11.2's painter's algorithm sort. The painter
  // sort darkened lines occluded by foreground geometry — here we use
  // sag amplitude as a proxy: deeper sag → dimmer line. Not bit-
  // equivalent to painter sort (back rims and front rims of the same
  // well dim equally), but it gives the same "you are above, the well
  // is below" visual cue and avoids the painter-sort's per-frame CPU
  // cost (~1-2 ms for the segment sort).
  //   uMaxSag is the per-frame peak sag (raw, before yFactor).
  //   Normalizing by uMaxSag keeps the dynamic range adapted to the
  //   scene — flat scenes get no dimming, deep scenes get the full
  //   1.0 → uContrastFloor ramp. The 0.5 floor for uMaxSag avoids
  //   divide-by-tiny in nearly-empty scenes.
  float depthFactor = clamp(sag / max(uMaxSag, 0.5), 0.0, 1.0);
  float brightness = mix(1.0, uContrastFloor, depthFactor);

  outColor = vec4(uColor.rgb * brightness, uColor.a * line);
}`,
};

// --- Particle flow (sparse luminous overlay on 2D grid warp) -----------
// Renders glowing point sprites advected along the gravitational force
// field. Particle positions are computed on the CPU each frame (cheap at
// ~200 particles) and uploaded to a VBO. The shader draws each particle
// as a soft circular sprite via gl_PointSize + radial alpha falloff.
//
// Additive blending creates the "luminous dust" feel — particles in
// dense regions glow brighter as they overlap.
//
// Per-vertex attributes: aPos (vec2, CSS-px), aAge (float, 0..1).
// Per-program uniforms: uOrtho, uColor, uPointSize.
export const PARTICLE_FLOW = {
  VS: `#version 300 es
in vec2 aPos;
in float aAge;        // 0 = just-spawned, 1 = about to recycle
uniform mat4 uOrtho;
uniform float uPointSize;
out float vAge;
${SAG_VS_HELPER}
void main() {
  vAge = aAge;
  // Younger particles are slightly bigger so the recycle flicker is
  // less visible — they fade in/out instead of pop.
  float ageMul = mix(1.2, 0.6, aAge);
  gl_PointSize = uPointSize * ageMul;
  // Project each particle's CSS-px position through sag so particles
  // flow visually along the projected gravity wells.
  gl_Position = uOrtho * vec4(sagProject(aPos), 0.0, 1.0);
}`,
  FS: `#version 300 es
precision highp float;
in float vAge;
uniform vec4 uColor;
out vec4 outColor;
void main() {
  // gl_PointCoord is [0..1] across the point sprite. Radial alpha:
  // peaks at center (1.0), zero at edges (0.5 radius). Smooth falloff.
  vec2 uv = gl_PointCoord - 0.5;
  float r = length(uv);
  if (r > 0.5) discard;
  float radial = 1.0 - smoothstep(0.0, 0.5, r);
  // Age fade: bright when young, dim when old. Triangle ramp gives a
  // gentle birth+death so the eye doesn't catch the recycle.
  float ageFade = 1.0 - abs(vAge * 2.0 - 1.0);
  outColor = vec4(uColor.rgb, uColor.a * radial * ageFade);
}`,
};
