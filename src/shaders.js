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
uniform int uMode;            // 0 = 3D oblique, 1 = 2D in-plane
uniform mat4 uOrtho;          // CSS-px → NDC
out float vIntensity;         // displacement magnitude — drives alpha in FS
void main() {
  float eps2 = uEpsilon * uEpsilon;
  vec2 p = aPos;
  float phi = 0.0;             // Σ φ for 3D mode
  vec2 gradPhi = vec2(0.0);    // Σ ∇φ for 2D mode
  for (int i = 0; i < MAX_ENTITIES; i++) {
    if (i >= uEntityCount) break;
    vec4 e = uEntities[i];
    vec2 d = p - vec2(e.x, e.y);
    float r2 = d.x * d.x + d.y * d.y + eps2;
    float invR = inversesqrt(r2);
    float invR3 = invR * invR * invR;
    if (uMode == 0) {
      phi += -e.z * invR;
    } else {
      // ∇φ = G·q·m · d / (r²+ε²)^(3/2). Negate so vertices move
      // TOWARD attractor (visually intuitive).
      gradPhi += -e.z * d * invR3;
    }
  }
  vec2 worldPx;
  float intensity;
  if (uMode == 0) {
    // 3D oblique: phi is negative for attractors → -phi is positive
    // depth. Project Z into Y as a "tilt down" — depressions appear
    // below the body, matching the rubber-sheet mental model.
    float depth = clamp(-phi * uDispScale, 0.0, 250.0);
    worldPx = vec2(p.x, p.y + depth * uTiltY);
    intensity = depth;
  } else {
    // 2D in-plane: displacement capped so a body at distance ≈ ε
    // doesn't pull a vertex onto itself (visual nonsense).
    vec2 disp = gradPhi * uDispScale;
    float magSq = dot(disp, disp);
    float cap = 60.0;
    if (magSq > cap * cap) disp = disp * (cap / sqrt(magSq));
    worldPx = p + disp;
    intensity = length(disp);
  }
  vIntensity = intensity;
  gl_Position = uOrtho * vec4(worldPx, 0.0, 1.0);
}`,
  FS: `#version 300 es
precision highp float;
in float vIntensity;
uniform vec4 uColor;
uniform float uIntensityHalf;
out vec4 outColor;
void main() {
  float t = smoothstep(0.0, uIntensityHalf, vIntensity);
  float a = mix(0.25, 1.0, t);
  outColor = vec4(uColor.rgb, uColor.a * a);
}`,
};
