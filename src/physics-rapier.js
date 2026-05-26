// physics-rapier.js — Default physics backend (Rapier2D via WASM).
//
// Replaces planck.js. Rapier is Rust → WASM, ~5-10× faster than planck
// at typical scene sizes, and exposes `lengthUnit` directly on the world
// so the unit-scale wrangling we did for planck (Settings.linearSlop=0.5,
// Settings.maxTranslation=50, etc.) becomes a single-knob configuration
// here.
//
// Closure-interface contract — identical to physics-planck.js:
//   init(entities)               build world + bodies
//   prepareFrame(entities)       per-frame sync (currently same as
//                                inside step; kept for parity)
//   step(entities, dt,           one substep
//        viewport,
//        boundaryMode,
//        isLastSubstep)
//   onEntityMetaMaybeChanged()   slider edits (no-op for now, matches
//                                planck behavior)
//   destroy()                    teardown
//
// What WE still own:
//   - Charge-asymmetric Plummer-softened gravity (computeGravity here)
//   - BH absorption animation lifecycle (entity.absorbing object)
//   - applyBoundary destroy/wrap handling
//   - Entity-array lifecycle
//
// Wrap-boundary handling — IMPORTANT:
//   Rapier (like Box2D / planck) uses a Euclidean broadphase (BVH-based)
//   that knows nothing about wrap. Pre-Rapier we hit a bug where a body
//   wrapping from x=W+ε to x=ε would instantly appear next to a cluster
//   at x=2, planck would register a fake high-velocity contact and apply
//   a catastrophic impulse → oscillation.
//
//   Rapier fix in step(): after applyBoundary teleports a body across the
//   wrap edge, we DESTROY the Rapier rigid body and RECREATE it at the
//   wrapped position. This atomically clears all contact pairs that
//   reference the pre-wrap collider, so the new body starts with zero
//   contact history. Re-creation cost ~µs per wrap event; only a tiny
//   fraction of bodies wrap per frame in typical scenes.
//
// Unit scale:
//   world.lengthUnit = 30 (1 meter ≈ 30 px). All Rapier tolerances
//   (allowedLinearError, contact_erp, sleep thresholds, etc.) are
//   normalized by lengthUnit internally, so this single setting brings
//   everything into pixel-sane defaults.
//
// Sleep:
//   Bodies have setCanSleep(true). Mirror of physics-planck.js logic:
//   the gravity-apply loop checks isSleeping() and computes
//   |accel|*dt — if that would push velocity above sleep tolerance,
//   the force is applied (Rapier's addForce wakes by default).
//   Otherwise the body stays asleep.

// RAPIER is injected by the caller — see setRapier() below. Two contexts:
//   - Main thread (physics-backend.js → makeRapierBackend): imports RAPIER
//     via the bare specifier `@dimforge/rapier2d-compat` resolved through
//     index.html's import map, then calls setRapier() before invoking
//     makeRapierBackend().init().
//   - Worker thread (physics-rapier-worker.js): cannot use import maps
//     (workers don't inherit them), so imports via the full CDN URL and
//     calls setRapier() the same way.
// Calling makeRapierBackend().init() without setRapier() will throw.
let RAPIER = null;
export function setRapier(rapierModule) { RAPIER = rapierModule; }
import { state } from './state.js';
import { updateAbsorptions, applyBoundary } from './physics.js';
import { detectBackend, loadKernel, isVerbose } from './gpu-init.js';
import { createGravityGPU } from './physics-gpu-gravity.js';
import { AdaptiveOverlapManager } from './physics-planck-overlap.js';
import { recordSubstep } from './state-dump.js';
import { setContext as perfSetContext, isEnabled as perfEnabled, markPhase as perfMark } from './perf-monitor.js';
import { getQualityKnobs, resetQuality } from './quality-manager.js';

// ── Unit-scale + solver knobs ─────────────────────────────────────────
// lengthUnit converts pixel coordinates to physics-engine "meters" so
// Rapier's internal tolerances (slop, sleep thresholds, etc.) land at
// values appropriate for our scene (radii 8-30 px, viewport ~2000 px,
// velocities ≤ 1000 px/s). 30 px ≈ 1 m is the standard game ratio.
const LENGTH_UNIT = 30;

// Velocity solver iterations. Rapier's default is 4; we previously
// over-shot to 8 to chase dense-cluster perfection. Empirically the
// residual error at 4 iters is < 1 % of contact-spring stiffness — not
// visible. 8 iters cost ~2× the solver time for invisible accuracy gain.
// MAX governs the adaptive escalator upper bound. Reduced 12 → 8
// (Phase C, 2026-05-25): perf-mon found worldStep max 19ms with 12-iter
// cap firing every dense-cluster frame. 8 cuts that to ~13ms while
// keeping 2× baseline for escalation headroom. Acceptable trade-off
// per user spec ("失去物理保真度换性能"): slightly more sub-pixel
// penetration in extreme piles but never beyond body-radius.
const SOLVER_ITERATIONS_BASE = 4;
const SOLVER_ITERATIONS_MAX  = 8;

// NGS position solver iterations. Rapier's default is 1; we previously
// used 4 to suppress sub-pixel penetration in dense piles. 2 is a stable
// middle ground: penetration depth stabilizes at ~1.5 px (vs ~0.5 px at
// 4 iters). Cap reduced 6 → 4 (Phase C) for the same reason as velocity.
const PGS_ITERATIONS_BASE = 2;
const PGS_ITERATIONS_MAX  = 4;

// Sleep tolerance (Rapier exposes this via integration params; in px/s
// after lengthUnit conversion). Mirrors what gets passed to Rapier's
// internal sleep threshold setter.
const SLEEP_LINEAR_THRESHOLD_PX_PER_S = 5.0;

// JS-side gravity-wake gate. We avoid the applyImpulse WASM round-trip
// for sleeping bodies whose acceleration is small enough that they
// wouldn't accumulate meaningful motion before Rapier re-sleeps them.
//
// HISTORY (2026-05-25 BUG): the first version compared `|a|·dt` (Δv
// per substep) against SLEEP_LINEAR_THRESHOLD_PX_PER_S (5 px/s). That
// is a UNIT MISMATCH — `|a|·dt` at dt=1/60 produces ≈ 0.02–0.17 px/s
// for typical gravity (|a| = 1–10 px/s²), always under the 5 px/s
// threshold. Single-body scenes with click-placed (vel=0) bodies
// spawned in Rapier's sleeping state and never woke: gravity computed,
// but `forceApplied=false` every substep (confirmed in
// dump_1779714748246.json — entity 351 frozen with gravity (1.18, 4.56)).
//
// CORRECT discriminator: raw acceleration magnitude. In a settled
// pile, pairwise forces partially cancel and the net |a| is typically
// ≤ 0.05 px/s² — stays asleep. In free space with a meaningful
// attractor (G=80, m=100 at r=200 → |a|=0.2 px/s²), |a| crosses
// 0.1 cleanly → wakes.
const SLEEP_WAKE_ACCEL_THRESHOLD_SQ = 0.01;   // (0.1 px/s²)²

// Dynamic CCD reeval — every N RAF frames, walk all bodies and toggle
// each body's CCD flag based on its CURRENT velocity rather than its
// spawn-time velocity. Without this, a body that spawned slow but is
// now orbiting at 800 px/s would have CCD off (tunneling risk), and a
// body that spawned fast but is now resting in a cluster would still
// pay the CCD cost forever. 10 frames ≈ 167 ms at 60 Hz — fast enough
// to respond to gravity-induced acceleration, slow enough that bodies
// hovering near the threshold don't thrash CCD on/off.
const CCD_REEVAL_INTERVAL_FRAMES = 10;
let _ccdReevalCounter = 0;

// TGS-Soft contact-spring natural frequency ω₀ is now user-tunable
// via state.contactStiffness (高级调参 → 接触刚度). Default 6.
//
// SPAWN-EXPLOSION GUARD (2026-05-25 redesign — semantic change):
// The user's stiffness setting applies ONLY to bodies that are NOT
// currently in spawn-overlap resolution. While any body still has
// `_spawnDampingSubstepsLeft` set (i.e., its initial spawn placed it
// inside another collider and Rapier is still pushing it out), the
// world-wide contact_natural_frequency is clamped to SPAWN_SAFE_STIFFNESS.
// Once that body cleanly exits the resolution phase (overlap query
// returns empty → counter deleted), the world stiffness returns to
// state.contactStiffness on the next prepareFrame.
//
// This makes spawn-explosion impossible by construction — the high
// stiffness never gets a chance to act on a spawn-overlap. The
// previous mechanism (damping coefficient scaled to 2·ω₀) is no
// longer needed for safety; damping is now a constant 2·SPAWN_SAFE_STIFFNESS,
// purely a smoothing aid for the safe-stiffness contact resolution.
//
// Edge case: existing bodies in contact during a spawn event will
// briefly see the safe stiffness too (the gate is world-wide, not
// per-pair — Rapier's integration params are a global setting).
// Spawn-resolution windows are < 0.5 sec; the visual impact is nil.

// Safe stiffness ceiling enforced during any active spawn-overlap
// resolution. Matches the original default. Used by both the
// world-wide gate (see prepareFrame) and the spawn damping coefficient.
const SPAWN_SAFE_STIFFNESS = 6;

// Gate release cadence (frames). After the last body cleanly exits
// spawn-overlap, the world stiffness LINEARLY INTERPOLATES from
// SPAWN_SAFE_STIFFNESS toward state.contactStiffness over this many
// frames before fully releasing.
//
// Why a smooth ramp instead of binary on/off:
//   - Bodies that just cleared overlap are still tightly clustered.
//   - Gravity immediately pulls them back into shallow re-contact.
//   - A discrete stiffness jump (SAFE → user value) at any single
//     frame means that frame's contact spring force suddenly grows
//     by (user/SAFE)² — exactly 100× at ω₀=60 vs SAFE=6 — producing
//     a velocity injection → explosion.
//   - A smooth ramp distributes the spring-force growth over many
//     frames, giving gravity + Rapier's contact damping time to
//     dissipate the per-frame energy injection before it accumulates.
//
// Empirical evidence: 5-body same-point stack at ω₀=60 with discrete
// 30-frame cooldown still produced 290 px/s explosion. With smooth
// ramp over 60 frames, drops below acceptance threshold for realistic
// scenarios (chain placements). Contrived same-point stacks of N≥5
// at ω₀≥60 remain a known-unavoidable edge case: bodies cleanly resolve
// out of overlap under safe stiffness, gravity pulls them back into a
// tight cluster, and once the ramp finishes the cluster's shallow
// contacts at user stiffness produce ~75 px/s residual motion. All
// bodies survive — no boundary loss. Realistic interaction cannot
// reach this state (a user cannot click-place 5 bodies at identical
// coordinates).
const GATE_RELEASE_RAMP_FRAMES = 60;
let _gateRampLeft = 0;

// Normalized slop threshold. Default ~0.001 (= 0.03 px at
// lengthUnit=30). Widening to 0.005 (~0.15 px) cuts final-
// convergence oscillation on near-resolved contacts without
// visible geometry impact.
const NORMALIZED_ALLOWED_LINEAR_ERROR = 0.005;

// Conditional spawn-damping: applied to a newly-created body ONLY
// if Rapier's intersectionsWithShape query reports it overlaps an
// existing collider at the spawn moment. Goal per user spec: bodies
// arrive at the no-overlap boundary with relative velocity ≈ 0
// (not "fly out at residual speed").
//
// Approach: critical damping during the burst + active termination
// the moment overlap clears.
//   1. d = 2·ω₀ ≈ 12 (critical damping at ω₀ = 6). Critically
//      damped springs decelerate monotonically toward equilibrium
//      without overshoot. Rapier 0.12+ applies linearDamping as
//      `v *= max(0, 1 − d·dt)` per substep (linear, clamped). At
//      d=12, dt=1/60: factor = 0.80/sub — velocity decays smoothly.
//   2. Each substep we re-query intersectionsWithShape at the
//      body's CURRENT Rapier position. The instant the query
//      reports no-overlap, we call `setLinvel(0, 0)` on the body,
//      restore linearDamping=0, and clear the counter. This is
//      the only place in the codebase where setLinvel is called
//      on an existing dynamic body — a clean transition point
//      that ends the spawn-resolution phase (blueprint §3 P2
//      exception M7).
//   3. SPAWN_DAMPING_SUBSTEPS is now a SAFETY CAP (≤ 30 substeps
//      ≈ 0.25 s real-time). The normal path terminates earlier
//      via overlap-detection. If the cap is hit (e.g., body
//      completely engulfed → can't separate), we release damping
//      and accept current state — DOES NOT snap velocity (the
//      cap is reserved for pathological cases).
//
// Non-overlapping spawns (drag-place into empty space, slingshot
// launch) skip this entirely — no damping, no termination, full
// launch velocity preserved.
//
// Rebuild paths (wrap-teleport, meta-drift) also skip this — see
// the isRebuild=true callers (code-review F1).
//
// Damping coefficient: 2·SPAWN_SAFE_STIFFNESS (= 12, critical damping
// at ω₀ = 6). DECOUPLED from state.contactStiffness as of the
// 2026-05-25 semantic redesign: while a body is in spawn-overlap
// resolution, the world contact stiffness is already clamped to
// SPAWN_SAFE_STIFFNESS, so the damping doesn't need to track the
// user's stiffness setting — it just needs to match whatever the
// active spawn-time stiffness is.
const SPAWN_DAMPING_VALUE = 2 * SPAWN_SAFE_STIFFNESS;
function _spawnDampingValue() { return SPAWN_DAMPING_VALUE; }
const SPAWN_DAMPING_SUBSTEPS = 30;

let RAPIER_READY = false;
let world = null;
let eventQueue = null;       // RAPIER.EventQueue — collects contact start/stop
                             // events that fire INSIDE world.step. Drained
                             // each substep, then passed into the state-dump
                             // trace so the analyzer can see brief impacts
                             // even when the post-step contactPairsWith
                             // query sees nothing (because the contact has
                             // already resolved by the time we query).
let bodyById = new Map();    // entity.id (u32) → Rapier RigidBody
let colliderById = new Map(); // entity.id → Rapier Collider
let overlapMgr = null;

// ── Wrap-boundary ghost bodies ──────────────────────────────────────
// Rapier's broadphase is Euclidean — it has no notion of toroidal
// topology. To make bodies near opposite edges "see" each other for
// contact, we maintain mirror copies (ghosts) placed at the 8 possible
// wrap offsets. Each ghost is a dynamic body whose position/velocity
// are slaved to its real body at the start of each substep. Other
// bodies can collide with ghosts; the contact impulse the ghost
// receives is forwarded to the real body via applyImpulse after the
// step (see step()'s ghost-delta phase).
//
// Map: realEntityId → Map<edgeSig, ghostBody>
// edgeSig is one of: 'L', 'R', 'T', 'B', 'LT', 'LB', 'RT', 'RB'.
const ghostsByRealId = new Map();

// Viewport-change tracking for ghost invalidation. Ghosts of sleeping
// real bodies retain their last-synced offset; if the viewport W/H
// changes (window resize), the offsets become stale. We force-destroy
// all ghosts when W/H change so the next syncGhosts rebuilds fresh.
let _lastSyncedViewportW = 0;
let _lastSyncedViewportH = 0;

// Collision-group encoding: Rapier packs membership and filter masks
// into a single u32 as `(membership << 16) | filter`.
// REAL: belongs to group 1, collides with groups 1 and 2 (REAL and GHOST).
// GHOST: belongs to group 2, collides only with group 1 (REAL — never
//        with other GHOSTs, which would double-count cross-wrap contacts).
const COLLISION_GROUP_REAL  = (0x0001 << 16) | 0x0003;
const COLLISION_GROUP_GHOST = (0x0002 << 16) | 0x0001;

// Floor for the dynamic ghost-creation margin. The effective margin
// used per-substep is `max(GHOST_MARGIN_PX_FLOOR, maxRadiusInScene + 30)`
// so even with very-large-radius bodies (UI radius slider up to 10×
// base), any cross-wrap contact pair always has both ghosts spawned
// before they could physically contact. See _effectiveGhostMargin.
const GHOST_MARGIN_PX_FLOOR = 120;

// Stage 5 skip-sync thresholds. When the real body's cumulative drift
// since the last ghost sync stays under these, we skip the per-ghost
// setTranslation/setLinvel WASM round-trips for existing ghosts (new
// ghost creation + stale ghost destruction still runs — those are cheap
// JS-map operations). Ghost position is at most √(POS_SQ) ≈ 0.5 px
// stale, which is < 4 % of the smallest body radius (14 px). Contact
// normal direction error is bounded at ~2 % — well below the solver's
// already-tolerated residuals.
const GHOST_SYNC_POS_DELTA_SQ = 0.25;   // (0.5 px)²
const GHOST_SYNC_VEL_DELTA_SQ = 25.0;   // (5 px/s)²

// Reusable Set for dedup'ing contact pairs in step()'s touching-count
// loop. Module-scope to avoid per-step allocation.
const _contactCounterSeen = new Set();

// Iter 2 (2026-05-25): perf-mon found contactsTrace eating ~28 ms / 35 ms
// max at N=400 dense (vs Rapier's own world.step at only ~2 ms!). Cause:
// 1111 contact pairs × readManifold reading normal + depth + numContacts
// = ~4400 WASM round-trips per frame. Two-pronged fix:
//   (1) When dump is OFF, skip normal/depth reads — just check numContacts
//       > 0 via the lighter `_hasLiveContact`. Cuts ~50 % of per-pair cost.
//   (2) Sample only every Nth frame in dump-off mode. Adaptive overlap
//       manager's state machine tolerates a few frames of stale count;
//       it's used for next-frame iter decision, not real-time control.
const CONTACT_SAMPLE_EVERY_N = 3;
let _contactSampleCounter = 0;

// Stage 6b: reusable {x, y} scratch object for WASM-bound vector args
// (applyImpulse, setTranslation, setLinvel). Rapier 0.19 JS bindings
// marshal vector arguments synchronously inside the call (wasm-bindgen
// reads x/y into Rust f32 fields before returning), so we can safely
// mutate this single object across consecutive calls without WASM
// retaining a reference.
//
// Replaces ~800 per-frame `{x: a, y: b}` literal allocations at N=200
// in the common (non-wrap, non-absorbing) case. In wrap mode with many
// edge-adjacent bodies the saving climbs into the low thousands per
// frame. All hot-path sites converted; the two remaining literals at
// pin-toggle (line ~934) and spawn-damp clean-exit (line ~993) are
// cold (fire on user UI events, not per substep) and use this scratch
// too for consistency.
const _vec2A = { x: 0, y: 0 };

// ── GPU gravity threshold ───────────────────────────────────────────
// Floor for considering GPU path. Above this N the GPU CANDIDATE path
// runs; below it CPU always. The runtime crossover detector (see
// computeGravityGPUSync + _gpuDisabled) then measures actual GPU cost
// and latches GPU off if it's slower than expected. So the constant
// here is just a "don't even probe at trivial N" floor — set well
// below the user's typical workload so the detector has data.
//
// 200 → 400 → 300 → 400 → 1500 → 300 history: the constant kept being
// wrong because user environment varied. Replaced its decision power
// with the runtime detector; the constant now only gates probe entry.
const GPU_THRESHOLD_SYNC  = 300;
let gpuDevice = null;
let gpuGravityHandle = null;

async function tryInitGpuGravity() {
  try {
    const detection = await detectBackend();
    if (detection.backend !== 'webgpu') {
      if (isVerbose()) console.info('[physics-rapier] no WebGPU, gravity stays on CPU:', detection.reason);
      return false;
    }
    const wgslSource = await loadKernel('./kernels/gravity_accel.wgsl');
    gpuDevice = detection.device;
    gpuGravityHandle = await createGravityGPU(gpuDevice, wgslSource);
    if (isVerbose()) console.info('[physics-rapier] GPU K1 gravity ready');
    return true;
  } catch (e) {
    if (isVerbose()) console.warn('[physics-rapier] GPU init failed; staying on CPU gravity:', e);
    return false;
  }
}

// Synchronous GPU gravity path. Used when N ≥ GPU_THRESHOLD_SYNC.
// Uses staging slot 0. The async pipeline that previously used slot 1
// was removed (see GPU_THRESHOLD_SYNC comment block).
// Crossover detector state (#1, 2026-05-25): the static GPU_THRESHOLD_SYNC
// constant is fundamentally wrong because the CPU-vs-GPU crossover depends
// on hardware + browser environment we can't know at build time. Instead,
// time the first GPU_PROBE_SAMPLES dispatches; if avg cost > the disable
// threshold, latch GPU off for the session and use CPU forever. Catches:
// headless throttle, browser GPU process congestion, slow GPU hardware.
const GPU_PROBE_SAMPLES = 5;
const GPU_DISABLE_THRESHOLD_MS = 3.0;   // > 3 ms / dispatch → CPU wins
let _gpuProbeSamples = [];
let _gpuDisabled = false;
// Consecutive GPU dispatch exceptions before latching off. Distinct from
// the probe path which latches based on AVERAGE COST being too slow; this
// catches the case where dispatch / readback THROWS (e.g., device lost,
// buffer destroyed mid-flight). Without this latch, every substep retries
// the failing GPU call and silently falls back to CPU forever.
// Code-review MEDIUM 2026-05-25 (silent-failure-hunter).
const GPU_FAIL_LATCH_COUNT = 3;
let _gpuConsecutiveFailures = 0;

async function computeGravityGPUSync(entities, n) {
  gpuGravityHandle.growIfNeeded(n);
  const wrap = state.boundaryMode === 'wrap';
  gpuGravityHandle.uploadPositions(entities, n);
  gpuGravityHandle.uploadMetaAll(entities, n);
  gpuGravityHandle.uploadParams(
    n, state.G, state.epsilon,
    wrap ? state.viewport.width  : 0,
    wrap ? state.viewport.height : 0,
  );
  // Probe timer must wrap submit() AND readback so the measurement
  // captures dispatch-queue wait time too. If we only timed readback,
  // a GPU finishing the kernel quickly inside submit()'s synchronous
  // queue would read near-zero — leaving GPU enabled on hardware where
  // total round-trip is actually slow.
  const t0 = performance.now();
  const enc = gpuDevice.createCommandEncoder({ label: 'rapier sync gravity' });
  gpuGravityHandle.recordDispatch(enc, n, 0);
  gpuDevice.queue.submit([enc.finish()]);
  const result = await gpuGravityHandle.readbackStaging(0, n);
  const elapsedMs = performance.now() - t0;
  if (_gpuProbeSamples.length < GPU_PROBE_SAMPLES) {
    _gpuProbeSamples.push(elapsedMs);
    if (_gpuProbeSamples.length === GPU_PROBE_SAMPLES) {
      const avg = _gpuProbeSamples.reduce((a, b) => a + b, 0) / GPU_PROBE_SAMPLES;
      if (avg > GPU_DISABLE_THRESHOLD_MS) {
        _gpuDisabled = true;
        // Reset the throw-failure counter too so the latent invariant
        // "_gpuDisabled cleared ⇒ counter at 0" holds (a future code path
        // that clears _gpuDisabled wouldn't immediately re-latch on the
        // first throw). Code-review HIGH 2026-05-25.
        _gpuConsecutiveFailures = 0;
        if (isVerbose()) {
          console.warn(`[physics-rapier] GPU gravity avg ${avg.toFixed(1)} ms/dispatch ` +
                       `> ${GPU_DISABLE_THRESHOLD_MS} ms threshold; CPU fallback latched.`);
        }
      } else if (isVerbose()) {
        console.info(`[physics-rapier] GPU gravity avg ${avg.toFixed(1)} ms/dispatch — GPU stays on.`);
      }
    }
  }
  return result;
}

function computeGravityCPU(entities, n, out) {
  const G = state.G;
  const eps = state.epsilon;
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width  : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;
  for (let i = 0; i < n; i++) {
    const a = entities[i];
    if (a.absorbing) continue;
    for (let j = i + 1; j < n; j++) {
      const b = entities[j];
      if (b.absorbing) continue;
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
      const r2Raw = dx * dx + dy * dy;
      const minR = Math.max(a.radius + b.radius, eps);
      const r2 = r2Raw + minR * minR;
      const r = Math.sqrt(r2);
      const nx = dx / r;
      const ny = dy / r;
      if (b.charge !== 0) {
        const mag = b.charge * G * b.mass / r2;
        out[i * 2]     += mag * nx;
        out[i * 2 + 1] += mag * ny;
      }
      if (a.charge !== 0) {
        const mag = a.charge * G * a.mass / r2;
        out[j * 2]     += mag * -nx;
        out[j * 2 + 1] += mag * -ny;
      }
    }
  }
  return out;
}

// ── Body lifecycle ───────────────────────────────────────────────────

// `isRebuild = true` is set by callers that destroy + recreate an
// existing entity's body (wrap-teleport, meta-drift slider edit). On
// the rebuild path we MUST NOT apply the spawn-damping burst because:
//   - wrap-rebuild: the body has been continuously simulated and is
//     just being teleported; treating each wrap as a "spawn" would
//     decay its velocity 34 %/substep × 4 every time it crossed an
//     edge — destroying orbits in wrap mode (see code-review F1).
//   - meta-drift rebuild: the user is interactively editing radius/
//     mass via the slider; injecting damping mid-edit would feel
//     unpredictable. The softer global contact spring still resolves
//     any new overlap gently.
// Initial spawn (user click → input.js → state.entities.push) is
// the only path that calls this with isRebuild = false (the default).
function createBodyForEntity(e, isRebuild = false) {
  const initSpeed = Math.hypot(e.vx || 0, e.vy || 0);
  const wantsCcd = initSpeed > state.overlapBulletThreshold;

  let desc;
  if (e.pinned) {
    desc = RAPIER.RigidBodyDesc.kinematicPositionBased();
  } else {
    desc = RAPIER.RigidBodyDesc.dynamic();
  }
  // Decomposed (not chained) — some rapier2d-compat 0.19.x builder methods
  // either don't return `this` or have surprising shapes. Calling each in
  // statement form makes chain-break failure modes impossible.
  desc.setTranslation(e.x, e.y);
  desc.setLinvel(e.vx || 0, e.vy || 0);
  desc.setCanSleep(true);
  desc.setCcdEnabled(wantsCcd);
  desc.lockRotations();

  const body = world.createRigidBody(desc);
  // userData is a post-construction plain JS property on RigidBody in
  // rapier2d-compat (NOT a builder method on the desc). Storing the
  // entity id here lets future code identify which body belongs to which
  // entity in callbacks (currently we use bodyById Map; userData is kept
  // for parity with planck path and future contact-event use).
  body.userData = e.id;

  const r = Math.max(e.radius, 0.01);
  const density = e.mass / (Math.PI * r * r);
  let cDesc = RAPIER.ColliderDesc.ball(r)
    .setDensity(density)
    .setFriction(0)
    .setRestitution(state.elasticRestitution)
    // ActiveEvents.COLLISION_EVENTS opts this collider into emitting
    // contact-started / contact-stopped notifications into the world's
    // EventQueue. Default is "no events" — we need this on every collider
    // (sensors included) for the state-dump trace to see brief impacts.
    .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)
    // Real body — collides with both other REAL bodies and with GHOSTs
    // (which represent real bodies on the other side of a wrap edge).
    .setCollisionGroups(COLLISION_GROUP_REAL);
  // Black holes are sensors — they detect overlap (so the absorption
  // lifecycle in detectAndStartBHAbsorptions can fire) but apply no
  // impulse to penetrating planets. Sensor "intersection" start/stop also
  // fires through drainCollisionEvents when COLLISION_EVENTS is enabled.
  if (e.type === 'black_hole') cDesc = cDesc.setSensor(true);

  const collider = world.createCollider(cDesc, body);

  // We store isCcd as a JS-side mirror to skip redundant Rapier toggles.
  body._isCcdEnabled = wantsCcd;

  // Stamp the entity's "physics-baked" meta. syncWorldToEntities checks
  // these every substep — if e.mass / e.radius / e.type drift (because
  // the user moved the mass or radius slider while the entity is selected,
  // or toggled planet ↔ black_hole), the body MUST be destroyed and
  // recreated (Rapier has no setMass / setRadius / setSensor API safe to
  // call mid-life). Stamping here keeps the bake-vs-live comparison cheap.
  e._bakedMass   = e.mass;
  e._bakedRadius = e.radius;
  e._bakedType   = e.type;

  bodyById.set(e.id, body);
  colliderById.set(e.id, collider);

  // Conditional spawn-damping: only if this is an INITIAL spawn
  // (not a rebuild) AND the body actually overlaps an existing
  // collider at creation time, apply a 4-substep linear-damping
  // burst to absorb the contact-spring's separation energy.
  // Non-overlapping spawns (empty space / slingshot) and rebuilds
  // (wrap-teleport, meta-drift) get nothing here. The overlap
  // check uses Rapier's native intersectionsWithShape query (no
  // JS-side geometry iteration). Black-hole sensors are excluded
  // (they don't physically collide; absorption is JS-driven).
  if (!isRebuild &&
      e.type !== 'black_hole' &&
      _spawnOverlapsExisting(e, collider.handle)) {
    body.setLinearDamping(_spawnDampingValue());
    e._spawnDampingSubstepsLeft = SPAWN_DAMPING_SUBSTEPS;
  }

  return body;
}

// Asks Rapier's narrowphase: "given a ball of radius e.radius at
// (e.x, e.y), does it intersect any existing collider?" Returns true
// on first hit. Uses world.intersectionsWithShape — Rapier's spatial
// query, NOT a JS-side geometric iteration. The just-created
// collider for this entity is excluded via filterExcludeCollider so
// the query never matches the body against itself.
//
// Filter: COLLISION_GROUP_REAL = (0x0001 << 16) | 0x0003.
//   - Query treats itself as group 1 (REAL).
//   - Filter mask 0x0003 = groups 1+2 → matches both REAL bodies AND
//     GHOST wrap-mirror bodies.
// Hitting a GHOST IS a real overlap: the ghost represents matter on
// the other side of the wrap edge, and the spawn-damping should
// suppress the would-be explosion when a body spawns next to a
// wrap-mirrored body.
// Per-substep "is this spawn-damped body still overlapping anything?"
// query, used by the countdown loop to detect the clean-exit moment.
// Reads the body's CURRENT Rapier position (not the JS mirror, which
// is stale by one substep) and asks the narrowphase for any
// intersecting collider via intersectionsWithShape.
//
// On query failure (API mismatch / WASM hiccup) returns true (be
// conservative — keep damping rather than prematurely snap to v=0).
function _bodyStillOverlapping(body, e, ownColliderHandle) {
  if (!world) return false;
  const pos = body.translation();
  const r = Math.max(e.radius, 0.01);
  const shape = new RAPIER.Ball(r);
  let hit = false;
  try {
    world.intersectionsWithShape(
      { x: pos.x, y: pos.y },
      0,
      shape,
      (otherHandle) => { hit = true; return false; },  // first hit aborts
      undefined,
      COLLISION_GROUP_REAL,
      ownColliderHandle,
    );
  } catch {
    try {
      hit = false;
      world.intersectionsWithShape(
        { x: pos.x, y: pos.y },
        0,
        shape,
        (otherHandle) => {
          if (otherHandle !== ownColliderHandle) { hit = true; return false; }
          return true;
        },
      );
    } catch (err) {
      // Both intersectionsWithShape variants threw. Surface this — a
      // persistent failure here means spawn-damping silently rides to
      // the SPAWN_DAMPING_SUBSTEPS cap for every spawned body, which
      // zeros launch velocity (drag-place would stop dead). The
      // conservative `return true` (keep damping) is the right behavior
      // but the failure shouldn't be invisible.
      // Code-review MEDIUM 2026-05-25 (silent-failure-hunter).
      if (isVerbose()) console.warn('[physics-rapier] _bodyStillOverlapping: both intersect variants failed; conservative return=true', err);
      return true;
    }
  }
  return hit;
}

function _spawnOverlapsExisting(e, ownColliderHandle) {
  if (!world) return false;
  const r = Math.max(e.radius, 0.01);
  const shape = new RAPIER.Ball(r);
  let hit = false;
  try {
    world.intersectionsWithShape(
      { x: e.x, y: e.y },
      0,                         // rotation
      shape,
      (otherHandle) => { hit = true; return false; },  // first hit aborts
      undefined,                 // filterFlags
      COLLISION_GROUP_REAL,      // membership=REAL → collides with REAL+GHOST;
                                 // but the filter is interpreted by Rapier as
                                 // "treat the query shape as a member of group
                                 // 1 (REAL) — i.e., test against group-1 colliders
                                 // and group-2 colliders that allow group 1 contact"
      ownColliderHandle,         // exclude self
    );
  } catch (err) {
    // Some 0.19 bundles have a slightly different parameter list. Fall
    // back to the minimal signature; downside is the self-exclusion
    // filter is dropped, so we have to manually re-check the hit handle.
    try {
      hit = false;
      world.intersectionsWithShape(
        { x: e.x, y: e.y },
        0,
        shape,
        (otherHandle) => {
          if (otherHandle !== ownColliderHandle) { hit = true; return false; }
          return true;
        },
      );
    } catch {
      // If the entire query API isn't reachable, fail safe by skipping
      // spawn-damping (body spawns normally — original explosion
      // behavior remains, but no further regression).
      return false;
    }
  }
  return hit;
}

function destroyBody(id) {
  // Destroy all ghosts of this real body first — they have no purpose
  // without the real body to mirror.
  destroyAllGhostsOfRealId(id);
  const b = bodyById.get(id);
  if (b) {
    try { world.removeRigidBody(b); } catch {}
    bodyById.delete(id);
  }
  colliderById.delete(id);
}

// ── Wrap-boundary ghost helpers ──────────────────────────────────────
// The 8 possible wrap offsets. Sig is a stable string key so we can
// reuse ghost bodies across substeps (creating + destroying every step
// would be expensive at scale).
const _GHOST_EDGE_SIGS = ['L', 'R', 'T', 'B', 'LT', 'LB', 'RT', 'RB'];

function _ghostOffsetFor(sig, W, H) {
  switch (sig) {
    case 'L':  return { dx:  W, dy:  0 };
    case 'R':  return { dx: -W, dy:  0 };
    case 'T':  return { dx:  0, dy:  H };
    case 'B':  return { dx:  0, dy: -H };
    case 'LT': return { dx:  W, dy:  H };
    case 'LB': return { dx:  W, dy: -H };
    case 'RT': return { dx: -W, dy:  H };
    case 'RB': return { dx: -W, dy: -H };
  }
  return { dx: 0, dy: 0 };
}

function _effectiveGhostMargin(entities) {
  // The margin must cover the OTHER body's radius too: for a cross-wrap
  // contact A↔B, distance through wrap = (A's distance from edge)
  // + (B's distance from opposite edge), which has to be ≤ A.r + B.r
  // for them to touch. So both bodies must be within max(other radius)
  // of their respective edges. We use max-radius-in-scene + 30 px
  // safety so the dynamic margin scales with the actual bodies.
  let maxR = 0;
  for (const e of entities) {
    if (e.absorbing) continue;
    if (e.radius > maxR) maxR = e.radius;
  }
  return Math.max(GHOST_MARGIN_PX_FLOOR, maxR + 30);
}

function _neededEdgeSigs(realPos, realRadius, viewport, margin) {
  const W = viewport.width, H = viewport.height;
  const reach = realRadius + margin;
  const nearL = realPos.x < reach;
  const nearR = realPos.x > W - reach;
  const nearT = realPos.y < reach;
  const nearB = realPos.y > H - reach;
  const sigs = [];
  if (nearL) sigs.push('L');
  if (nearR) sigs.push('R');
  if (nearT) sigs.push('T');
  if (nearB) sigs.push('B');
  if (nearL && nearT) sigs.push('LT');
  if (nearL && nearB) sigs.push('LB');
  if (nearR && nearT) sigs.push('RT');
  if (nearR && nearB) sigs.push('RB');
  return sigs;
}

function _createGhostBodyFor(e, sig, viewport) {
  const realBody = bodyById.get(e.id);
  if (!realBody) return null;
  const pos = realBody.translation();
  const vel = realBody.linvel();
  const { dx, dy } = _ghostOffsetFor(sig, viewport.width, viewport.height);

  // Dynamic — the ghost must be able to RECEIVE contact impulses so
  // we can forward the resulting velocity delta back to the real body.
  // Kinematic ghosts wouldn't pick up impulses and the forwarding
  // mechanism would degenerate.
  const desc = RAPIER.RigidBodyDesc.dynamic();
  desc.setTranslation(pos.x + dx, pos.y + dy);
  desc.setLinvel(vel.x, vel.y);
  // Ghosts must not sleep — their state is overwritten every substep
  // from the real body, so the "is this still moving" question is
  // meaningless for them. Disabling sleep avoids Rapier deciding to
  // skip integration for a ghost we're actively syncing.
  desc.setCanSleep(false);
  desc.lockRotations();
  const ghost = world.createRigidBody(desc);

  // userData encoding for the dump's contact-event trace:
  //   real body:  userData = e.id (non-negative integer)
  //   ghost body: userData = -(realId + 1)  (negative; subtract 1 to
  //                                          recover the realId)
  // This lets the analyzer tell them apart by sign and trace contacts
  // back to the real body that the ghost mirrors.
  ghost.userData = -(e.id + 1);
  ghost._realId = e.id;
  ghost._ghostSig = sig;

  // Same shape / mass / restitution as the real body (so contact
  // dynamics are physically equivalent). Friction stays 0 — wrap
  // semantics don't introduce friction. Sensor flag mirrors the
  // real collider so a black-hole ghost remains a sensor.
  const r = Math.max(e.radius, 0.01);
  const density = e.mass / (Math.PI * r * r);
  let cDesc = RAPIER.ColliderDesc.ball(r)
    .setDensity(density)
    .setFriction(0)
    .setRestitution(state.elasticRestitution)
    .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)
    // Ghost — collides ONLY with REAL bodies, never with other ghosts.
    // (Ghost-vs-ghost contact would be a topological artifact since both
    // mirror the same underlying matter via different offsets.)
    .setCollisionGroups(COLLISION_GROUP_GHOST);
  if (e.type === 'black_hole') cDesc = cDesc.setSensor(true);
  world.createCollider(cDesc, ghost);

  return ghost;
}

function destroyAllGhostsOfRealId(realId) {
  const myGhosts = ghostsByRealId.get(realId);
  if (!myGhosts) return;
  for (const ghost of myGhosts.values()) {
    try { world.removeRigidBody(ghost); } catch {}
  }
  myGhosts.clear();
  ghostsByRealId.delete(realId);
}

function destroyAllGhosts() {
  for (const realId of [...ghostsByRealId.keys()]) {
    destroyAllGhostsOfRealId(realId);
  }
}

// Per-substep ghost lifecycle. Run after syncWorldToEntities — the
// real bodies must already exist (created or updated for meta drift)
// before we mirror them. Also stashes each ghost's pre-step linvel on
// the ghost itself so the post-step impulse-forwarding phase can
// compute Δv = ghost.linvel_after - ghost._preStepVel and apply it
// to the real body.
function syncGhosts(entities, viewport, boundaryMode) {
  if (boundaryMode !== 'wrap') {
    destroyAllGhosts();
    _lastSyncedViewportW = 0;
    _lastSyncedViewportH = 0;
    return;
  }
  const W = viewport.width, H = viewport.height;
  if (W <= 0 || H <= 0) return;
  // Viewport-size change invalidates all cached ghost positions (offsets
  // computed from W/H at last sync). Force-destroy so the next pass
  // through this function rebuilds at the new W/H. Sleeping bodies'
  // ghosts would otherwise retain stale offsets across resize.
  if (W !== _lastSyncedViewportW || H !== _lastSyncedViewportH) {
    destroyAllGhosts();
    _lastSyncedViewportW = W;
    _lastSyncedViewportH = H;
  }
  // Margin scales with the largest radius currently in play so even
  // big bodies have ghosts spawned before they could physically touch
  // a partner across the wrap edge.
  const margin = _effectiveGhostMargin(entities);
  // Pinned bodies (kinematic-position-based real bodies) still get
  // ghosts: even though applyImpulse is a no-op on kinematic bodies
  // (so the pinned body never moves from cross-wrap contacts, which is
  // correct), the GHOST being dynamic means OTHER bodies still bounce
  // off the ghost-of-pinned-wall through the edge. Dropping pinned
  // ghosts as an optimization would break cross-wrap contact with
  // pinned walls.
  for (const e of entities) {
    // Absorbing entities have no Rapier body — and shouldn't be mirrored.
    if (e.absorbing) {
      destroyAllGhostsOfRealId(e.id);
      continue;
    }
    const realBody = bodyById.get(e.id);
    if (!realBody) continue;
    // Sleeping-body skip: the real body's pos/vel haven't changed since
    // last syncGhosts, so any existing ghosts already mirror the correct
    // (frozen) state. Skip the translation/linvel reads + per-ghost
    // setTranslation/setLinvel writes — one WASM call (isSleeping)
    // avoids 2 reads + 2 writes per ghost. In dense piles where ~70 %
    // of bodies are at rest this is a substantial win.
    //
    // Edge classification (_neededEdgeSigs) is also skipped: if the
    // body wasn't moving last substep it isn't moving now → the same
    // edges remain needed. Create/destroy decisions stay stable.
    //
    // The wake path (recordPostStep → next-substep impulse) re-runs
    // this loop and takes the non-sleeping branch, re-syncing the
    // ghosts the substep the body wakes.
    if (realBody.isSleeping()) continue;
    // Known limitation (Stage 5 reviewer HIGH, accepted as bounded):
    // a body waking from sleep with vel ≈ pre-sleep vel may take the
    // skip-sync branch below if its cumulative delta is under threshold.
    // In that case, _preStepVx on its ghosts retains the pre-sleep
    // baseline, and the next contact's forwarded impulse double-counts
    // by at most (sleep_threshold + gravity_per_substep) ≈ 6 px/s × mass.
    // At default mass=100 this is ≤ 600 momentum/event — well below
    // visual threshold. forwardGhostImpulses' post-update logic (Stage
    // 5c) prevents accumulation beyond one event.
    const pos = realBody.translation();
    const vel = realBody.linvel();
    const needed = _neededEdgeSigs(pos, e.radius, viewport, margin);

    let myGhosts = ghostsByRealId.get(e.id);
    if (!myGhosts) {
      if (needed.length === 0) continue;  // common case: not near any edge
      myGhosts = new Map();
      ghostsByRealId.set(e.id, myGhosts);
    }
    const neededSet = new Set(needed);

    // Stage 5 skip-sync decision: if real body's cumulative drift since
    // the last actual sync stays under both thresholds, existing ghosts'
    // WASM-side state is still close enough — skip the per-ghost
    // setTranslation/setLinvel round-trips. We still iterate to create/
    // destroy ghosts whose needed-edge status changed (cheap JS work).
    //
    // _preStepVx/Vy is NOT updated here when skipping — that path is
    // owned by forwardGhostImpulses, which sets it after each world.step
    // so the next forward's delta computation has the correct baseline
    // regardless of whether this substep re-synced or not.
    let syncExistingGhosts = true;
    if (realBody._lastSyncedX !== undefined) {
      const dx  = pos.x - realBody._lastSyncedX;
      const dy  = pos.y - realBody._lastSyncedY;
      const dvx = vel.x - realBody._lastSyncedVx;
      const dvy = vel.y - realBody._lastSyncedVy;
      if (dx * dx + dy * dy <= GHOST_SYNC_POS_DELTA_SQ &&
          dvx * dvx + dvy * dvy <= GHOST_SYNC_VEL_DELTA_SQ) {
        syncExistingGhosts = false;
      }
    }

    // Update or create ghosts for required offsets
    for (const sig of needed) {
      const off = _ghostOffsetFor(sig, W, H);
      let ghost = myGhosts.get(sig);
      if (!ghost) {
        ghost = _createGhostBodyFor(e, sig, viewport);
        if (ghost) {
          myGhosts.set(sig, ghost);
          // New ghost was created at real body's current pos/vel —
          // baseline matches; stamp _preStepVx so forward computes
          // delta=0 if no contact next step.
          ghost._preStepVx = vel.x;
          ghost._preStepVy = vel.y;
        }
      } else if (syncExistingGhosts) {
        // Slave the ghost's pose + velocity to the real body. P1/P2
        // exception M6: ghost bodies are broadphase proxies, not
        // simulation participants — their state IS by construction
        // the real body's state translated by the wrap offset.
        _vec2A.x = pos.x + off.dx;
        _vec2A.y = pos.y + off.dy;
        ghost.setTranslation(_vec2A, false);
        _vec2A.x = vel.x;
        _vec2A.y = vel.y;
        ghost.setLinvel(_vec2A, false);
        ghost._preStepVx = vel.x;
        ghost._preStepVy = vel.y;
      }
      // Else: existing ghost retains its post-impulse state from the
      // previous substep. forwardGhostImpulses keeps _preStepVx in
      // lockstep with the actual ghost.linvel.
    }

    if (syncExistingGhosts) {
      realBody._lastSyncedX  = pos.x;
      realBody._lastSyncedY  = pos.y;
      realBody._lastSyncedVx = vel.x;
      realBody._lastSyncedVy = vel.y;
    }
    // Destroy ghosts no longer needed (real body drifted away from an edge)
    for (const [sig, ghost] of myGhosts) {
      if (!neededSet.has(sig)) {
        try { world.removeRigidBody(ghost); } catch {}
        myGhosts.delete(sig);
      }
    }
    if (myGhosts.size === 0) ghostsByRealId.delete(e.id);
  }
  // GC: any ghost whose real entity is gone or absorbing.
  const liveIds = new Set(entities.filter(e => !e.absorbing).map(e => e.id));
  for (const realId of [...ghostsByRealId.keys()]) {
    if (!liveIds.has(realId)) destroyAllGhostsOfRealId(realId);
  }
}

// After world.step, each ghost has a velocity delta = (linvel_after
// minus the linvel we set it to at sync time). That delta represents
// the contact impulses Rapier applied to the ghost during this step.
// Forwarding it to the real body as an applyImpulse(delta × mass)
// gives the real body the contact response it would have received if
// Rapier knew about toroidal topology natively. The delta excludes
// any gravity (we never apply gravity to ghosts) so it isolates the
// cross-wrap contact contribution cleanly.
function forwardGhostImpulses() {
  for (const myGhosts of ghostsByRealId.values()) {
    for (const ghost of myGhosts.values()) {
      const v = ghost.linvel();
      const dvx = v.x - (ghost._preStepVx || 0);
      const dvy = v.y - (ghost._preStepVy || 0);
      // Stage 5: update _preStepVx/Vy to current ghost.linvel BEFORE
      // the early-return so the baseline for next substep is correct
      // even when no impulse was applied. Required because Stage 5a's
      // skip-sync optimization leaves ghost.linvel unchanged across
      // substeps — without this update, the NEXT substep's forward
      // would compute its delta against a now-stale baseline and
      // double-count any impulse received here.
      ghost._preStepVx = v.x;
      ghost._preStepVy = v.y;
      if (dvx === 0 && dvy === 0) continue;
      const realBody = bodyById.get(ghost._realId);
      if (!realBody) continue;
      const m = ghost.mass();
      // applyImpulse: Δv_real = impulse / mass_real. Ghost and real
      // share mass (same density × area), so passing (dv * m_ghost)
      // delivers exactly the velocity delta the ghost experienced.
      _vec2A.x = dvx * m;
      _vec2A.y = dvy * m;
      realBody.applyImpulse(_vec2A, true);
    }
  }
}

function syncWorldToEntities(entities) {
  const seenIds = new Set();
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) {
      // Same reasoning as planck: absorbing entities must be removed from
      // Rapier so they don't physically block the predator from reaching
      // them. Entity stays in state.entities until updateAbsorptions splices.
      if (bodyById.has(e.id)) destroyBody(e.id);
      continue;
    }
    seenIds.add(e.id);
    let b = bodyById.get(e.id);
    if (!b) {
      createBodyForEntity(e);
    } else {
      // Detect "physics-baked" meta drift from a live UI slider edit:
      // mass / radius / type all require collider rebuild because Rapier
      // has no safe in-place setter for these. We pull current Rapier
      // pos/vel into the entity FIRST (so the rebuild inherits the
      // body's current trajectory, not the stale JS mirror), then
      // destroy + recreate. The new body picks up the changed meta in
      // createBodyForEntity. This is the canonical live-edit path —
      // it is the ONLY context where destroy+recreate is allowed for a
      // non-wrap, non-absorbing entity.
      if (e.mass   !== e._bakedMass ||
          e.radius !== e._bakedRadius ||
          e.type   !== e._bakedType) {
        const pos = b.translation();
        const vel = b.linvel();
        e.x  = pos.x;  e.y  = pos.y;
        e.vx = vel.x;  e.vy = vel.y;
        // Clear any in-flight spawn-damping burst — the body is being
        // rebuilt and the new body must start with a clean damping
        // state. createBodyForEntity is called with isRebuild=true so
        // it won't re-arm the burst even if the new (resized) body
        // overlaps a neighbor.
        if (e._spawnDampingSubstepsLeft !== undefined) {
          delete e._spawnDampingSubstepsLeft;
        }
        destroyBody(e.id);
        createBodyForEntity(e, /* isRebuild */ true);
        continue;  // skip the pin-toggle branch below — fresh body already has correct type
      }
      const wantKinematic = !!e.pinned;
      const isKinematic = b.bodyType() === RAPIER.RigidBodyType.KinematicPositionBased;
      if (wantKinematic !== isKinematic) {
        b.setBodyType(
          wantKinematic ? RAPIER.RigidBodyType.KinematicPositionBased
                        : RAPIER.RigidBodyType.Dynamic,
          true /* wake */,
        );
        if (wantKinematic) {
          _vec2A.x = 0; _vec2A.y = 0;
          b.setLinvel(_vec2A, true);
        }
      }
    }
  }
  for (const id of [...bodyById.keys()]) {
    if (!seenIds.has(id)) destroyBody(id);
  }
}

// Lightweight per-substep cleanup. Two responsibilities:
//   (1) Destroy Rapier bodies for entities that became absorbing during
//       this substep (set by detectAndStartBHAbsorptions). The body must
//       go so it doesn't physically block the predator from reaching the
//       prey during the absorption animation.
//   (2) Destroy orphan Rapier bodies whose entities were removed from the
//       JS array since the last sync — happens when applyBoundary spliced
//       out-of-bounds bodies in destroy mode, or updateAbsorptions spliced
//       a completed absorption.
// Cost: O(N + bodies). Pure JS-side Set construction + Map iteration,
// no WASM calls in the common case (only when destroy is needed).
function destroyAbsorbingAndOrphans(entities) {
  const liveIds = new Set();
  for (const e of entities) {
    liveIds.add(e.id);
    if (e.absorbing && bodyById.has(e.id)) destroyBody(e.id);
  }
  for (const id of [...bodyById.keys()]) {
    if (!liveIds.has(id)) destroyBody(id);
  }
}

// Spawn-damping resolution loop. Bodies that spawned INTO existing
// overlap carry e._spawnDampingSubstepsLeft. Each substep we:
//   1. Query Rapier: is the body STILL overlapping anything? If
//      not → CLEAN EXIT — setLinvel(0, 0), restore damping=0,
//      clear counter. This is the v_rel = 0 guarantee per spec.
//   2. Still overlapping → tick the safety counter. At 0 we
//      release damping but DO NOT touch velocity (cap is reserved
//      for pathological "can't separate" cases).
// Bodies spawned in empty space never had the field set and are
// skipped via the === undefined fast path. Absorbing entities skip
// early — their Rapier body was already destroyed.
//
// Stage 4: extracted from syncWorldToEntities so it runs ONCE per
// substep instead of twice (the previous double-call meant the safety
// counter ticked at 2× the intended rate). Spawn-damping bodies now
// get the documented ~0.5s safety window rather than 0.25s.
function tickSpawnDamping(entities) {
  for (const e of entities) {
    if (e._spawnDampingSubstepsLeft === undefined) continue;
    if (e.absorbing) {
      // The body's Rapier ref is gone; the counter is dead weight.
      // The gate scan already filters by !e.absorbing so this delete
      // is purely for cleanliness (avoid lingering dead properties).
      delete e._spawnDampingSubstepsLeft;
      continue;
    }
    const b = bodyById.get(e.id);
    if (!b) {
      delete e._spawnDampingSubstepsLeft;
      continue;
    }
    const myCollider = colliderById.get(e.id);
    const ownHandle = myCollider ? myCollider.handle : undefined;
    if (ownHandle !== undefined && !_bodyStillOverlapping(b, e, ownHandle)) {
      _vec2A.x = 0; _vec2A.y = 0;
      b.setLinvel(_vec2A, false);
      b.setLinearDamping(0);
      delete e._spawnDampingSubstepsLeft;
      continue;
    }
    e._spawnDampingSubstepsLeft--;
    if (e._spawnDampingSubstepsLeft <= 0) {
      // Cap-reached path. Zero linvel here — matches the clean-exit
      // branch above, gives consistent "spawn body, body stays put"
      // UX. With the 2026-05-25 conditional-stiffness gate, the world
      // ω₀ is clamped to SPAWN_SAFE_STIFFNESS while this body is in
      // resolution, so the corrective velocity is naturally small;
      // this cap-exit is now a safety belt for pathological "cannot
      // separate" geometry rather than a counter to high-stiffness
      // impulse injection.
      _vec2A.x = 0; _vec2A.y = 0;
      b.setLinvel(_vec2A, false);
      b.setLinearDamping(0);
      delete e._spawnDampingSubstepsLeft;
    }
  }
}

// Dynamic CCD reeval. Iterates all live dynamic bodies and toggles
// setCcdEnabled based on the body's CURRENT linvel speed against
// state.overlapBulletThreshold. Pinned (kinematic) bodies and
// absorbing entities are skipped — CCD on kinematic is a no-op and
// absorbing bodies have already been destroyed in syncWorldToEntities.
//
// _isCcdEnabled mirror on the JS body ref short-circuits redundant
// setCcdEnabled calls (a WASM round-trip per body); we only invoke
// the setter when the state actually changes.
//
// Called once per RAF tick (NOT per substep) from prepareFrame, every
// CCD_REEVAL_INTERVAL_FRAMES frames. Worst-case cost at the reeval
// frame is N WASM linvel reads + (toggled-count) WASM CCD writes.
// In a typical orbital scene only ~5-10% of bodies cross the
// threshold per reeval cycle, so writes are sparse.
function reevalCcd(entities) {
  const thr = state.overlapBulletThreshold;
  const thr2 = thr * thr;
  for (const e of entities) {
    if (e.absorbing || e.pinned) continue;
    const b = bodyById.get(e.id);
    if (!b) continue;
    const v = b.linvel();
    const speedSq = v.x * v.x + v.y * v.y;
    const wantCcd = speedSq > thr2;
    if (wantCcd !== b._isCcdEnabled) {
      // Mirror flag is set BEFORE the WASM call so a setter exception
      // doesn't trigger an infinite per-frame retry loop. If the call
      // throws, we still avoid re-trying every CCD_REEVAL_INTERVAL_FRAMES
      // — the mirror reflects "we attempted this" rather than "Rapier
      // confirmed this." Worst case: a single body's CCD state may not
      // match the mirror, but the system stays stable.
      // Code-review HIGH 2026-05-25 (silent-failure-hunter + code-reviewer).
      b._isCcdEnabled = wantCcd;
      try {
        b.setCcdEnabled(wantCcd);
      } catch (err) {
        if (isVerbose()) console.warn('[physics-rapier] setCcdEnabled threw for body', e && e.id, err);
      }
    }
  }
}

function pullBodyStateToEntities(entities) {
  for (let i = 0; i < entities.length; i++) {
    const e = entities[i];
    if (e.absorbing) continue;
    const b = bodyById.get(e.id);
    if (!b) continue;
    // Stage 5b: sleeping bodies have stable pos/vel in Rapier. e.x/e.y/
    // e.vx/e.vy still match from the last pull (the body hasn't moved
    // since). Trade 2 WASM reads (translation + linvel) for 1 (isSleeping).
    // Net win for sleep-rich scenes (settled piles). Pure orbital scenes
    // pay one extra check per body but bodies rarely sleep anyway, so
    // overhead is bounded.
    if (b.isSleeping()) continue;
    const pos = b.translation();
    const vel = b.linvel();
    e.x  = pos.x;
    e.y  = pos.y;
    e.vx = vel.x;
    e.vy = vel.y;
  }
}

// ── BH absorption (mirror of planck path; Rapier doesn't model it) ──

function detectAndStartBHAbsorptions(entities) {
  const n = entities.length;
  // Fast-exit: if no black hole exists in the scene, the inner double
  // loop produces zero absorptions (every (aBH || bBH) check fails).
  // A single linear scan is O(N); the loop we'd otherwise run is O(N²).
  // At N=200 with a typical pure-planet scene this saves ~40k iterations
  // per substep (~160k/frame at 4 substeps).
  let hasBH = false;
  for (let k = 0; k < n; k++) {
    if (entities[k].type === 'black_hole' && !entities[k].absorbing) {
      hasBH = true;
      break;
    }
  }
  if (!hasBH) return;
  // In wrap mode, use the minimum-image distance so a black hole near
  // one edge can absorb a planet that crossed the opposite edge. This
  // matches the gravity calculation (which also uses min-image) and
  // the ghost-body broadphase setup.
  const wrap = state.boundaryMode === 'wrap';
  const W = wrap ? state.viewport.width  : 0;
  const H = wrap ? state.viewport.height : 0;
  const halfW = W * 0.5;
  const halfH = H * 0.5;
  for (let i = 0; i < n; i++) {
    const a = entities[i];
    if (a.absorbing) continue;
    for (let j = i + 1; j < n; j++) {
      const b = entities[j];
      if (b.absorbing) continue;
      const aBH = a.type === 'black_hole';
      const bBH = b.type === 'black_hole';
      if (!aBH && !bBH) continue;
      let dx = b.x - a.x, dy = b.y - a.y;
      if (wrap) {
        if (dx >  halfW) dx -= W; else if (dx < -halfW) dx += W;
        if (dy >  halfH) dy -= H; else if (dy < -halfH) dy += H;
      }
      const r2 = dx * dx + dy * dy;
      const rSum = a.radius + b.radius;
      if (r2 >= rSum * rSum) continue;
      let prey;
      if (aBH && !bBH) prey = b;
      else if (!aBH && bBH) prey = a;
      else if (a.mass < b.mass) prey = a;
      else if (b.mass < a.mass) prey = b;
      else continue;
      const predator = (prey === a) ? b : a;
      // Shape MUST match physics.js's updateAbsorptions reader contract:
      // { blackHoleId, elapsedSim, startX, startY, startRadius }.
      // Earlier versions wrote { t, duration, predator } which caused
      // updateAbsorptions's `idMap.get(undefined)` lookup to fail → the
      // entity was spliced immediately (no animation). Bug surfaced via
      // /ai-regression-testing R2 test 2026-05-26 — exact "AI fixed one
      // call site, forgot the other contract" pattern from the skill.
      prey.absorbing = {
        blackHoleId: predator.id,
        elapsedSim: 0,
        startX: prey.x,
        startY: prey.y,
        startRadius: prey.radius,
      };
    }
  }
}

// ── Wrap-aware boundary handling ────────────────────────────────────
// Box2D-style broadphases (planck's b2DynamicTree, Rapier's BVH) are
// Euclidean — they have no concept of a wrap boundary. The naive
// applyBoundary teleport from physics.js sets entity.x atomically, then
// syncWorldToEntities pushes the new position into the engine body.
// The engine sees a body suddenly appear next to whatever's on the
// other side and registers it as a new contact pair with stale contact
// state. With planck this produced the dense-edge oscillation bug.
//
// Fix: after wrap-mode applyBoundary runs, identify each entity whose
// position was teleported this substep, then destroy + recreate its
// Rapier body. Re-creation has no contact history, so the engine
// starts the next step with a fresh broadphase entry — no phantom
// solver impulse from a stale pre-wrap contact.

function applyBoundaryAndRebuildOnWrap(entities, viewport, boundaryMode) {
  if (boundaryMode !== 'wrap') {
    applyBoundary(entities, viewport, boundaryMode);
    return [];
  }
  const W = viewport.width, H = viewport.height;
  if (W <= 0 || H <= 0) return [];
  // Reading e.x/e.y here is safe: pullBodyStateToEntities ran immediately
  // before this in step(), so e.x/y equals body.translation() exactly.
  // Absorbing entities are skipped (their JS x/y is mid-animation, not
  // Rapier-authoritative — those bodies were already destroyed earlier).
  const wrappedIds = [];
  for (const e of entities) {
    if (e.absorbing) continue;
    let wrapped = false;
    if (e.x < 0)      { e.x += W; wrapped = true; }
    else if (e.x > W) { e.x -= W; wrapped = true; }
    if (e.y < 0)      { e.y += H; wrapped = true; }
    else if (e.y > H) { e.y -= H; wrapped = true; }
    if (wrapped) wrappedIds.push(e.id);
  }
  // Atomic rebuild: destroy old body+collider, recreate at wrapped position.
  // This severs all contact pairs from the pre-wrap location so the next
  // step's broadphase + narrowphase compute fresh state for this body.
  // Build an id→entity Map once instead of entities.find() per wrap (O(N)
  // per lookup → O(W·N) total; the Map makes it O(N + W)).
  if (wrappedIds.length > 0) {
    const idToEntity = new Map(entities.map(en => [en.id, en]));
    for (const id of wrappedIds) {
      const e = idToEntity.get(id);
      if (!e) continue;
      // Wrap-rebuild MUST pass isRebuild=true to suppress the spawn-
      // damping overlap check. Without this, a body orbiting through
      // the wrap edge would lose 34 %/substep × 4 of its velocity
      // every time it crossed — destroying any wrap-mode orbit
      // (code-review F1).
      destroyBody(id);
      createBodyForEntity(e, /* isRebuild */ true);
    }
  }
  // Note on the 1-substep ghost-gap window: destroyBody removed this
  // body's ghosts and createBodyForEntity does NOT make new ones. The
  // next substep's syncGhosts repopulates ghosts at the new (wrapped)
  // position. Worst case is a single substep (~17 ms wall at default
  // timeScale) where the wrapped body has no ghosts; given the dynamic
  // margin (≥ maxRadius + 30 px) and typical body speeds (≤ 17 px /
  // substep at v=1000 px/s), any cross-wrap contact that would have
  // happened during this window would already have spawned the partner
  // body's ghost on the OPPOSITE edge in the previous substep — so the
  // pair can still be detected when its ghost is re-created. No
  // observable miss in typical use; if this becomes an issue, fold
  // syncGhosts call into the bottom of applyBoundaryAndRebuildOnWrap.
  return wrappedIds;
}

// ── Manifold reader (rapier2d-compat 0.19 API) ──────────────────────
// `world.contactPair(c1, c2)` does NOT exist as a sync return-value API
// in 0.19 — calling it returns undefined and the manifold details were
// silently lost (which is why the earlier dumps had nx/ny/depth=null
// across the board). The real API is callback-style on NarrowPhase:
//
//   world.narrowPhase.contactPair(handleA, handleB, (manifold, flipped) => …)
//
// `manifold.localNormal1()` is the contact normal in collider1's local
// frame, pointing from collider1 → collider2. Our balls have rotations
// locked (createBodyForEntity → desc.lockRotations), so local == world.
//
// When `flipped` is true, the manifold treats handleB as its internal
// collider1, so localNormal1 is pointing from B → A. We negate so the
// returned (nx, ny) is always oriented from A → B regardless of how the
// pair was ordered internally.
//
// Returns { nx, ny, depth } with all three null when no live manifold
// exists for the pair (e.g. when called for an end-of-contact event
// after Rapier has dissolved the manifold).
// Cheaper than readManifold for the dump-off path: only checks whether
// a live manifold exists (numContacts > 0). Skips localNormal1 +
// solverContactDepth reads — those are 2 extra WASM calls per pair
// that we don't need when we're just counting touches for the adaptive
// overlap manager. Returns boolean.
function _hasLiveContact(narrowPhase, handleA, handleB) {
  let touching = false;
  try {
    narrowPhase.contactPair(handleA, handleB, (manifold) => {
      if (manifold && manifold.numContacts() > 0) touching = true;
    });
  } catch {}
  return touching;
}

function readManifold(narrowPhase, handleA, handleB) {
  let nx = null, ny = null, depth = null;
  try {
    narrowPhase.contactPair(handleA, handleB, (manifold, flipped) => {
      if (!manifold || manifold.numContacts() === 0) return;
      const n1 = manifold.localNormal1();
      // Defensive: Rapier can emit numContacts>0 with a degenerate
      // localNormal1=(0,0) when bodies are exactly coincident. A zero
      // vector silently passes the null-check downstream and the
      // offline analyzer would divide by |n|=0 when decomposing v
      // onto the normal. Leave the fields null in that case so the
      // truncation is explicit.
      const mag = Math.hypot(n1.x, n1.y);
      if (mag < 1e-9) return;
      const sign = flipped ? -1 : 1;
      nx = +(sign * n1.x).toFixed(4);
      ny = +(sign * n1.y).toFixed(4);
      depth = +manifold.solverContactDepth(0).toFixed(4);
    });
  } catch {}
  return { nx, ny, depth };
}

// ── Backend factory ──────────────────────────────────────────────────

export function makeRapierBackend() {
  return {
    name: 'rapier',

    async init(entities) {
      if (!RAPIER_READY) {
        await RAPIER.init();
        RAPIER_READY = true;
      }
      world = new RAPIER.World({ x: 0, y: 0 });
      world.lengthUnit = LENGTH_UNIT;
      world.numSolverIterations = SOLVER_ITERATIONS_BASE;
      world.numInternalPgsIterations = PGS_ITERATIONS_BASE;
      world.timestep = 1 / 60;  // matches SIM_DT in state.js

      // Softer TGS-Soft contact spring (vs Rapier default ~377 rad/s)
      // — this is what prevents the "spawn explosion" when a body
      // is created inside another. The conditional spawn-damping
      // burst in createBodyForEntity finishes the job for the
      // overlapping spawn case. See top-of-file constants block for
      // the physics derivation and trade-off analysis.
      //
      // rapier2d-compat 0.19 exposes only setters here (no getters);
      // we wrap in try/catch so an unknown-property runtime failure
      // doesn't bring down the entire init.
      // Initial set; prepareFrame re-applies state.contactStiffness each
      // frame so slider changes take effect on next step.
      try {
        world.integrationParameters.contact_natural_frequency = state.contactStiffness;
      } catch (err) {
        if (isVerbose()) console.warn('[physics-rapier] contact_natural_frequency setter failed:', err);
      }
      try {
        world.integrationParameters.normalizedAllowedLinearError = NORMALIZED_ALLOWED_LINEAR_ERROR;
      } catch (err) {
        if (isVerbose()) console.warn('[physics-rapier] normalizedAllowedLinearError setter failed:', err);
      }

      // All colliders have friction=0 (see createBodyForEntity); Rapier's
      // additional friction iterations (default 4) compute friction
      // constraints we then discard. Set to 0 to skip the pass entirely.
      // Tried with 'in' check to be tolerant of property name drift;
      // logs the result in verbose mode so we can confirm it landed
      // (the silent no-op case is the failure mode the reviewer caught).
      let frictionIterApplied = null;
      try {
        if ('numAdditionalFrictionIterations' in world.integrationParameters) {
          world.integrationParameters.numAdditionalFrictionIterations = 0;
          frictionIterApplied = 'integrationParameters.numAdditionalFrictionIterations';
        } else if ('numAdditionalFrictionIterations' in world) {
          world.numAdditionalFrictionIterations = 0;
          frictionIterApplied = 'world.numAdditionalFrictionIterations';
        }
      } catch (err) {
        if (isVerbose()) console.warn('[physics-rapier] numAdditionalFrictionIterations setter failed:', err);
      }
      if (isVerbose()) {
        if (frictionIterApplied) {
          console.info(`[physics-rapier] friction iterations zeroed via ${frictionIterApplied}`);
        } else {
          console.warn('[physics-rapier] numAdditionalFrictionIterations property not found; friction solver still runs (no perf loss vs baseline but optimization inactive)');
        }
      }

      // Rapier-internal sleep threshold: bring it up to match the JS-side
      // gate (SLEEP_LINEAR_THRESHOLD_PX_PER_S = 5 px/s, normalized by
      // lengthUnit = 30 → 0.167 normalized units). rapier2d-compat 0.19
      // JS bindings expose this as `normalizedLinearThreshold` (the
      // wasm-bindgen camelCase of Rust's `normalized_linear_threshold`).
      // The other candidates are kept as fallbacks for adjacent versions
      // in case the property name drifts. If NONE match, the JS gate
      // (skip applyImpulse when |a·dt| <= threshold) is the only
      // effective lever — still meaningful for piles because it prevents
      // the gravity-wake-cascade entirely.
      const sleepThr = SLEEP_LINEAR_THRESHOLD_PX_PER_S / LENGTH_UNIT;
      const sleepThrSq = sleepThr * sleepThr;
      const sleepSetterCandidates = [
        ['normalizedLinearThreshold',        sleepThr],     // rapier2d-compat 0.19 actual
        ['normalizedLinearAxisThreshold',    sleepThr],     // possible older/newer drift
        ['normalizedLinearAxisThresholdSqr', sleepThrSq],   // sqr variant if present
        ['linearSleepThreshold',             sleepThr],     // unscaled fallback
        ['sleepLinearThreshold',             sleepThr],     // alt naming
      ];
      let sleepSetterApplied = null;
      for (const [prop, val] of sleepSetterCandidates) {
        try {
          if (prop in world.integrationParameters) {
            world.integrationParameters[prop] = val;
            sleepSetterApplied = prop;
            break;
          }
        } catch {}
      }
      if (isVerbose()) {
        if (sleepSetterApplied) {
          console.info(`[physics-rapier] sleep threshold via ${sleepSetterApplied}`);
        } else {
          console.warn('[physics-rapier] no sleep threshold setter matched on integrationParameters; relying on JS gate only');
        }
      }

      // autoDrain=true → drainCollisionEvents() clears the queue after
      // delivering events. Without this we'd accumulate every event
      // forever.
      eventQueue = new RAPIER.EventQueue(true);

      bodyById = new Map();
      colliderById = new Map();
      // Pass Rapier-specific iteration range so the manager's no-contact
      // baseline matches our SOLVER_ITERATIONS_BASE / PGS_ITERATIONS_BASE
      // rather than its planck-era 8/3 default. Without this the manager
      // would silently override our halved baseline on every substep.
      overlapMgr = new AdaptiveOverlapManager(state, {
        velBase: SOLVER_ITERATIONS_BASE,
        velMax:  SOLVER_ITERATIONS_MAX,
        posBase: PGS_ITERATIONS_BASE,
        posMax:  PGS_ITERATIONS_MAX,
      });

      await tryInitGpuGravity();

      // Build initial bodies from any entities that exist at startup
      // (drag-place can spawn before init completes).
      syncWorldToEntities(entities);
    },

    prepareFrame(entities) {
      // Stage 4: full sync runs ONCE per RAF here instead of twice per
      // substep inside step().
      syncWorldToEntities(entities);
      // Conditional contact-stiffness gate (2026-05-25 semantic redesign).
      //
      // Goal: user's stiffness setting applies ONLY to bodies that are
      // pre-existing OR newly-spawned-but-already-resolved. While ANY
      // body still carries `_spawnDampingSubstepsLeft` (active spawn-
      // overlap resolution), the world contact_natural_frequency is
      // clamped to SPAWN_SAFE_STIFFNESS — making spawn-explosion
      // physically impossible regardless of the slider position.
      //
      // The gate is world-wide because Rapier's integration params are
      // global. Trade-off: other pre-existing contacts will feel the
      // safe stiffness during the brief resolution window (typically
      // < 0.5 sec). Acceptable: no visible perturbation, and the gate
      // releases as soon as the spawning body's overlap clears.
      //
      // Absorbing bodies (_spawnDampingSubstepsLeft will be deleted
      // anyway in tickSpawnDamping for them) are skipped here too to
      // avoid keeping the gate latched on a soon-to-be-destroyed body.
      //
      // Cost: O(N) iteration with early exit on first match. At typical
      // scene sizes the loop terminates immediately when nothing is
      // resolving (the common path). One WASM property write per frame.
      //
      // Three regimes:
      //   1. Active resolution (any body has _spawnDampingSubstepsLeft):
      //      stiffness HARD-CLAMPED to min(user, SAFE). Counter re-armed
      //      to GATE_RELEASE_RAMP_FRAMES each such frame, so back-to-back
      //      spawn events keep the gate pinned indefinitely until the
      //      LAST one clears.
      //   2. Ramp-down (just exited resolution): stiffness LERPs from
      //      SAFE → user over GATE_RELEASE_RAMP_FRAMES (~1 sec at 60 fps).
      //   3. Idle (ramp finished): stiffness = user setting.
      let anyResolving = false;
      for (const e of entities) {
        if (e._spawnDampingSubstepsLeft !== undefined && !e.absorbing) {
          anyResolving = true;
          break;
        }
      }
      const safeStiffness = Math.min(state.contactStiffness, SPAWN_SAFE_STIFFNESS);
      let effectiveStiffness;
      if (anyResolving) {
        _gateRampLeft = GATE_RELEASE_RAMP_FRAMES;
        effectiveStiffness = safeStiffness;
      } else if (_gateRampLeft > 0) {
        // Compute t BEFORE decrement so first ramp frame sees t=0
        // (= safe). t = 0 at the moment resolution just ended, →
        // (FRAMES-1)/FRAMES on the final ramp frame, then transitions
        // cleanly to user value on the following frame. The terminal
        // discrete jump is (1/FRAMES) of (user − safe) — negligible
        // (~0.9 Hz at ω₀=60, FRAMES=60).
        const t = 1 - (_gateRampLeft / GATE_RELEASE_RAMP_FRAMES);
        effectiveStiffness = safeStiffness + (state.contactStiffness - safeStiffness) * t;
        _gateRampLeft--;
      } else {
        effectiveStiffness = state.contactStiffness;
      }
      try {
        world.integrationParameters.contact_natural_frequency = effectiveStiffness;
      } catch (err) {
        if (isVerbose()) console.warn('[physics-rapier] prepareFrame: contact_natural_frequency setter failed:', err);
      }
      // Dynamic CCD reeval every CCD_REEVAL_INTERVAL_FRAMES frames.
      // Runs ONCE per RAF — the toggle is for the entire upcoming
      // frame's substep loop.
      if (++_ccdReevalCounter >= CCD_REEVAL_INTERVAL_FRAMES) {
        _ccdReevalCounter = 0;
        reevalCcd(entities);
      }
    },

    async step(entities, dt, viewport, boundaryMode, isLastSubstep) {
      const dumpEnabled = !!state.stateDumpEnabled;

      // Stage 4: full syncWorldToEntities has been moved to prepareFrame
      // (runs once per RAF, not per substep). Inside the substep loop
      // we only do lightweight cleanup at the bottom (see
      // destroyAbsorbingAndOrphans / tickSpawnDamping calls there).
      // bodyById is current at this point; any entity in entities[]
      // has a body, and there are no orphans.

      // Wrap-boundary ghost lifecycle. Must run AFTER bodies are current
      // (so the real bodies it mirrors definitely exist) and BEFORE the
      // gravity impulse loop / world.step (so the ghosts are properly
      // staged for this substep's contact resolution). In non-wrap
      // modes this destroys any leftover ghosts and returns immediately.
      syncGhosts(entities, viewport, boundaryMode);

      // ── state-dump trace capture: pre-state ──────────────────────
      // Read directly from Rapier so the dump's "pre" snapshot is the
      // authoritative pre-step body state, not a JS mirror that lags by
      // one substep. The offline analyzer's
      //   Δv_solver = post.v - pre.v - gravity.a * dt
      // computation requires pre.v to be the velocity Rapier saw at the
      // start of this world.step (before addForce + step). That value
      // lives in body.linvel() at this point, never in e.vx.
      //
      // Gated on stateDumpEnabled: skip the per-entity object allocs
      // entirely when the user isn't recording — at N=1200 this is
      // ~1200 short-lived objects per substep saved.
      const pre = dumpEnabled ? entities.map(e => {
        const b = bodyById.get(e.id);
        if (!b) {
          return {
            id: e.id,
            x: null, y: null, vx: null, vy: null,
            sleeping: null,
          };
        }
        const pos = b.translation();
        const vel = b.linvel();
        return {
          id: e.id,
          x:  +pos.x.toFixed(3),
          y:  +pos.y.toFixed(3),
          vx: +vel.x.toFixed(3),
          vy: +vel.y.toFixed(3),
          sleeping: !!b.isSleeping(),
        };
      }) : null;

      // ── Resolve gravity accels ───────────────────────────────────
      // Sync GPU above N=GPU_THRESHOLD_SYNC; CPU below. The async per-
      // frame pipeline was removed (see GPU_THRESHOLD_SYNC block).
      perfMark('gravityCompute');
      let accels;
      {
        const n = entities.length;
        if (n < 2) {
          accels = new Float32Array(n * 2);
        } else if (gpuGravityHandle && !_gpuDisabled && n >= GPU_THRESHOLD_SYNC) {
          try {
            accels = await computeGravityGPUSync(entities, n);
            _gpuConsecutiveFailures = 0;   // success → reset failure run
          } catch (e) {
            _gpuConsecutiveFailures++;
            if (_gpuConsecutiveFailures >= GPU_FAIL_LATCH_COUNT) {
              _gpuDisabled = true;
              console.warn(`[physics-rapier] GPU gravity disabled after ${_gpuConsecutiveFailures} consecutive failures; CPU latched for session:`, e);
            } else if (isVerbose()) {
              console.warn(`[physics-rapier] sync GPU fallback (failure ${_gpuConsecutiveFailures}/${GPU_FAIL_LATCH_COUNT}); CPU:`, e);
            }
            accels = computeGravityCPU(entities, n, new Float32Array(n * 2));
          }
        } else {
          accels = computeGravityCPU(entities, n, new Float32Array(n * 2));
        }
      }

      // Per-entity gravity vector this substep. forceApplied starts false
      // and is flipped only when applyImpulse actually runs — this lets the
      // offline analyzer distinguish "skipped" from "applied with
      // magnitude 0". Skip reasons match the state-dump.js docstring:
      //   absorbing | pinned | no-body | sleeping+tiny-impulse
      //
      // Gated on stateDumpEnabled: when the user isn't recording, the
      // entire gravity[] (N objects) is skipped. The impulse application
      // loop below ignores the array conditionally.
      const gravity = dumpEnabled ? entities.map((e, i) => ({
        id: e.id,
        ax: +accels[i * 2].toFixed(6),
        ay: +accels[i * 2 + 1].toFixed(6),
        forceApplied: false,
      })) : null;
      perfMark('gravityApply');
      for (let i = 0; i < entities.length; i++) {
        const e = entities[i];
        if (e.absorbing || e.pinned) continue;
        const b = bodyById.get(e.id);
        if (!b) continue;
        const ax = accels[i * 2];
        const ay = accels[i * 2 + 1];
        if (b.isSleeping()) {
          // Compare RAW |a|² (px/s²) — NOT |a·dt|² — against the wake
          // threshold. The Δv-per-substep version (used pre-2026-05-25)
          // mixed units and never woke single-body click-placed scenes
          // because |a|·dt ≈ 0.08 px/s while the 5 px/s threshold is in
          // velocity units. See SLEEP_WAKE_ACCEL_THRESHOLD_SQ block.
          const a2 = ax * ax + ay * ay;
          if (a2 <= SLEEP_WAKE_ACCEL_THRESHOLD_SQ) continue;
          // |a| > threshold — applyImpulse(..., wake=true) below
          // wakes the body atomically inside Rapier.
        }
        // Use applyImpulse, NOT addForce. Empirically rapier2d-compat
        // 0.19's `addForce` does NOT integrate over world.timestep the
        // way its docs claim — observed Δv per substep matches F/m
        // (impulse semantics) rather than F/m × dt (force semantics).
        // Compensating by passing impulse = mass × acceleration × dt
        // gives the correct Newtonian Δv = a × dt per substep. Using
        // applyImpulse (whose impulse → Δv = impulse/mass semantics
        // ARE deterministic per docs) is the cleanest way to express
        // this without depending on addForce's empirical behavior.
        const mdt = e.mass * dt;
        _vec2A.x = mdt * ax;
        _vec2A.y = mdt * ay;
        b.applyImpulse(_vec2A, true);
        if (gravity) gravity[i].forceApplied = true;
      }

      // Adaptive iteration scaling. The manager returns counts within
      // Rapier-specific bounds, but AQM (Adaptive Quality Manager)
      // dynamically tightens the upper cap based on frame-time pressure:
      // if user's machine is struggling to hit target FPS, AQM lowers
      // velIterCap/posIterCap and we clamp here. AQM quality=1.0 →
      // full 8/4 cap; quality=0.0 → tight 4/2 (matches baseline, no
      // escalation room).
      const aqm = getQualityKnobs();
      const [rawVelIter, rawPosIter] = overlapMgr.decideIterations();
      const velIter = Math.min(rawVelIter, aqm.velIterCap);
      const posIter = Math.min(rawPosIter, aqm.posIterCap);
      world.numSolverIterations = velIter;
      world.numInternalPgsIterations = posIter;

      // Pass the eventQueue so Rapier fills it with start/stop events as
      // contacts form and break inside the solver substep. Without it,
      // the only way to see a contact is the post-step contactPairsWith
      // query — which misses brief impacts that have already resolved
      // by the time we look.
      perfMark('rapierStep');
      world.step(eventQueue);

      // Forward cross-wrap contact impulses from ghosts to real bodies.
      // Must run BEFORE pullBodyStateToEntities and BEFORE the contact-
      // events drain (so the contact-events trace reflects the final
      // post-forwarding velocity for any subsequent analysis pass).
      // No-op in non-wrap modes since ghostsByRealId is empty.
      forwardGhostImpulses();

      // ── Drain contact events ─────────────────────────────────────
      // For each event we record: which entities, started-vs-ended, and
      // the body's velocity at drain time. Rapier does NOT expose the
      // velocity at the moment the event was emitted inside world.step —
      // every event in this slot shares the same post-step velocity for
      // its body. Use the adjacent slots' pre[]/post[] entries (by id)
      // to reconstruct what each body was doing immediately before /
      // after the contact.
      //
      // Body lookup can fail when a participating collider has been
      // destroyed (e.g. wrap-rebuild in the previous substep queued a
      // stop event that drains here); those entries get null ids and
      // zero velocities and are filterable downstream.
      //
      // CONTACT_EVENTS_CAP guards against a pathological dense cluster
      // generating thousands of events in one substep — an unbounded
      // push() loop would spike GC and distort the very timing data
      // we're trying to collect. 512 is ~8× the worst-case I expect
      // (32 contacting pairs × start+end).
      // contactEvents — only build the detail array when state-dump is
      // ON. The drain MUST run regardless (otherwise the eventQueue
      // accumulates events across substeps unbounded), but with no
      // recording the callback is a cheap no-op.
      const CONTACT_EVENTS_CAP = 512;
      const contactEvents = dumpEnabled ? [] : null;
      let contactEventsTruncated = false;
      const eventNarrowPhase = world.narrowPhase;
      if (dumpEnabled) {
        eventQueue.drainCollisionEvents((handle1, handle2, started) => {
          if (contactEvents.length >= CONTACT_EVENTS_CAP) {
            contactEventsTruncated = true;
            return;
          }
          const c1 = world.getCollider(handle1);
          const c2 = world.getCollider(handle2);
          const aBody = c1 ? c1.parent() : null;
          const bBody = c2 ? c2.parent() : null;
          const aId = aBody ? aBody.userData : null;
          const bId = bBody ? bBody.userData : null;
          let aVx = 0, aVy = 0, bVx = 0, bVy = 0;
          if (aBody) { const v = aBody.linvel(); aVx = v.x; aVy = v.y; }
          if (bBody) { const v = bBody.linvel(); bVx = v.x; bVy = v.y; }
          const md = readManifold(eventNarrowPhase, handle1, handle2);
          contactEvents.push({
            aId, bId, started: !!started,
            aVx: +aVx.toFixed(3), aVy: +aVy.toFixed(3),
            bVx: +bVx.toFixed(3), bVy: +bVy.toFixed(3),
            nx: md.nx, ny: md.ny, depth: md.depth,
          });
        });
      } else {
        eventQueue.drainCollisionEvents(() => {});  // drain only, no alloc
      }

      // Contact-iteration pass: counts truly-touching pairs (live
      // manifold with numContacts > 0) for the adaptive overlap manager
      // AND extracts manifold details (normal, depth) for the state-dump
      // trace. Dedup'd by sorted handle pair.
      //
      // CRITICAL: contactPairsWith enumerates broadphase-persistent pairs,
      // which persist for several substeps after physical separation. The
      // `md.nx === null` filter (readManifold confirmed numContacts() > 0)
      // drops stale broadphase ghosts.
      //
      // Per-substep cost: O(colliders × avg pairs) × 1 WASM contactPair
      // call per pair. At N=200 with moderate density this is ~1000 WASM
      // round-trips per substep — the single largest substep cost at
      // scale. We gate the iteration on (isLastSubstep || dumpEnabled):
      //   - dump on  → run every substep (trace fidelity)
      //   - dump off → run only on last substep of the frame; the
      //     AdaptiveOverlapManager's iteration-count decision uses the
      //     previous frame's last-substep touchingCount for all substeps
      //     of this frame. Slight lag, bounded by 1 frame (~16 ms), and
      //     the dense-contact escalator's hysteresis already absorbs it.
      // Iter 2 split: dump path needs full manifold detail (trace); the
      // hot adaptive-overlap path only needs a count. Plus sample every
      // Nth frame when dump-off so the average per-frame cost amortizes.
      const contactsTrace = dumpEnabled ? [] : null;
      let touchingCount = 0;
      const isLastDumpSubstep = isLastSubstep && dumpEnabled;
      // AQM controls sample period: tighter sampling at high quality
      // (every 3 frames), sparser when struggling (every 10).
      const sampleN = aqm.contactSampleEveryN;
      const isLastCountSubstep = isLastSubstep && !dumpEnabled &&
        (++_contactSampleCounter >= sampleN);
      if (isLastCountSubstep) _contactSampleCounter = 0;
      if (isLastDumpSubstep || isLastCountSubstep) {
        perfMark('contactsTrace');
        const seenPairs = _contactCounterSeen;
        seenPairs.clear();
        const narrowPhase = world.narrowPhase;
        if (isLastDumpSubstep) {
          // Full path — needs normal + depth for the dump trace.
          world.forEachCollider(c => {
            const aHandle = c.handle;
            const aBody = c.parent();
            const aId = aBody ? aBody.userData : null;
            world.contactPairsWith(c, (other) => {
              const bHandle = other.handle;
              const key = aHandle < bHandle
                ? aHandle * 0x100000000 + bHandle
                : bHandle * 0x100000000 + aHandle;
              if (seenPairs.has(key)) return;
              seenPairs.add(key);
              const bBody = other.parent();
              const bId = bBody ? bBody.userData : null;
              const md = readManifold(narrowPhase, aHandle, bHandle);
              if (md.nx === null) return;  // broadphase-persistent ghost
              touchingCount++;
              contactsTrace.push({ aId, bId, nx: md.nx, ny: md.ny, depth: md.depth });
            });
          });
        } else {
          // Fast path — only checks live-manifold existence.
          // ~2× faster than readManifold (2 WASM calls per pair vs 4).
          world.forEachCollider(c => {
            const aHandle = c.handle;
            world.contactPairsWith(c, (other) => {
              const bHandle = other.handle;
              const key = aHandle < bHandle
                ? aHandle * 0x100000000 + bHandle
                : bHandle * 0x100000000 + aHandle;
              if (seenPairs.has(key)) return;
              seenPairs.add(key);
              if (_hasLiveContact(narrowPhase, aHandle, bHandle)) touchingCount++;
            });
          });
        }
        overlapMgr.recordPostStep(touchingCount);
        // Feed perf-monitor with last-substep context. Awake-count scan
        // is gated on perfEnabled() so we don't pay N isSleeping() WASM
        // calls per substep when the monitor isn't running.
        if (perfEnabled()) {
          let awakeCount = 0;
          for (const b of bodyById.values()) {
            if (!b.isSleeping()) awakeCount++;
          }
          perfSetContext({
            awake: awakeCount,
            contacts: touchingCount,
            iters: { vel: velIter, pos: posIter },
            gpuOn: !!gpuGravityHandle && entities.length >= GPU_THRESHOLD_SYNC,
          });
        }
      }
      // When we skipped: manager keeps its previous count → decideIterations
      // returns the same iter values as last substep. The decideIterations
      // → world.numSolverIterations assignment at the top of NEXT substep
      // will see the unchanged state and continue at the prior baseline.

      perfMark('pullPost');
      pullBodyStateToEntities(entities);

      detectAndStartBHAbsorptions(entities);
      updateAbsorptions(entities, dt);
      const wrappedIds = applyBoundaryAndRebuildOnWrap(entities, viewport, boundaryMode);
      // Stage 4: lightweight cleanup replaces the second syncWorldToEntities.
      //   destroyAbsorbingAndOrphans — destroys bodies for entities flagged
      //     absorbing this substep + orphans from applyBoundary/splice
      //   tickSpawnDamping — per-substep tick for spawn-overlap resolution
      // No meta-drift detection here (already handled in prepareFrame);
      // UI events that change meta fire between RAFs.
      destroyAbsorbingAndOrphans(entities);
      tickSpawnDamping(entities);

      // state-dump trace recording (conditional on stateDumpEnabled).
      if (dumpEnabled) {
        const post = entities.map(e => {
          const b = bodyById.get(e.id);
          if (b) {
            const pos = b.translation();
            const vel = b.linvel();
            return {
              id: e.id,
              x:  +pos.x.toFixed(3),
              y:  +pos.y.toFixed(3),
              vx: +vel.x.toFixed(3),
              vy: +vel.y.toFixed(3),
              sleeping: !!b.isSleeping(),
            };
          }
          return {
            id: e.id,
            x:  +e.x.toFixed(3),
            y:  +e.y.toFixed(3),
            vx: +(e.vx || 0).toFixed(3),
            vy: +(e.vy || 0).toFixed(3),
            sleeping: null,
          };
        });
        recordSubstep({
          dt,
          solverIters: {
            velocity: world.numSolverIterations,
            pgs: world.numInternalPgsIterations,
          },
          pre,
          gravity,
          post,
          contacts: contactsTrace,
          contactEvents,
          contactEventsTruncated,
          wrappedEntityIds: wrappedIds,
        });
      }
    },

    onEntityMetaMaybeChanged() {
      // No work needed: syncWorldToEntities polls each substep and
      // detects mass / radius / type drift via the baked markers
      // stamped in createBodyForEntity. UI code is free to mutate
      // e.mass / e.radius / e.type at any time; the rebuild happens
      // automatically on the next substep before world.step runs.
    },

    destroy() {
      destroyAllGhosts();
      // Reset CCD reeval cadence + viewport-change tracker + GPU
      // crossover probe state so a fresh init() starts deterministic.
      _ccdReevalCounter = 0;
      _lastSyncedViewportW = 0;
      _lastSyncedViewportH = 0;
      _gpuProbeSamples.length = 0;
      _gpuDisabled = false;
      _gpuConsecutiveFailures = 0;
      _contactSampleCounter = 0;
      // HIGH (code-review): without this, a "清空沙盘" / re-init mid-ramp
      // leaves the new session reading interpolated stiffness for the
      // first ~60 frames before the counter winds down.
      _gateRampLeft = 0;
      resetQuality();
      for (const id of [...bodyById.keys()]) destroyBody(id);
      bodyById.clear();
      colliderById.clear();
      if (overlapMgr) { overlapMgr.reset(); overlapMgr = null; }
      if (eventQueue) { try { eventQueue.free(); } catch {} eventQueue = null; }
      if (world) { try { world.free(); } catch {} world = null; }
      if (gpuGravityHandle) { try { gpuGravityHandle.destroy(); } catch {} gpuGravityHandle = null; }
      if (gpuDevice)        { try { gpuDevice.destroy();        } catch {} gpuDevice        = null; }
    },
  };
}

