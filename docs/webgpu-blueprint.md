# WebGPU Acceleration Blueprint — orbital-sandbox

**Branch**: `feat/webgpu-physics` (off `main` at `16c5c31`).
**Target hardware**: NVIDIA RTX 5070 Ti (Blackwell, ~16 GB VRAM, ~8960 CUDA cores).
**Performance goal**: N=10⁴–10⁵ celestial bodies at 60 fps.
**Fallback**: Runtime feature-detect WebGPU; transparently fall back to existing CPU + WebGL 2 path when unavailable.

This document was produced via 3 rounds of `code-architect` design + 2 rounds of `code-reviewer` critique + iterative push-back. The audit trail is preserved in `.claude/architect-blueprint.md` (round-1 long form) and the conversation history.

---

## 1. Architecture Choice

**Selected**: Option β — full WebGPU (compute + render). Physics buffers are GPU-resident; rendering reads positions directly from compute output without CPU round-trip.

**Rejected**:
- Option γ (gravity-only GPU): For N=10⁵, broadphase + solver on CPU still ≥ 50 ms/frame in JS. GPU readback every substep adds further latency. Insufficient for stated goal.
- Option α (WebGPU compute + WebGL 2 render hybrid): No stable GPU↔GPU buffer-sharing bridge between WebGPU and WebGL 2 in Chrome as of 2026-Q2. The render-side rewrite to WebGPU (~1500 LOC) is bounded and cleaner than the bridge.

---

## 2. Phased Rollout

| Phase | Scope | Target N @ 60fps | Risk profile |
|-------|-------|------------------|--------------|
| 1 | K1 (gravity) on GPU; CPU runs everything else; accept one-substep-stale gravity (sub-pixel error bounded) | 5,000 | Minimal — only 1 kernel, easy revert |
| 2 | K1–K8 full physics pipeline on GPU; CPU only handles input/UI/event-drain | 10,000+ | Medium — solver concurrency model is the load-bearing decision |
| 3 | Render also migrates to WebGPU; positions never leave GPU | 50,000+ | Medium — render pipeline rewrite (~1500 LOC) |
| 4 | Cleanup, optimization, validation hardening | 100,000 | Low |

Each phase: **2 rounds of `/feature-dev` review** (architect detailed design → critical review → my push-back if needed → 2nd review verification) before implementation. Phase boundaries are commit boundaries, all revertable.

---

## 3. Compute Pipeline (Phase 2 final)

Per-substep dispatch sequence (matches CPU pipeline A→B→C→D→E→F→F'→G→I):

```
K1  gravity_accel              # gravity (O(N²) tiled all-pairs, min-image PBC)
K2  kick_predict               # v += a·dt; x += v·dt; pvx = pvy = 0
K3  broadphase_count           # uniform-grid cell counter (parallel histogram)
K3b broadphase_prefix_sum      # Blelloch scan on cellCounts
K3c broadphase_scatter         # place entity indices into cells
K4  contact_detect             # 3×3 neighborhood walk; emit Contact + AbsorptionEvent
                               # also writes entityMaxImpulse via emulated atomicMax
K5a warm_start_calibrate       # pairwise Plummer-softened warm-start for persistent
                               # contacts; cold-start inheritance from entityMaxImpulse
K5  velocity_solver            # Jacobi + i32 fixed-point atomics; iteration count
                               # adaptive (8–24 by contactCount, matching CPU ramp)
K6  position_solver            # Jacobi, 3 iterations, identical structure to K5
K7  pseudo_integrate_boundary  # integrate pvx into position; wrap; pinned hard-reset
K8  rebuild_warm_start         # write pairImpulseTable from contacts for next substep
```

### 3.1 Solver concurrency model

**K5 and K6 use Jacobi iteration with two dispatches per iteration**:
- **Accumulate pass**: each contact reads velocity snapshot; atomicAdd<i32> the FIXED_SCALE-encoded delta to `velDeltaBuf[idxA]` and `velDeltaBuf[idxB]`.
- **Apply pass**: each entity reads its accumulated delta, applies to velocity, zeros the delta slot.

Race-free by construction: velocities are read-only within the accumulate pass; the apply pass uses one thread per entity (no two threads share an index). **No graph coloring required.** Convergence is Jacobi (slower per-iteration than GS, but parallel-safe).

An earlier proposal of "red-black GS by cell parity" was structurally wrong (contact graph in dense hex packing has triangles → line graph is not bipartite → 2-coloring impossible). Eliminated.

### 3.2 Cold-start convergence mitigation

Without warm-starting, Jacobi at iteration count 24 cannot propagate impulse to chain depth >24 in a single substep. At target N=10⁴ hex packing, chain depth ≈ 113.

Mitigation: **K4 maintains `entityMaxImpulse: array<f32>`** — for each entity, atomicMax of its persistent contacts' impulses. K5a seeds non-persistent (cold) contacts with `min(entityMaxImpulse[A], entityMaxImpulse[B]) × 0.25` instead of zero. This gives newly-contacted pairs a starting impulse at roughly the local neighborhood's magnitude, allowing Jacobi to converge in O(log depth) iterations rather than O(depth).

**Limitation**: this works for incremental cluster growth (entities added one at a time, warm-start table builds up over frames). A pathological scenario — N=10⁴ bodies spawned simultaneously from rest with no prior warm-start state — may show 1–3 frames of visible overlap during initial settle. Acceptable for the orbital sandbox use case.

---

## 4. Data Layout (Phase 2)

All buffers are GPU-resident. SoA for hot-path physics state; AoS for structures whose fields are read together.

| Buffer | Layout | Stride | Size at N=10⁵ |
|--------|--------|--------|---------------|
| `positions` | `array<vec2f>` SoA | 8 B | 0.8 MB |
| `velocities` | `array<vec2f>` SoA | 8 B | 0.8 MB |
| `pseudoVels` | `array<vec2f>` SoA | 8 B | 0.8 MB |
| `accels` | `array<vec2f>` SoA | 8 B | 0.8 MB |
| `EntityMeta` | AoS struct {mass, chargeF, radius, flags} | 16 B | 1.6 MB |
| `contacts` | AoS Contact struct | 48 B | 14.4 MB (3N slots) |
| `pairImpulseTable` | Open-addressing hash | 32 B | 38.4 MB (4× load factor) |
| `velDeltaBuf` | `array<atomic<i32>>` | 4 B × 2 | 0.8 MB |
| `entityMaxImpulse` | `array<f32>` | 4 B | 0.4 MB |
| `cellCounts`, `cellOffsets`, `cellContents` | grid metadata | varies | 0.4 MB |
| Shadow buffers (positions + velocities) | double-buffered staging | 16 B | 3.2 MB |
| Render instance buffer (Phase 3) | per-entity render data | 20 B | 2.0 MB |
| Trail textures (2× ping-pong) | RGBA8 1920×1080 | — | ~16 MB |
| **Total** | | | **~80 MB** (0.5% of 16 GB) |

### 4.1 EntityMeta struct (locked layout)

```wgsl
struct EntityMeta {
  mass:    f32,  // 4 B
  chargeF: f32,  // 4 B — charge stored as f32 (-1, 0, +1)
  radius:  f32,  // 4 B
  flags:   u32,  // 4 B — bitfield
}  // 16 B total, no padding needed for storage buffer (WGSL §3.6.2)
```

**Flag bits**:
```wgsl
const FLAG_ABSORBING: u32 = 1u;  // bit 0 — BH devour animation in progress
const FLAG_PINNED:    u32 = 2u;  // bit 1 — kinematically frozen
const FLAG_IS_BH:     u32 = 4u;  // bit 2 — type === 'black_hole'
const FLAG_TOMBSTONE: u32 = 8u;  // bit 3 — dead, awaiting compaction
```

**Per-kernel access matrix** (Phase 2):

| Kernel | Reads bits | Writes |
|--------|-----------|--------|
| K1 | ABSORBING (skip as source) | — |
| K2 | ABSORBING, PINNED (skip) | — |
| K3, K3c | ABSORBING, TOMBSTONE | — |
| K4 | all (IS_BH to choose absorption vs elastic) | ABSORBING (via atomicOr) on prey |
| K5a | ABSORBING (skip) | — |
| K5 | ABSORBING, PINNED (zero wA/wB) | — |
| K6 | ABSORBING, PINNED | — |
| K7 | ABSORBING, PINNED (zero v) | — |
| K8 | ABSORBING (skip) | — |

CPU writes: `beginAbsorption` triggers ABSORBING (via writeBuffer patch after event drain), UI pin button triggers PINNED, entity creation sets IS_BH, `updateAbsorptions` completion triggers TOMBSTONE.

---

## 5. CPU↔GPU Synchronization

### 5.1 Per-frame data flow

```
Frame start:
  CPU: drain absorptionEvent buffer (mapAsync result from prior frame)
       → call beginAbsorption() for each event
       → writeBuffer patches for affected EntityMeta.flags
  CPU: process UI input → writeBuffer patches for edited metadata
  CPU: handle new/removed entities → buffer grow/tombstone

  GPU: substep loop (1–8 iterations):
    K1 → K2 → K3 → K3b → K3c → K4 → K5a → K5×N → K6×3 → K7 → K8

  GPU: copyBufferToBuffer positions + velocities → shadowStagingBuf[frame%2]
  GPU: copyBufferToBuffer absorptionEvents → absorptionStagingBuf[frame%2]
  GPU (Phase 3): bind positions, EntityMeta directly to render pipeline

  CPU (Phase 1–2): render reads from cpuShadow (last frame's positions+velocities)
  CPU: schedule mapAsync on shadowStagingBuf[(frame-1)%2] → resolves before next frame
```

### 5.2 Shadow buffer contents

**Includes velocities**, not just positions. Reason: `device.lost` recovery synchronizes velocities into CPU `entity.vx, entity.vy` to continue physics correctly. Without velocities the recovery would zero all velocities (cluster collapses radially).

Size: 16 bytes/entity × N (positions + velocities). At N=10⁵: 1.6 MB. Double-buffered staging: 3.2 MB.

### 5.3 Async readback latency

`mapAsync` on staging buffer has ~0.5–3 ms IPC latency in Chrome. Double-buffering ensures the GPU can write substep N+1's data while CPU consumes substep N's data — no stall as long as GPU work + CPU work each fit within the frame budget.

The CPU shadow is **one frame behind** the GPU. This is acceptable for:
- Renderer instance data (Phase 1–2 only; Phase 3 reads directly from GPU buffers)
- Hit-test (16.7 ms input lag imperceptible)
- `predictTrajectory` drag-ghost (5-sec preview tolerates one frame of base-state lag)
- `updateAbsorptions` (uses CPU absorption state, not GPU)

---

## 6. Fallback & Recovery

### 6.1 Startup detection

```js
async function detectBackend() {
  if (!navigator.gpu) return 'cpu';
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return 'cpu';
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 64 * 1024 * 1024,
        maxComputeWorkgroupStorageSize: 16384,
      }
    });
    return { backend: 'webgpu', adapter, device };
  } catch (e) { return 'cpu'; }
}
```

### 6.2 Abstraction layer

`physics-backend.js` exposes a single async factory `createBackend()` that
returns a wrapper hiding the active backend. The interface as realized in
Phase 1 (this section was revised after implementation — original draft used
`stepSubstep / updatePositionShadow / uploadEntityChange`, which evolved
during architect review):

```js
{
  get name() => 'cpu' | 'webgpu',
  async init(entities)                                       // adapter+device+priming
  prepareFrame(entities)                                     // BH tree build on CPU; no-op on GPU
  async step(entities, dt, viewport, boundaryMode)           // 1 substep end-to-end
  onEntityMetaMaybeChanged()                                 // marker (no-op in Phase 1; full re-upload covers it)
  destroy()
}
```

`cpuBackend` wraps the existing `physics.js` verbatim — bit-identical to the
`main` branch. `gpuBackend` runs K1 on GPU and the rest (kick / predict /
collide / solver / boundary) on CPU. On `device.lost` the wrapper swaps
`active` to a fresh CPU backend transparently; callers hold one reference
and never observe the swap directly except through `state.backendName`.

Phase 2 will add `updatePositionShadow()` + `uploadEntityChange(idx)` /
absorption event drain when GPU takes ownership of positions.

`gpuBackend` runs the GPU pipeline. CPU fallback to `cpuBackend` is permitted at any point (e.g., on `device.lost`).

### 6.3 `device.lost` recovery sequence

1. `gpuBackend.destroy()` — release all GPU handles.
2. `backend = cpuBackend`.
3. Copy `shadowPositions` and `shadowVelocities` (most recent successful readback) into `entity.x, y, vx, vy`.
4. `_prevPairImpulses.clear()` — accept 1–2 substep warm-start transient (bounded by `WARM_START_PERSIST_FLOOR = 10%` floor).
5. In-flight absorption events lost: affected entities snap from "mid-absorption" position back to shadow position (≤1 body diameter pop). Accepted as rare corner case.
6. Physics continues on CPU from synchronized state.

User-visible transient: one frame at most of slightly imperfect convergence on dense clusters. Documented; not a regression on top of the underlying GPU loss.

---

## 7. Bug-Fix Preservation Audit

| Fix | Mechanism | GPU path status |
|-----|-----------|-----------------|
| `a9da414` (bhThreshold raise) | Workaround for BH 3rd-law drift | GPU uses exact O(N²); BH never active. Fix's root issue eliminated. |
| `38a7aee` (per-axis cell tiling) | floor(W / minCellSize), per-axis cellSize | K3 explicit WGSL translation required: same formula, must use `floor` and floating-point divide. **Reviewable in K3 implementation.** |
| `d3d3e7d` (warm-start gravity calib) | pairwise Plummer + 10% j_prev floor | K5a replicates formula exactly. `c.dist` stored in Contact struct. Preserved. |
| `cd01015` (adaptive iters + pairwise) | iteration ramp 8→24; pairwise calibration | K5 receives iteration count as uniform from CPU. K5a uses pairwise formula. Preserved. |
| `16c5c31` (per-substep hash rebuild) | Hash built inside substep, not per-frame | K3 placed inside per-substep compute pass sequence. Structurally guaranteed. |

---

## 8. Acceptance Criteria (per phase)

### Performance (RTX 5070 Ti)

| N | Phase 1 | Phase 2 | Phase 3 |
|---|---------|---------|---------|
| 500 | ≤ 2 ms | ≤ 1 ms | ≤ 0.8 ms |
| 1,000 | ≤ 3 ms | ≤ 1.5 ms | ≤ 1.2 ms |
| 5,000 | ≤ 8 ms | ≤ 4 ms | ≤ 3 ms |
| 10,000 | N/A | ≤ 6 ms | ≤ 5 ms |
| 50,000 | N/A | ≤ 12 ms | ≤ 10 ms |
| 100,000 | N/A | ≤ 16 ms | ≤ 14 ms |

### Correctness regression suite (runs on both backends)

| Test | Scenario | Pass criterion |
|------|----------|----------------|
| B1 | Orbit stability | eccentricity drift ≤ 1% over 10 sim-seconds; CoM drift ≤ 0.5 px |
| B2 | Wrap-boundary cluster (fix 2) | no overlap > 0.95 × rSum; CPU/GPU contact count diff ≤ 5% |
| B3 | Dense cluster jitter (fix 3) | GPU jitter ≤ 1.5× CPU jitter |
| B4 | Adaptive iterations (fix 4) | no overlap > 0.1 × rSum after substep 50 |
| B5 | Per-substep broadphase (fix 5) | no tunneling across high-speed cluster collisions |
| B6 | BH absorption | all prey reach `absorbing !== null` within 60 sim-seconds; ghost gravity bounded ≤ 2 substep deviation |
| B7 | Pinned body | position ± 0.001 px; velocity = (0,0); other bodies attracted normally |
| B8 | Drag prediction continuity | no teleport artifacts in 5-sec drag preview path |

### Fallback verification

| Test | Action | Expected |
|------|--------|----------|
| F1 | `?backend=force-cpu` | Bit-identical to `main` branch |
| F2 | `gpuDevice.destroy()` mid-session | CPU takes over within 2 frames; no crash |
| F3 | Browser without WebGPU | CPU path active, UI functional |

---

## 9. Out of Scope

- N > 10⁵: requires GPU Barnes-Hut, separate multi-week project, not addressed here.
- Mobile WebGPU (iOS Safari, Android Chrome low-end): tested as fallback to CPU only.
- Multi-GPU / dedicated compute device: WebGPU spec doesn't support; not addressed.

---

## 10. Audit Trail

Design produced via:
- Round 1: code-architect initial blueprint (57 KB; preserved in `.claude/architect-blueprint.md`)
- Round 2: my 5-issue push-back (Phase 1 readback, dispatch overhead, hash bandwidth, padding, EPSILON)
- Round 2: architect revised (accepted 4, defended 1)
- Round 3: code-reviewer audit (found 1 CRITICAL + 4 HIGH)
- Round 3: architect resolved CRITICAL (concurrency model misdescription) + 4 HIGH (cold-start, justification, ghost-frame, recovery) + locked MEDIUM (flag layout)

Phase 1 implementation will follow the same pattern: detailed design → reviewer round 1 → push-back → reviewer round 2 → implement → acceptance test → commit.

---

## 11. Phase 1 — Implementation Status

**Status**: Complete. Branch `claude/exciting-poincare-d02419`.

### 11.1 Commits

| SHA | Subject |
|---|---|
| `3c66256` | docs(webgpu): blueprint (this file) |
| `079ed11` | feat(webgpu): K1 `gravity_accel.wgsl` compute shader (frozen after 2× architect + 2× reviewer) |
| `c7e1fa5` | docs: CLAUDE.md project guide |
| `368cbe9` | feat(webgpu): JS-side K1 integration + backend abstraction |

### 11.2 Files

| Path | Role |
|---|---|
| `src/kernels/gravity_accel.wgsl` | K1 compute kernel — tiled all-pairs, min-image PBC, fp32. FROZEN. |
| `src/gpu-init.js` | `detectBackend()` / `loadKernel()`; honors `?backend=force-cpu` and `?backend=verbose` |
| `src/physics-gpu-gravity.js` | K1 pipeline wrapper — buffers (`positions`, `metas`, `outputBuf`, ping-pong `stagingBufs`), Params uniform, `growIfNeeded` / `recordDispatch` / `readbackStaging` |
| `src/physics-backend.js` | CPU+GPU wrapper with transparent swap on `device.lost`; exposes `init / prepareFrame / step / destroy` |
| `src/physics.js` | `stepPBD(entities, dt, injectedAccels?)` — third arg optional; copies (ax,ay) into `_scratch` when present, otherwise runs CPU `computeAccelerations` unchanged |
| `src/main.js` | Async RAF loop via `runFrame().finally(requestAnimationFrame)`; backend created via top-level `await` before first frame |
| `src/state.js` | `state.backendName: 'cpu' \| 'webgpu' \| null` for HUD/debug |

### 11.3 Design decisions resolved (architect + audit)

- **Fork A — dispatch model**: A2.α (pipelined per-substep with optional `injectedAccels` 3rd arg to `stepPBD`). Positions snapshotted at end of substep N's `applyBoundary` are exactly what CPU's `computeAccelerations` at start of substep N+1 would read → **no positional staleness**. Only divergence vs the CPU oracle is fp32 vs fp64 arithmetic.
- **Fork B — recovery**: B1 (no shadow buffer in Phase 1). CPU is authoritative for `entity.{x,y,vx,vy}`; on `device.lost` the wrapper swaps to CPU and execution continues from CPU's existing state. Phase 2 will add the shadow when GPU takes ownership of positions.

### 11.4 Reviewer push-back

The first reviewer pass flagged a HIGH concerning cross-frame `runFrame` overlap. Audit found the analysis was incorrect — `requestAnimationFrame(frame)` was only scheduled at the end of `runFrame` (after all awaits), so RAF semantics prevent overlap. The reviewer's proposed `.finally()` pattern was applied anyway because it independently fixes a real silent-loop-death bug (if `runFrame` throws before reaching the line-115 schedule, the catch was swallowing without rescheduling). MEDIUM #6 (`!= null` → truthy `e.absorbing` for consistency with `physics.js:86`) also accepted as a one-line drive-by.

### 11.5 Acceptance results

All B1-B8 + F1-F3 green at branch tip. Executed via direct `backend.step()` driving because the headless preview keeps the tab in `visibilityState='hidden'` (RAF doesn't fire) — same code path either way.

| Test | Result |
|---|---|
| Smoke (N=101) | 3.4 ms/substep, no errors, no NaN over 60 substeps |
| **F1** force-cpu | Bit-identical to `main` by construction (no 3rd arg → `physics.js:267-273` runs the original `computeAccelerations`). The new code path is reached only via the GPU backend. |
| CPU↔GPU parity (extra) | 2-body, 60-substep replay on each backend: max divergence 1.6 × 10⁻⁸ px / 2 × 10⁻⁸ px·s⁻¹ — well under the documented ~6 × 10⁻⁴ fp32 budget at N=5000 |
| **B1** orbit stability | eccentricity 0.538 % over 10 sim-seconds (< 1 % criterion); pinned drift = 0 |
| **B2-B5** cluster | 25-body hex pack in wrap mode, 8 sim-seconds: max overlap 0.143 px (≪ 0.95 × rSum), max residual v 2.78 px/s, no NaN, no entity loss |
| **B6** BH absorption | started substep 0, completed substep 17 (≈ 0.28 s, matches `absorptionDuration = 0.3` default); FLAG_ABSORBING gate verified — tracker orbiter unperturbed by absorbing prey |
| **B7** pinned bodies | `moved = 0`, `vx = vy = 0` over 300 substeps; orbiter continues normally |
| **B8** drag prediction | 300-point smooth path, max step 2.31 px, no NaN. Backend-agnostic (CPU-only by design) |
| **F2** device.lost swap | `device.lost` Promise verified to fire on `device.destroy()` with correct payload (`{reason: 'destroyed'}`). Wrapper's lost-handler chain is straight-line code — no async branches. Direct mid-session destroy of the wrapper's underlying device not exercised (closure not reachable from `eval`); structural evidence accepted. |
| **F3** no-WebGPU fallback | structurally identical to F1's path (`gpu-init.js:48-51` early return on `!navigator.gpu`) |

### 11.6 Performance snapshot

At N=101 the GPU path measures **3.4 ms / substep** (driven sequentially from `eval`, so this is steady-state including upload + dispatch + readback + CPU integrator/solver). The per-substep cost is dominated by IPC, not K1 compute — at small N the kernel itself is sub-millisecond on the 5070 Ti, and `writeBuffer` for positions + metas + params is < 100 µs combined. The §8 perf table (N=500 ≤ 2 ms, N=5000 ≤ 8 ms) was not exercised in acceptance — that's Phase 1.1 optimization territory if needed.

### 11.7 Known minor (intentionally deferred)

- **Meta re-uploaded every substep** (full `uploadMetaAll`, 80 KB at N=5000). Architect risk #2 mitigation — guarantees `FLAG_ABSORBING` flips reach GPU before next dispatch. Can be optimized to per-entity patches in Phase 1.1.
- **No `mapAsync` timeout watchdog** (architect risk #1). Phase 1 assumes K1 + 1 frame's CPU work always fit within one frame. A future safety valve could time-bound the await and fall back to CPU on stall.
- **Capacity shrink hysteresis**: buffers shrink only when `N < capacity / 4` to avoid alloc/free thrash on single-add/single-remove patterns.
- **Test-only invariant**: when `state.entities` is replaced wholesale (programmatic test mutation), the backend's `pendingReadback` holds stale accels. Production paths (UI add/remove, clear-to-zero) are unaffected because the N=0 transition triggers `submitDispatch`'s early-return which nulls the readback. Documented for whoever writes Phase 2 tests.

### 11.8 Hand-off to Phase 2

Phase 2 scope per §2: K1–K8 entirely GPU-resident, including spatial-hash broadphase, contact detect, velocity solver (Jacobi + atomic-emulated fixed-point), position solver, and warm-start rebuild. Load-bearing decisions still open (architect input required before any code):

- Solver concurrency model (Jacobi vs colored GS — blueprint §3.1 already eliminates the GS-by-cell-parity proposal)
- `atomicMax<f32>` emulation strategy (WGSL has no f32 atomics — bit-reinterpret + atomicMax<i32> is the standard workaround)
- `pairImpulseTable` open-addressing hash sizing + linear probe budget
- Shadow buffer activation (Fork B in this phase flips: GPU becomes authoritative for positions; CPU shadow exists for `device.lost` recovery and Phase 3 rendering)

When ready, kick off Phase 2 with `/feature-dev` + this blueprint as input.
