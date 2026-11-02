# Project guidance

Read README.md and docs/maintenance.md before changing physics or publishing.

- Keep this a static ESM app: no production bundler or npm dependency tree.
- The supported engine is Rapier 2D 0.19.3; the JS solver is the offline fallback. K1 is the only retained WebGPU kernel. Historical alternatives are recoverable from commit 9c6b894.
- Preserve the source-only charge rule unless the user asks to change the game model. It intentionally breaks action/reaction for unequal charges.
- Pair softening is max(radiusA + radiusB, epsilon). Gravity uses a minimum image in wrap mode; field illustrations use different kernels and must not be described as exact dynamics or relativistic curvature.
- Use fixed simulation steps. Preserve fractional accumulator time and give pause priority over edit mode. Performance throttling may reduce achieved time speed.
- Wrap contacts must be solved once per physical pair; ghost impulses are forwarded once. Pinned ghosts must remain kinematic.
- GPU readback yields to input. Verify entity identities and edited metadata before applying an indexed result.
- WebGL context restoration must recreate every GPU resource, including sag textures, field targets and mesh buffers. Test restoration after a resize.
- Required validation: `node --test tests/core.test.mjs`, then `tests/browser.mjs` with Playwright (README has setup). Browser assertions must wait for real initialization and fail on exceptions. GPU skips must be disclosed.
- Before publishing, review the diff and run the workflow. Deploy only runtime assets; confirm Pages version.json matches the deployed commit.
- Keep documentation compact: README for operation, this file for maintenance rules, docs/maintenance.md for evidence and model limitations. Do not add phase-by-phase planning documents.
- Do not delete untracked user files. Remove tracked dead artifacts only in commits, preserving normal Git history. Do not force-push or remove historical branches for routine cleanup.
