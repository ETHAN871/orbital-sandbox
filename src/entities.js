// entities.js — Entity factory and color rules.
// Entities are plain objects (no class). Schema documented in state.js.

import { nextEntityId } from './state.js?v=ovl1';

// ─── Color rules ──────────────────────────────────────────────────
// Planets get a vivid random color (chosen once at creation).
// Black holes are deterministic: black for charge ∈ {0, +1}, white for charge = -1.
// `baseColor` is the entity's stored random color, used when the type is "planet"
// regardless of charge; the function returns the *display* color.

export function randomPlanetColor() {
  const hue = Math.floor(Math.random() * 360);
  return `hsl(${hue}, 75%, 62%)`;
}

export function resolveDisplayColor(type, charge, baseColor) {
  if (type === 'black_hole') {
    return charge === -1 ? '#ffffff' : '#000000';
  }
  return baseColor;
}

// ─── Factory ──────────────────────────────────────────────────────
// Creates a fully-formed entity ready to push into state.entities.
// Position and velocity come from the placement gesture; mass/radius/charge
// come from the "pending" template (or selected entity in edit mode).

export function createEntity({ type, x, y, vx, vy, mass, radius, charge, pinned = false }) {
  const baseColor = type === 'planet' ? randomPlanetColor() : '#000000';
  return {
    id: nextEntityId(),
    type,
    mass,
    radius,
    charge,
    x, y,
    vx, vy,
    baseColor,                 // stable random color for planets
    color: resolveDisplayColor(type, charge, baseColor), // derived; refresh on edit
    // V8.1: per-entity trail history removed. Trails now render as a
    // global phosphor-decay FBO (renderer-webgl.js); each frame plots one
    // dot at the entity's current position and a decay pass handles
    // history.
    // Verlet uses last acceleration; cached so we don't recompute mid-step.
    ax: 0,
    ay: 0,
    // Black-hole absorption animation state. null = normal entity. When a
    // black hole consumes this entity, physics.handleCollisions sets:
    //   { blackHoleId, elapsedSim, startRadius, startX, startY }
    // and physics.updateAbsorptions then shrinks + lerps it into the hole.
    absorbing: null,
    // Pinned bodies are kinematically frozen — they keep applying gravity
    // and participate in collisions (as infinite-mass anchors), but their
    // position and velocity never change. Settable at creation (panel
    // toggle) or via the per-entity edit-mode button.
    pinned,
  };
}

// Refresh the display color after an in-place charge or type change.
export function refreshEntityColor(entity) {
  entity.color = resolveDisplayColor(entity.type, entity.charge, entity.baseColor);
}
