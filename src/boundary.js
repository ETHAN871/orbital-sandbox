export function wrapCoordinate(value, span) {
  return span > 0 ? ((value % span) + span) % span : value;
}

export function minImageDelta(delta, span) {
  if (span <= 0) return delta;
  delta %= span;
  if (delta > span * 0.5) return delta - span;
  if (delta < -span * 0.5) return delta + span;
  return delta;
}
