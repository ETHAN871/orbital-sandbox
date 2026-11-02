import { SIM_DT } from './state.js';

// Drop whole overdue steps only when overloaded; fractional time must survive
// normal frames or simulation speed depends on the display refresh rate.
export function advanceClock(accumulator, realDt, ratio, budget) {
  if (ratio <= 0) return { steps: 0, remainder: 0 };
  const total = accumulator + Math.max(0, Math.min(0.1, realDt)) * ratio;
  const available = Math.floor((total + 1e-12) / SIM_DT);
  const steps = Math.min(available, Math.max(1, Math.floor(budget)));
  return { steps, remainder: Math.max(0, total - available * SIM_DT) };
}
