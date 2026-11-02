// Local completed-frame meter: no third-party request on the startup path.
export function createFpsMeter() {
  if (new URLSearchParams(location.search).get('fps') === 'off') {
    return { begin() {}, end() {}, getFps: () => 0, getRafFps: () => 0 };
  }
  const el = document.createElement('output');
  el.id = 'fps-meter';
  el.setAttribute('aria-label', '完成渲染帧率');
  Object.assign(el.style, {
    position: 'fixed', bottom: '8px', right: '8px', zIndex: '10',
    padding: '4px 7px', background: '#13202dcc', color: '#a5e7bc',
    font: '11px monospace', pointerEvents: 'none', borderRadius: '4px',
  });
  document.body.appendChild(el);
  let last = performance.now(), frames = 0, fps = 0;
  return {
    begin() {},
    end(completed = true) {
      if (completed) frames++;
      const now = performance.now();
      if (now - last >= 500) {
        fps = frames * 1000 / (now - last);
        el.textContent = `${fps.toFixed(0)} FPS`;
        frames = 0;
        last = now;
      }
    },
    getFps: () => fps,
    getRafFps: () => fps,
  };
}
