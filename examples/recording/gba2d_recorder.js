// Run in the page console (top-level await). Page: gba-2d/index.html served by rec_server.py
while (!window.PIXEL) await new Promise(r => setTimeout(r, 50));
window.requestAnimationFrame = () => 0;                                                       // the recorder steps time itself
const up = async (name, i, canvas) => { const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
  for (let a = 0; a < 6; a++) { try { const r = await fetch('/upload/' + name + '/' + i, { method: 'POST', body: blob }); if (r.ok) return; } catch (e) { await new Promise(r => setTimeout(r, 200 * (a + 1))); } } throw new Error('upload failed ' + i); };
const ex = document.createElement('canvas'); ex.width = cv.width; ex.height = cv.height + 70; const ec = ex.getContext('2d');
function exportFrame() {
  ec.fillStyle = '#1b1517'; ec.fillRect(0, 0, ex.width, ex.height); ec.drawImage(cv, 0, 0);
  ec.fillStyle = '#e8d9dc'; ec.font = '600 26px Consolas, monospace'; ec.textAlign = 'center';
  ec.fillText((S.current || 'press a button'), ex.width / 2, cv.height + 30);
  ec.fillStyle = '#b9a9ad'; ec.font = '22px Consolas, monospace';
  ec.fillText('frame ' + (S.f + 1) + '/' + N + '  ' + (S.paused ? 'paused' : (S.dir < 0 ? 'reverse ' : 'playing ') + S.speed + 'x'), ex.width / 2, cv.height + 58);
}
// 1) one loop for the pixel comparison with the GIF
for (let k = 0; k < N; k++) { S.f = k; S.paused = true; drawFrame(); await up('gba_loop', k, cv); }
// 2) scripted button demo at 20 fps with a manual clock
S.f = 0; S.acc = 0; S.dir = 1; S.paused = false; S.speed = 1; S.dirty = true; S.current = ''; S.pressed.clear(); held.clear();
const DT = 50, script = [
  [1.0, 'a', 'down', 0.3], [2.0, 'dpad', 'ne', 0.3], [3.0, 'dpad', 'sw', 0.3], [4.0, 'dpad', 'nw', 0.3], [5.0, 'dpad', 'se', 0.3],
  [6.0, 'start', 'down', 0.3], [7.0, 'select', 'down', 0.3], [8.5, 'select', 'down', 0.3], [9.5, 'b', 'down', 0.3], [11.5, 'b', 'down', 0.3],
  [12.5, 'power', 'down', 0.3], [14.0, 'power', 'down', 0.3], [15.5, 'power', 'down', 0.3], [17.0, 'pokeball', 'down', 0.25], [17.6, 'pokeball', 'down', 0.25], [18.2, 'pokeball', 'down', 0.25]];
const frames = 400; let si = 0, active = null, id = 0;
for (let i = 0; i < frames; i++) {
  const t = i * DT / 1000;
  if (active && t >= active.until) { release(active.id); active = null; }
  if (!active && si < script.length && t >= script[si][0]) { const [, n, k, hold] = script[si++]; active = { id: 'd' + (id++), until: t + hold }; press(n, k, active.id); }
  if (!S.paused) { S.acc += DT * S.speed; while (S.acc >= FRAME_MS) { S.acc -= FRAME_MS; S.f = (S.f + S.dir + N) % N; S.dirty = true; } }
  if (S.dirty) { drawFrame(); S.dirty = false; updateHud(); }
  exportFrame(); await up('gba_demo', i, ex);
}
