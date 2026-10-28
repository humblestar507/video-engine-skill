// Run in the page console (top-level await) with an 800x600 viewport. Page: pixel-3d/index.html served by rec_server.py
window.requestAnimationFrame = () => 0; await new Promise(r => setTimeout(r, 300));          // let the pending real frame finish
PIX = 3; resize();
const FPS = 24, DUR = 14, FR = FPS * DUR;
const by = a => interactive.find(m => m.userData.action === a);
const script = [[1.0, 'a'], [2.0, 'ne'], [3.0, 'sw'], [4.0, 'nw'], [5.0, 'se'], [6.0, 'start'], [7.0, 'select'], [8.0, 'select'], [9.0, 'b'], [10.0, 'power'], [11.0, 'pokeball'], [11.6, 'pokeball']];
const upl = async (i, data) => { for (let a = 0; a < 6; a++) { try { const r = await fetch('/upload/c3d/' + i, { method: 'POST', body: data }); if (r.ok) return; } catch (e) { await new Promise(r => setTimeout(r, 200 * (a + 1))); } } throw new Error('upload ' + i); };
S.f = 0; S.acc = 0; S.dir = 1; S.paused = false; S.speed = 1; S.current = ''; S.dirty = true; held.clear();
Object.assign(goal, { theta: 0.62, phi: 1.1, r: 360 }); Object.assign(cam, goal);
last = -1000 / FPS; let si = 0, active = null, pid = 0;
(async () => {
  for (let k = 0; k < FR; k++) {
    const t = k / FPS, now = k * 1000 / FPS;
    if (active && t >= active.until) { release(active.id); active = null; }
    if (!active && si < script.length && t >= script[si][0]) { const m = by(script[si][1]); active = { id: 'r' + (pid++), until: t + 0.35 }; press(m, active.id); si++; }
    const orb = Math.min(1, Math.max(0, (t - 1.5) / 8.5));                                    // one full turn between 1.5 s and 10 s
    goal.theta = 0.62 + orb * 2 * Math.PI; goal.phi = 1.1 - 0.28 * Math.sin(orb * Math.PI);
    goal.r = t < 10.5 ? 360 : 360 - 140 * Math.min(1, (t - 10.5) / 2.5);
    loop(now);                                                                                // renders synchronously
    await upl(k, cvs.toDataURL('image/png'));
  }
})();
