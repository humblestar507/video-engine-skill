// Run in the page console (top-level await). Page: hover-buttons/index.html served by rec_server.py
await document.fonts.ready; relayout();
window.requestAnimationFrame = () => 0; let FAKE = 0; performance.now = () => FAKE;           // the recorder owns the clock
const P = await fetch('/rec-tmp/cursor_path_tip.json').then(r => r.json());                    // measured cursor tip per video frame (design px)
const X0 = 300, Y0 = 450, CW = 1350, CH = 155, CY = 527, BASEL = 17.65;                       // crop of the 1920x1054 design stage
const cv2 = document.createElement('canvas'); cv2.width = CW; cv2.height = CH; const g = cv2.getContext('2d');
function mirror() {                                                                           // canvas mirror of the DOM state
  const sr = document.getElementById('stage').getBoundingClientRect(), s = window.__scale;
  g.fillStyle = '#dfe0d6'; g.fillRect(0, 0, CW, CH);
  for (const b of btns) {
    const r = b.el.getBoundingClientRect(), ox = (r.left - sr.left) / s - X0, oy = CY - Y0, c = b.cur;
    const col = getComputedStyle(b.el).getPropertyValue('--c').trim();
    g.fillStyle = col; g.fillRect(ox + c.L, oy + c.cy - c.h / 2, c.R - c.L, c.h);
    g.font = '400 49px "JetBrains Mono"'; g.textBaseline = 'alphabetic'; g.fillStyle = col; g.fillText(b.name, ox + 55.5, oy + BASEL);
    g.save(); g.beginPath(); g.rect(0, 0, ox + c.R, CH); g.clip(); g.fillStyle = '#dfe0d6'; g.fillText(b.name, ox + 55.5, oy + BASEL); g.restore();
    g.fillStyle = '#dfe0d6'; for (const [dx, dy] of c.d) g.fillRect(ox + dx - 1.875, oy + dy - 1.875, 3.75, 3.75);
  }
}
function simFrame(k) {                                                                        // same logic as the page's frame()
  FAKE = P.t[k] * 1000; const q = P.p[k], sr = document.getElementById('stage').getBoundingClientRect(), s = window.__scale;
  pointer = q ? { x: sr.left + q[0] * s, y: sr.top + q[1] * s } : null;
  for (const b of btns) { b.want = pointer ? b.zone(b.dist(pointer.x, pointer.y)) : 'idle';
    if (b.canRetarget()) { const nx = b.nextStep(); if (nx !== b.state) b.go(nx); } b.step(FAKE); b.render(); }
}
const log = [];
const upl = async (url, body) => { for (let a = 0; a < 6; a++) { try { const r = await fetch(url, { method: 'POST', body }); if (r.ok) return; } catch (e) { await new Promise(r => setTimeout(r, 200 * (a + 1))); } } throw new Error('upload failed ' + url); };
(async () => {
  for (let k = 0; k < P.t.length; k++) {
    simFrame(k); mirror(); log.push(btns.map(b => [b.state, +b.cur.L.toFixed(2), +b.cur.R.toFixed(2)]));
    const blob = await new Promise(r => cv2.toBlob(r, 'image/png')); await upl('/upload/btn3/' + k, blob);
  }
  await upl('/upload/btnlog3/0', JSON.stringify({ t: P.t, log }));                             // per-frame states for the timing comparison
})();
