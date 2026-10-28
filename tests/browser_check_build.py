"""Builds tests/browser_check/ (CSS exports + a page that samples them with the Web Animations API).
Serve it (python -m http.server 8765 --directory tests/browser_check), open index.html in a real
browser and read the JSON in <pre id=out>: per-channel max |browser - model| in fh / ratio."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VE_CACHE", str(ROOT / "tests" / ".cache_test"))
from engine import export, model  # noqa: E402
from engine import spec as S  # noqa: E402
from engine.store import Store  # noqa: E402

out_dir = ROOT / "tests" / "browser_check"
out_dir.mkdir(exist_ok=True)
expect = {}
for name, clip in (("ball", "ball.mp4"), ("ease", "ease.mp4")):
    sp = S.build(Store(ROOT / "tests" / "clips" / clip), "1")
    path = export.export_css(sp, out_dir / f"{name}.css")
    css = path.read_text(encoding="utf-8").replace(".ve-obj-1", f".ve-{name}").replace("ve-1-", f"ve-{name}-")
    path.write_text(css, encoding="utf-8")
    ts = np.linspace(0, sp["duration"] * 0.999, 60)
    ev = model.eval_object(sp["objects"][0], ts)
    expect[name] = {"ts": ts.tolist(), "expect": {k: v.tolist() for k, v in ev.items()}, "T": sp["duration"]}
(out_dir / "expect.json").write_text(json.dumps(expect))
(out_dir / "index.html").write_text("""<!doctype html><meta charset=utf-8><title>css check</title>
<link rel=stylesheet href="ball.css"><link rel=stylesheet href="ease.css">
<div style="position:relative;width:640px;height:360px"><div class="ve-ball"></div><div class="ve-ease"></div></div>
<pre id=out>running</pre>
<script>
fetch('expect.json').then(r=>r.json()).then(E=>{
  const res={};
  for (const name of ['ball','ease']) {
    const el=document.querySelector('.ve-'+name), anims=el.getAnimations();
    res[name]={n_anims:anims.length, max_err:{x:0,y:0,scale:0,opacity:0}};
    E[name].ts.forEach((t,i)=>{
      anims.forEach(a=>{a.pause(); a.currentTime=t*1000;});
      const cs=getComputedStyle(el);
      for (const prop of ['x','y','scale','opacity'])
        res[name].max_err[prop]=Math.max(res[name].max_err[prop], Math.abs(parseFloat(cs.getPropertyValue('--ve-'+prop))-E[name].expect[prop][i]));
    });
  }
  document.getElementById('out').textContent=JSON.stringify(res);
});
</script>""", encoding="utf-8")
print("built", out_dir)
