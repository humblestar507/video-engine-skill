"""Exporters: a scene spec as code.

js     - <name>.js (pure model + spec, runs in node and browser, checked against model.py
         numerically) and <name>.html (canvas viewer)
css    - @keyframes per channel on registered custom properties. Ballistic flights are
         EXACT cubic-beziers (a parabola over [0,T] is cubic-bezier(1/3,a,2/3,a+1/3)),
         springs become linear() with sampled points (approximation, error reported)
godot  - GDScript implementing the same model. NOT run in Godot here (no engine available)

Positions are fh (fraction of stage height); multiply by the stage height in your
engine. The header of every export lists which channels are fitted models and which
are replayed samples or assumptions."""
import json
from pathlib import Path

import numpy as np

from .curves import PRESETS, spring_step
from .model import css_ballistic_flights

CSS_LINEAR_POINTS = 320


def _header(spec, comment):
    lines = [f"Exported from video-engine spec (source: {spec.get('source')})",
             f"stage {spec['stage']['w']}x{spec['stage']['h']} px, {spec['duration']:.3f} s, units: fh = fraction of stage height"]
    for o in spec["objects"]:
        for name, ch in o["channels"].items():
            lines.append(f"object {o['id']} {name}: {ch['type']} [{ch.get('src', '?')}] {ch.get('provenance', '')}")
    for w in spec.get("warnings", []):
        lines.append(f"warning: {w}")
    return "\n".join(comment + l for l in lines)


# ---------------------------------------------------------------------------- JS

JS_MODEL = r"""
const LUT_N = 513;
const _lut = new Map();
function bezLut(s) {
  const key = s.join(',');
  if (_lut.has(key)) return _lut.get(key);
  const [x1, y1, x2, y2] = s, bx = new Float64Array(LUT_N), by = new Float64Array(LUT_N);
  for (let i = 0; i < LUT_N; i++) {
    const u = i / (LUT_N - 1), m = 1 - u;
    bx[i] = 3 * m * m * u * x1 + 3 * m * u * u * x2 + u * u * u;
    by[i] = 3 * m * m * u * y1 + 3 * m * u * u * y2 + u * u * u;
  }
  _lut.set(key, [bx, by]);
  return [bx, by];
}
function interp(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  return ys[lo] + (x - xs[lo]) / (xs[hi] - xs[lo]) * (ys[hi] - ys[lo]);
}
function springStep(tau, w0, z) {
  tau = Math.max(tau, 0);
  if (Math.abs(z - 1) < 1e-3) return 1 - Math.exp(-w0 * tau) * (1 + w0 * tau);
  if (z < 1) {
    const wd = w0 * Math.sqrt(1 - z * z);
    return 1 - Math.exp(-z * w0 * tau) * (Math.cos(wd * tau) + z * w0 / wd * Math.sin(wd * tau));
  }
  const s = w0 * Math.sqrt(z * z - 1), r1 = -z * w0 + s, r2 = -z * w0 - s;
  return 1 - (r2 * Math.exp(r1 * tau) - r1 * Math.exp(r2 * tau)) / (r2 - r1);
}
function segProgress(seg, t) {
  if (seg.model === 'spring') return springStep(t - seg.t_start, seg.omega0, seg.zeta);
  const p = Math.min(Math.max((t - seg.t_start) / seg.duration, 0), 1);
  const shape = seg.model === 'bezier' ? seg.bezier : PRESETS[seg.model];
  const [bx, by] = bezLut(shape);
  return interp(p, bx, by);
}
function evalSegments(ch, t) {
  let v = ch.initial;
  for (const seg of ch.segments) if (t >= seg.t_start) v = seg.from + (seg.to - seg.from) * segProgress(seg, t);
  return v;
}
function flights(ch, tEnd) {
  const g = ch.g, ground = ch.ground, e = ch.restitution == null ? 0 : ch.restitution, out = [];
  let t0 = ch.t_ref, y0 = ch.y0, v0 = ch.v0;
  for (let k = 0; k < 80; k++) {
    if (ground == null || g <= 0) { out.push([t0, y0, v0, Infinity]); break; }
    const disc = v0 * v0 + 2 * g * (ground - y0);
    const tau = (-v0 + Math.sqrt(Math.max(disc, 0))) / g, t1 = t0 + tau;
    out.push([t0, y0, v0, t1]);
    if (t1 >= tEnd || tau < 1e-4) break;
    const vIn = v0 + g * tau;
    t0 = t1; y0 = ground; v0 = -e * vIn;
    if (Math.abs(v0) < 1e-4) { out.push([t0, ground, 0, Infinity]); break; }
  }
  return out;
}
const _fl = new WeakMap();
function evalBallistic(ch, t) {
  let fl = _fl.get(ch);
  if (!fl || fl.tEnd < t) { fl = { tEnd: Math.max(t, 30), list: flights(ch, Math.max(t, 30)) }; _fl.set(ch, fl); }
  let k = 0;
  for (let i = 0; i < fl.list.length; i++) if (fl.list[i][0] <= t) k = i; else break;
  const f = fl.list[k], tau = Math.max(t - f[0], 0);
  let y = f[1] + f[2] * tau + 0.5 * ch.g * tau * tau;
  if (ch.ground != null) y = Math.min(y, ch.ground);
  return y;
}
function evalChannel(ch, t) {
  switch (ch.type) {
    case 'const': return ch.v;
    case 'linear': return ch.x0 + ch.vx * t;
    case 'samples': return interp(t, ch.t, ch.v);
    case 'segments': return evalSegments(ch, t);
    case 'ballistic': return evalBallistic(ch, t);
  }
  throw new Error('unknown channel ' + ch.type);
}
function state(t, objIndex = 0) {
  const c = SPEC.objects[objIndex].channels;
  return { x: evalChannel(c.x, t), y: evalChannel(c.y, t), scale: evalChannel(c.scale, t), opacity: evalChannel(c.opacity, t) };
}
if (typeof module !== 'undefined') module.exports = { SPEC, state, evalChannel };
else window.VE = { SPEC, state, evalChannel };
"""

HTML = """<!doctype html>
<meta charset="utf-8"><title>video-engine viewer</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}canvas{display:block;margin:12px auto;background:#000}
.bar{display:flex;gap:12px;align-items:center;justify-content:center}input[type=range]{width:50%%}</style>
<canvas id="c"></canvas>
<div class="bar"><button id="b">pause</button><input id="s" type="range" min="0" max="1" step="0.0005" value="0"><span id="t"></span></div>
<script src="%(js)s"></script>
<script>
const S = VE.SPEC, W = S.stage.w, H = S.stage.h, cv = document.getElementById('c'), cx = cv.getContext('2d');
cv.width = W; cv.height = H; let playing = true, t0 = performance.now(), off = 0;
const rgb = a => `rgb(${a[2]},${a[1]},${a[0]})`;
function draw(t) {
  cx.fillStyle = rgb(S.bg); cx.fillRect(0, 0, W, H);
  if (S.ground) { cx.fillStyle = rgb(S.ground.color); cx.fillRect(0, S.ground.top_fh * H, W, H); }
  S.objects.forEach((o, i) => {
    if (!o.renderable) return;
    const s = VE.state(t, i), w = o.size_fh[0] * H * s.scale, h = o.size_fh[1] * H * s.scale;
    cx.globalAlpha = Math.min(Math.max(s.opacity, 0), 1); cx.fillStyle = rgb(o.color); cx.beginPath();
    if (o.shape === 'disc') cx.ellipse(s.x * H, s.y * H, w / 2, h / 2, 0, 0, 7); else cx.rect(s.x * H - w / 2, s.y * H - h / 2, w, h);
    cx.fill(); cx.globalAlpha = 1;
  });
}
function frame(now) {
  const T = S.duration; let t = playing ? ((now - t0) / 1000 + off) %% T : off;
  document.getElementById('s').value = t / T; document.getElementById('t').textContent = t.toFixed(3) + ' s'; draw(t);
  requestAnimationFrame(frame);
}
document.getElementById('b').onclick = e => { const T = S.duration, t = ((performance.now() - t0) / 1000 + off) %% T; playing = !playing; off = t; t0 = performance.now(); e.target.textContent = playing ? 'pause' : 'play'; };
document.getElementById('s').oninput = e => { playing = false; document.getElementById('b').textContent = 'play'; off = e.target.value * S.duration; };
requestAnimationFrame(frame);
</script>
"""


def export_js(spec, out_dir, name="scene"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = (f"/*\n{_header(spec, ' * ')}\n */\n'use strict';\nconst PRESETS = {json.dumps(PRESETS)};\n"
          f"const SPEC = {json.dumps(spec)};\n{JS_MODEL}")
    (out_dir / f"{name}.js").write_text(js, encoding="utf-8")
    (out_dir / f"{name}.html").write_text(HTML % {"js": f"{name}.js"}, encoding="utf-8")
    return [out_dir / f"{name}.js", out_dir / f"{name}.html"]


# ---------------------------------------------------------------------------- CSS

def _bez_css(shape):
    return "cubic-bezier(" + ", ".join(f"{v:.6g}" for v in shape) + ")"


def _linear_css(points):
    return "linear(" + ", ".join(f"{p:.5f}" for p in points) + ")"


def css_tracks(spec, obj):
    """Per channel: constant value, or keyframes [(t, value, timing)] (timing applies up to the next keyframe)."""
    T = spec["duration"]
    out = {}
    spring_err = []
    for name, ch in obj["channels"].items():
        k = ch["type"]
        if k == "const":
            out[name] = {"const": ch["v"]}
        elif k == "linear":
            out[name] = {"kf": [(0.0, ch["x0"], "linear"), (T, ch["x0"] + ch["vx"] * T, None)]}
        elif k == "samples":
            tt, vv = np.array(ch["t"]), np.array(ch["v"])
            idx = np.unique(np.linspace(0, len(tt) - 1, min(len(tt), 120)).astype(int))
            out[name] = {"kf": [(float(tt[i]), float(vv[i]), "linear") for i in idx], "approx": "sampled replay, linear between samples"}
        elif k == "segments":
            kf = [(0.0, ch["initial"], "linear")]
            last = ch["initial"]
            for seg in ch["segments"]:
                if abs(last - seg["from"]) > 1e-9:
                    kf.append((max(seg["t_start"] - 1e-4, kf[-1][0]), last, "linear"))
                if seg["model"] == "spring":
                    tau = np.linspace(0, 12 / max(seg["zeta"] * seg["omega0"], 0.5), 4000)
                    dev = np.abs(1 - spring_step(tau, seg["omega0"], seg["zeta"]))
                    settle = float(tau[np.nonzero(dev > 1e-3)[0][-1]]) if (dev > 1e-3).any() else 0.05
                    t_end = min(seg["t_start"] + settle, T)
                    ts = np.linspace(0, t_end - seg["t_start"], CSS_LINEAR_POINTS)
                    pts = spring_step(ts, seg["omega0"], seg["zeta"])
                    timing = _linear_css(pts)
                    to = seg["from"] + (seg["to"] - seg["from"]) * float(pts[-1])
                    dense = np.linspace(0, t_end - seg["t_start"], 4000)
                    err = float(np.max(np.abs(np.interp(dense, ts, pts) - spring_step(dense, seg["omega0"], seg["zeta"]))))
                    spring_err.append(err * abs(seg["to"] - seg["from"]))
                else:
                    t_end = seg["t_start"] + seg["duration"]
                    shape = seg["bezier"] if seg["model"] == "bezier" else PRESETS[seg["model"]]
                    timing = _bez_css(shape) if seg["model"] != "linear" else "linear"
                    to = seg["to"]
                kf.append((seg["t_start"], seg["from"], timing))
                kf.append((t_end, to, "linear"))
                last = to
            out[name] = {"kf": kf, "spring_max_error": max(spring_err) if spring_err else None}
        elif k == "ballistic":
            fl = css_ballistic_flights(ch, T)
            kf = []
            for f in fl:
                kf.append((f["t0"], f["from"], _bez_css(f["bezier"])))
            if fl:
                kf.append((fl[-1]["t1"], fl[-1]["to"], "linear"))
                if fl[-1]["t1"] < T:
                    kf.append((T, fl[-1]["to"], None))
            out[name] = {"kf": kf, "flights": len(fl)}
    for name, d in out.items():
        kf = d.get("kf")
        if not kf:
            continue
        # CSS: without a 100% keyframe the animation ends at the property's underlying value, not at a hold
        if kf[0][0] > 0:
            kf.insert(0, (0.0, kf[0][1], "linear"))
        if kf[-1][0] < T - 1e-9:
            kf[-1] = (kf[-1][0], kf[-1][1], "linear")
            kf.append((T, kf[-1][1], None))
        else:
            kf[-1] = (kf[-1][0], kf[-1][1], None)
    return out


def css_eval(kf, t, T):
    """CSS animation semantics for a keyframe list: interpolate with the timing function of the left keyframe."""
    ts = [k[0] for k in kf]
    assert ts[0] == 0.0 and abs(ts[-1] - T) < 1e-6, "keyframes must cover [0, T]: CSS does not hold the last value"
    if t <= ts[0]:
        return kf[0][1]
    if t >= ts[-1]:
        return kf[-1][1]
    i = int(np.searchsorted(ts, t, side="right") - 1)
    (t0, v0, tf), (t1, v1, _) = kf[i], kf[i + 1]
    p = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
    if tf is None or tf == "linear":
        f = p
    elif tf.startswith("linear("):
        pts = [float(x) for x in tf[7:-1].split(",")]
        f = float(np.interp(p, np.linspace(0, 1, len(pts)), pts))
    else:
        x1, y1, x2, y2 = [float(x) for x in tf[len("cubic-bezier("):-1].split(",")]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            u = (lo + hi) / 2
            bx = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u ** 2 * x2 + u ** 3
            lo, hi = (u, hi) if bx < p else (lo, u)
        u = (lo + hi) / 2
        f = 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u ** 2 * y2 + u ** 3
    return v0 + (v1 - v0) * f


def export_css(spec, out_path):
    T = spec["duration"]
    H = spec["stage"]["h"]
    errs = [d["spring_max_error"] for o in spec["objects"] for d in css_tracks(spec, o).values() if d.get("spring_max_error")]
    spring_note = (f"Spring channels use linear() with {CSS_LINEAR_POINTS} points (CSS easing level 2); max error vs the spring model "
                   f"{max(errs):.2g} in channel units." if errs else "No spring channels.")
    lines = [f"/*\n{_header(spec, ' * ')}\n * Not rendered in a browser by the engine tests; keyframes are checked numerically with CSS timing semantics.\n"
             f" * {spring_note} Ballistic flights are exact cubic-beziers, split at the apex.\n */",
             f":root {{ --H: {H}px; }}"]
    for prop, init in (("x", 0), ("y", 0), ("scale", 1), ("opacity", 1)):
        lines.append(f"@property --ve-{prop} {{ syntax: '<number>'; inherits: false; initial-value: {init}; }}")
    for o in spec["objects"]:
        tr = css_tracks(spec, o)
        sel = f".ve-obj-{o['id']}"
        anims = []
        base = {}
        for prop, d in tr.items():
            if "const" in d:
                base[prop] = d["const"]
                continue
            kf = d["kf"]
            lines.append(f"@keyframes ve-{o['id']}-{prop} {{")
            for (t, v, tf) in kf:
                tfs = f" animation-timing-function: {tf};" if tf else ""
                lines.append(f"  {t / T * 100:.6f}% {{ --ve-{prop}: {v:.8g};{tfs} }}")
            lines.append("}")
            anims.append(f"ve-{o['id']}-{prop} {T:.6g}s linear both infinite")
        w, h = (o["size_fh"] if o.get("size_fh") else (0.1, 0.1))
        lines.append(f"{sel} {{")
        lines.append("  position: absolute; left: 0; top: 0;")
        lines.append(f"  width: calc({w:.6g} * var(--H)); height: calc({h:.6g} * var(--H));")
        if o.get("color"):
            c = o["color"]
            lines.append(f"  background: rgb({c[2]:.0f} {c[1]:.0f} {c[0]:.0f}); border-radius: {'50%' if o.get('shape') == 'disc' else '0'};")
        for prop, v in base.items():
            lines.append(f"  --ve-{prop}: {v:.8g};")
        lines.append("  transform: translate(calc(var(--ve-x) * var(--H)), calc(var(--ve-y) * var(--H))) translate(-50%, -50%) scale(var(--ve-scale));")
        lines.append("  opacity: var(--ve-opacity);")
        if anims:
            lines.append("  animation: " + ", ".join(anims) + ";")
        lines.append("}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(out_path)


# ---------------------------------------------------------------------------- Godot

GD_MODEL = r'''
const PRESETS = %(presets)s
const LUT_N := 513
var _luts := {}

func _lut(s: Array) -> Array:
	var key := str(s)
	if _luts.has(key):
		return _luts[key]
	var bx := PackedFloat64Array(); var by := PackedFloat64Array()
	for i in LUT_N:
		var u := float(i) / (LUT_N - 1); var m := 1.0 - u
		bx.append(3 * m * m * u * s[0] + 3 * m * u * u * s[2] + u * u * u)
		by.append(3 * m * m * u * s[1] + 3 * m * u * u * s[3] + u * u * u)
	_luts[key] = [bx, by]
	return _luts[key]

func _interp(x: float, xs: PackedFloat64Array, ys: PackedFloat64Array) -> float:
	var n := xs.size()
	if x <= xs[0]: return ys[0]
	if x >= xs[n - 1]: return ys[n - 1]
	var lo := 0; var hi := n - 1
	while hi - lo > 1:
		var mid := (lo + hi) >> 1
		if xs[mid] <= x: lo = mid
		else: hi = mid
	return ys[lo] + (x - xs[lo]) / (xs[hi] - xs[lo]) * (ys[hi] - ys[lo])

func _spring(tau: float, w0: float, z: float) -> float:
	tau = maxf(tau, 0.0)
	if absf(z - 1.0) < 1e-3: return 1.0 - exp(-w0 * tau) * (1.0 + w0 * tau)
	if z < 1.0:
		var wd := w0 * sqrt(1.0 - z * z)
		return 1.0 - exp(-z * w0 * tau) * (cos(wd * tau) + z * w0 / wd * sin(wd * tau))
	var s := w0 * sqrt(z * z - 1.0); var r1 := -z * w0 + s; var r2 := -z * w0 - s
	return 1.0 - (r2 * exp(r1 * tau) - r1 * exp(r2 * tau)) / (r2 - r1)

func _seg_progress(seg: Dictionary, t: float) -> float:
	if seg["model"] == "spring":
		return _spring(t - seg["t_start"], seg["omega0"], seg["zeta"])
	var p := clampf((t - seg["t_start"]) / seg["duration"], 0.0, 1.0)
	var shape = seg["bezier"] if seg["model"] == "bezier" else PRESETS[seg["model"]]
	var l := _lut(shape)
	return _interp(p, l[0], l[1])

func _flights(ch: Dictionary, t_end: float) -> Array:
	var g: float = ch["g"]; var ground = ch.get("ground"); var e: float = ch.get("restitution", 0.0)
	var out := []
	var t0: float = ch["t_ref"]; var y0: float = ch["y0"]; var v0: float = ch["v0"]
	for k in 80:
		if ground == null or g <= 0.0:
			out.append([t0, y0, v0, INF]); break
		var disc := v0 * v0 + 2.0 * g * (ground - y0)
		var tau := (-v0 + sqrt(maxf(disc, 0.0))) / g
		var t1 := t0 + tau
		out.append([t0, y0, v0, t1])
		if t1 >= t_end or tau < 1e-4: break
		var v_in := v0 + g * tau
		t0 = t1; y0 = ground; v0 = -e * v_in
		if absf(v0) < 1e-4:
			out.append([t0, ground, 0.0, INF]); break
	return out

func eval_channel(ch: Dictionary, t: float) -> float:
	match ch["type"]:
		"const": return ch["v"]
		"linear": return ch["x0"] + ch["vx"] * t
		"samples": return _interp(t, PackedFloat64Array(ch["t"]), PackedFloat64Array(ch["v"]))
		"segments":
			var v: float = ch["initial"]
			for seg in ch["segments"]:
				if t >= seg["t_start"]:
					v = seg["from"] + (seg["to"] - seg["from"]) * _seg_progress(seg, t)
			return v
		"ballistic":
			var fl := _flights(ch, maxf(t, 30.0))
			var k := 0
			for i in fl.size():
				if fl[i][0] <= t: k = i
				else: break
			var tau := maxf(t - fl[k][0], 0.0)
			var y: float = fl[k][1] + fl[k][2] * tau + 0.5 * ch["g"] * tau * tau
			if ch.get("ground") != null: y = minf(y, ch["ground"])
			return y
	push_error("unknown channel type")
	return 0.0

func state(t: float, obj_index: int = 0) -> Dictionary:
	var c: Dictionary = SPEC["objects"][obj_index]["channels"]
	return {"x": eval_channel(c["x"], t), "y": eval_channel(c["y"], t),
			"scale": eval_channel(c["scale"], t), "opacity": eval_channel(c["opacity"], t)}
'''


def export_godot(spec, out_path):
    spec_json = json.dumps(spec)
    gd = (f"# {_header(spec, '# ').replace(chr(10), chr(10))}\n"
          "# NOT executed in Godot by the engine tests (no Godot available here); the model mirrors engine/model.py.\n"
          "# Positions are fh: multiply by the stage height. Attach to a Node and call state(t).\n"
          "extends Node\nclass_name VideoEngineScene\n\n"
          f"var SPEC: Dictionary = JSON.parse_string('''{spec_json}''')\n" + GD_MODEL % {"presets": json.dumps(PRESETS)})
    Path(out_path).write_text(gd, encoding="utf-8")
    return Path(out_path)
