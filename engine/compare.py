"""Layer 3: how far apart are two videos, in numbers.

The test video (a reproduction, a re-encode, ...) is re-analysed with exactly the
parameters the reference was analysed with (track boxes, curve channels, state
windows), then compared on:
  tracks  - distance in fh and in reference units, bias, time lag, worst windows
  params  - every fitted parameter, ranked by error / tolerance ("z"; z>1 is out)
  states  - boundary times and pose sequence
  events  - cuts
  ssim    - structural similarity per frame pair (informative only: it also sees
            everything the spec does not model)
A verdict compares agreed thresholds against measured values; the thresholds are
arguments and the report says whether they were defaults."""
import time

import cv2
import numpy as np

from . import media, probe
from .curves import arr
from .store import D, M

VERSION = "1"
DEFAULT_THR = {"track_p95_fh": 0.01}


# ---------------------------------------------------------------- re-analysis

def reanalyze(ref, test, tracks=None):
    """Run on `test` the same track/curves/states as stored in `ref`. Returns notes."""
    from . import curves, states, track
    probe.run(test)
    notes = []
    for tid, t in (ref.get("tracks") or {}).items():
        if tracks and tid not in tracks:
            continue
        p = t["_meta"]["params"]
        track.run(test, tid, p["box"], p["t_init"], name=p.get("name", ""), ref_fh=p.get("ref"), t0=p.get("t0"),
                  t1=p.get("t1"), update=p.get("update", 0.0), do_measure=p.get("measure", True), center=p.get("center", "ncc"))
    for key, c in (ref.get("curves") or {}).items():
        try:
            if key.startswith("ballistic_"):
                curves.run_ballistic(test, key.split("_", 1)[1], axis=c["axis"])
            else:
                tid, name = key.rsplit("_", 1)
                curves.run_curve(test, tid, name)
        except Exception as e:  # report, do not hide
            notes.append(f"curves {key} could not be computed on the test video: {e}")
    for sid, s in (ref.get("states") or {}).items():
        p = s["_meta"]["params"]
        try:
            states.run(test, sid, p["box"], track_id=p.get("track"), t_init=p.get("t_init", 0.0), t0=p.get("t0"),
                       t1=p.get("t1"), thr=p.get("thr"), max_side=p.get("max_side", 96))
        except Exception as e:
            notes.append(f"states {sid} could not be computed on the test video: {e}")
    return notes


# ---------------------------------------------------------------- tracks

def compare_tracks(ref, test, tid):
    a, b = ref.get("tracks", tid), test.get("tracks", tid)
    if not a or not b:
        return None
    A, B = a["samples"], b["samples"]
    ta, xa, ya = arr(A["t"]), arr(A["x"]), arr(A["y"])
    tb, xb, yb = arr(B["t"]), arr(B["x"]), arr(B["y"])
    oka = np.isfinite(xa) & (arr(A["lost"]) == 0)
    okb = np.isfinite(xb) & (arr(B["lost"]) == 0)
    m = oka & (ta >= tb[okb][0]) & (ta <= tb[okb][-1])
    xi, yi = np.interp(ta, tb[okb], xb[okb]), np.interp(ta, tb[okb], yb[okb])
    err = np.hypot(xa - xi, ya - yi)[m]
    tm = ta[m]
    refv = a["ref"]["fh"]
    # lag: shift of the test timeline that minimises the error
    best = (1e9, 0.0)
    for d in np.arange(-0.1, 0.1001, 0.001):
        xx, yy = np.interp(ta[m] + d, tb[okb], xb[okb]), np.interp(ta[m] + d, tb[okb], yb[okb])
        r = float(np.sqrt(np.mean((xa[m] - xx) ** 2 + (ya[m] - yy) ** 2)))
        if r < best[0]:
            best = (r, d)
    # worst windows: contiguous stretches above half the peak error
    thr = 0.5 * err.max() if len(err) else 0
    wins, i = [], 0
    while i < len(err):
        if err[i] > thr and err[i] > 0:
            j = i
            while j + 1 < len(err) and err[j + 1] > thr:
                j += 1
            wins.append({"t0": float(tm[i]), "t1": float(tm[j]), "max_fh": float(err[i:j + 1].max()), "mean_fh": float(err[i:j + 1].mean())})
            i = j + 1
        else:
            i += 1
    wins = sorted(wins, key=lambda w: -w["max_fh"])[:3]
    return {
        "samples": int(m.sum()),
        "rms_fh": float(np.sqrt(np.mean(err ** 2))), "p95_fh": float(np.percentile(err, 95)), "max_fh": float(err.max()),
        "rms_ref": float(np.sqrt(np.mean(err ** 2)) / refv), "max_ref": float(err.max() / refv),
        "ref_units": a["ref"]["src"],
        "bias_fh": [float(np.mean((xi - xa)[m])), float(np.mean((yi - ya)[m]))],
        "test_lag_ms": float(best[1] * 1000), "rms_fh_after_lag": float(best[0]),
        "lag_reliable": bool(best[0] < 0.7 * float(np.sqrt(np.mean(err ** 2))) and abs(best[1]) < 0.095),
        "worst_windows": wins,
    }


# ---------------------------------------------------------------- parameters

def params_of(store):
    """Flat {name: (value, tolerance, unit)} of every fitted parameter in curves."""
    out = {}
    fps = (store.get("probe")["video"]["fps"]["v"] or 30.0)
    frame = 1.0 / fps
    for key, c in (store.get("curves") or {}).items():
        if key.startswith("ballistic_"):
            out[f"{key}.g"] = (c["g"]["v"], 0.01 * abs(c["g"]["v"]), "fh/s2")
            if c["restitution_mean"]:
                out[f"{key}.restitution"] = (c["restitution_mean"]["v"], 0.02, "")
            out[f"{key}.vx"] = (c["vx_mean_fh_s"]["v"], 0.01 * abs(c["vx_mean_fh_s"]["v"]) + 1e-3, "fh/s")
            for k, im in enumerate(c["impacts"]):
                out[f"{key}.impact{k}.t"] = (im["t"], frame, "s")
            out[f"{key}.flights"] = (float(len(c["flights"])), 0.5, "")
            f0 = c["flights"][0]
            out[f"{key}.y0"] = (f0["y0"], 0.004, "fh")
            out[f"{key}.v0"] = (f0["v0"], 0.05, "fh/s")
        else:
            # ratios come from 8-bit contrast / bbox areas: resolution ~1 %; positions 0.4 % of frame height
            lvl_tol = 0.01 if key.endswith(("scale", "opacity")) else 0.004
            for k, s in enumerate(c["segments"]):
                p = s["params"]
                base = f"{key}.seg{k}"
                out[f"{base}.t_start"] = (p["t_start"], frame, "s")
                out[f"{base}.from"] = (s["from"], lvl_tol, "")
                out[f"{base}.to"] = (s["to"], lvl_tol, "")
                if s["model"] == "spring":
                    out[f"{base}.omega0"] = (p["omega0"], 0.02 * p["omega0"], "1/s")
                    out[f"{base}.zeta"] = (p["zeta"], 0.03 * p["zeta"], "")
                else:
                    out[f"{base}.duration"] = (p["duration"], frame, "s")
                out[f"{base}.model"] = (s["model"], None, "")
    return out


def compare_params(ref, test):
    pa, pb = params_of(ref), params_of(test)
    rows = []
    for name, (va, tol, unit) in pa.items():
        if name not in pb:
            rows.append({"name": name, "ref": va, "test": None, "z": float("inf"), "note": "missing in test"})
            continue
        vb = pb[name][0]
        if tol is None:
            rows.append({"name": name, "ref": va, "test": vb, "z": 0.0 if va == vb else float("inf"),
                         "note": "" if va == vb else "model differs"})
        else:
            rows.append({"name": name, "ref": va, "test": vb, "delta": vb - va, "z": abs(vb - va) / max(tol, 1e-12),
                         "tol": tol, "unit": unit})
    for name in pb:
        if name not in pa:
            rows.append({"name": name, "ref": None, "test": pb[name][0], "z": float("inf"), "note": "extra in test"})
    # numeric disagreements first (they say what to change); structural ones (missing impact, model differs) after
    return sorted(rows, key=lambda r: (not np.isfinite(r["z"]), -(r["z"] if np.isfinite(r["z"]) else 0)))


# ---------------------------------------------------------------- states / events / ssim

def compare_states(ref, test, sid):
    a, b = ref.get("states", sid), test.get("states", sid)
    if not a or not b:
        return None
    ta, tb = a["timeline"], b["timeline"]

    def canon(tl):
        m, out = {}, []
        for e in tl:
            out.append(m.setdefault(e["pose"], len(m)))
        return out
    same_n = len(ta) == len(tb)
    out = {"intervals": [len(ta), len(tb)], "poses": [len(a["palette"]), len(b["palette"])],
           "sequence_equal": same_n and canon(ta) == canon(tb)}
    if same_n:
        out["max_boundary_diff_ms"] = float(max(abs(x["t0"] - y["t0"]) for x, y in zip(ta, tb)) * 1000)
    return out


def compare_events(ref, test):
    ca = [e["t"] for e in ref.get("scenes")["events"] if e["type"] == "cut"]
    cb = [e["t"] for e in test.get("scenes")["events"] if e["type"] == "cut"] if test.get("scenes") else []
    out = {"cuts": [len(ca), len(cb)]}
    if ca and len(ca) == len(cb):
        out["max_cut_diff_ms"] = float(max(abs(x - y) for x, y in zip(ca, cb)) * 1000)
    return out


def ssim_gray(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    saa, sbb, sab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(s.mean())


def ssim_series(ref, test, width=240):
    ga, gb = probe.geometry(ref), probe.geometry(test)
    ia = media.iter_frames(ref.video, ga, probe.origin(ref), width=width, gray=True)
    ib = media.iter_frames(test.video, gb, probe.origin(test), width=width, gray=True)
    cur = next(ib)
    nxt = next(ib, None)
    ts, ss = [], []
    for t, fa in ia:
        while nxt is not None and abs(nxt[0] - t) <= abs(cur[0] - t):
            cur, nxt = nxt, next(ib, None)
        fb = cur[1]
        if fb.shape != fa.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]), interpolation=cv2.INTER_AREA)
        ts.append(t)
        ss.append(ssim_gray(fa, fb))
    ss = np.array(ss)
    w = int(np.argmin(ss))
    return {"mean": float(ss.mean()), "min": float(ss.min()), "min_at_s": float(ts[w]), "frames": len(ss),
            "note": "also penalises everything the spec does not model (decoration, texture); informative, not in the verdict"}


# ---------------------------------------------------------------- top level

def compare(ref, test, thr=None, label=None, ssim=True, tracks=None):
    t_start = time.time()
    thr_used = dict(DEFAULT_THR)
    defaults = True
    if thr:
        thr_used.update(thr)
        defaults = False
    notes = reanalyze(ref, test, tracks)
    tracks = {tid: compare_tracks(ref, test, tid) for tid in (ref.get("tracks") or {}) if not tracks or tid in tracks}
    tracks = {k: v for k, v in tracks.items() if v}
    params = compare_params(ref, test)
    st = {sid: compare_states(ref, test, sid) for sid in (ref.get("states") or {})}
    crit = []
    for tid, m in tracks.items():
        crit.append({"name": f"track {tid} p95 distance (fh)", "value": m["p95_fh"], "thr": thr_used["track_p95_fh"],
                     "ok": m["p95_fh"] <= thr_used["track_p95_fh"]})
    numeric_bad = [r["name"] for r in params if np.isfinite(r["z"]) and r["z"] > 1]
    if params:
        crit.append({"name": "fitted parameters within tolerance (numeric ones)", "value": len(numeric_bad), "thr": 0, "ok": not numeric_bad})
    for sid, m in st.items():
        if m:
            crit.append({"name": f"states {sid}: same pose sequence", "value": m["sequence_equal"], "thr": True, "ok": bool(m["sequence_equal"])})
    out = {
        "test": str(test.video), "tracks": tracks, "params_ranked": params[:25],
        "params_out_of_tolerance": [r["name"] for r in params if r["z"] > 1],
        "states": {k: v for k, v in st.items() if v},
        "events": compare_events(ref, test),
        "ssim": ssim_series(ref, test) if ssim else None,
        "verdict": {"pass": bool(crit) and all(c["ok"] for c in crit), "criteria": crit,
                    "thresholds_are_defaults": defaults,
                    "note": "thresholds must be agreed with the user" if defaults else "user thresholds"},
        "notes": notes, "warnings": [],
    }
    out["summary_line"] = ("PASS" if out["verdict"]["pass"] else "FAIL") + " | " + "; ".join(
        f"track {tid}: p95 {m['p95_fh']:.4f} fh ({m['p95_fh'] * ref.get('tracks', tid)['analysis_height_px']:.2f} px), lag {m['test_lag_ms']:+.0f} ms{'' if m['lag_reliable'] else ' (unreliable)'}" for tid, m in tracks.items()) \
        + (f" | worst param: {params[0]['name']} z={params[0]['z']:.1f}" if params and params[0]["z"] > 1 else " | all fitted params within tolerance")
    label = label or test.video.stem
    ref.put(("compare", label), out, VERSION, {"thr": thr_used, "test": str(test.video)}, deps=[("probe",)], t_start=t_start)
    return out
