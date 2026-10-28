"""The reproduction loop: spec -> render -> analyse like the original -> compare -> correct.

update_spec moves only parameters that the analyser can measure on both videos:
new = old + (measured on original - measured on reproduction). That is a
fixed-point iteration; it converges when the analyser is nearly unbiased on the
quantity (checked in the tests), and it stops with an explanation where a
channel's structure differs (segment count or model) instead of guessing."""
import copy
from pathlib import Path

import numpy as np

from . import compare as cmp
from . import probe, render, scenes
from .curves import arr
from .store import Store


def _lin_params(store, tid):
    c = store.get("curves", f"ballistic_{tid}")
    tr = store.get("tracks", tid)
    t, x = arr(tr["samples"]["t"]), arr(tr["samples"]["x"])
    ok = np.isfinite(x) & (arr(tr["samples"]["lost"]) == 0)
    vx = c["vx_mean_fh_s"]["v"]
    return {"vx": vx, "x0": float(np.median(x[ok] - vx * t[ok]))}


def _ball_params(store, tid):
    c = store.get("curves", f"ballistic_{tid}")
    f0 = c["flights"][0]
    d = {"g": c["g"]["v"], "y0": f0["y0"], "v0": f0["v0"],
         "restitution": c["restitution_mean"]["v"] if c["restitution_mean"] else None,
         "ground": float(np.mean([i["y_fh"] for i in c["impacts"]])) if c["impacts"] else None}
    return d


def _recovered(old, meas, rel=None, abs_=None):
    """Did the analyser recover the value the spec actually used? (self-check of the measurement)"""
    tol = max(rel * abs(old) if rel else 0.0, abs_ or 0.0)
    return abs(meas - old) <= tol


def update_spec(spec, ref, test):
    """Returns (new_spec, changes, blocked)."""
    new = copy.deepcopy(spec)
    changes, blocked = [], []
    for obj in new["objects"]:
        tid = obj["id"]
        for name, ch in obj["channels"].items():
            if ch["type"] == "ballistic":
                try:
                    a, b = _ball_params(ref, tid), _ball_params(test, tid)
                except Exception as e:
                    blocked.append(f"{tid}.{name}: ballistic parameters not measurable on the reproduction ({e})")
                    continue
                bad = [k for k, (rel, ab) in {"g": (0.1, None), "restitution": (None, 0.08), "ground": (None, 0.01)}.items()
                       if b[k] is not None and ch.get(k) is not None and not _recovered(ch[k], b[k], rel, ab)]
                if bad:
                    blocked.append(f"{tid}.{name}: the analyser did not recover the inputs it was given for {bad} "
                                   f"(spec {[round(ch[k], 4) for k in bad]} vs measured {[round(b[k], 4) for k in bad]}); "
                                   "the measurement on the reproduction is not trustworthy, nothing applied")
                    continue
                for k in ("g", "y0", "v0", "restitution", "ground"):
                    if a[k] is None or b[k] is None or ch.get(k) is None:
                        continue
                    d = a[k] - b[k]
                    changes.append({"param": f"{tid}.{name}.{k}", "old": ch[k], "new": ch[k] + d, "delta": d})
                    ch[k] += d
            elif ch["type"] == "linear":
                try:
                    a, b = _lin_params(ref, tid), _lin_params(test, tid)
                except Exception as e:
                    blocked.append(f"{tid}.{name}: not measurable ({e})")
                    continue
                if not _recovered(ch["vx"], b["vx"], 0.05, 0.01):
                    blocked.append(f"{tid}.{name}: vx not recovered (spec {ch['vx']:.4f} vs measured {b['vx']:.4f}); nothing applied")
                    continue
                for k in ("vx", "x0"):
                    d = a[k] - b[k]
                    changes.append({"param": f"{tid}.{name}.{k}", "old": ch[k], "new": ch[k] + d, "delta": d})
                    ch[k] += d
            elif ch["type"] == "segments":
                ca, cb = (ref.get("curves") or {}).get(f"{tid}_{name}"), (test.get("curves") or {}).get(f"{tid}_{name}")
                if not ca or not cb:
                    blocked.append(f"{tid}.{name}: no curve fit on one of the videos")
                    continue
                if len(ca["segments"]) != len(cb["segments"]) or any(x["model"] != y["model"] for x, y in zip(ca["segments"], cb["segments"])):
                    blocked.append(f"{tid}.{name}: segment structure differs (ref {[s['model'] for s in ca['segments']]} vs "
                                   f"test {[s['model'] for s in cb['segments']]}); change the model by hand")
                    continue
                miss = [k for k, (sb, seg) in enumerate(zip(cb["segments"], ch["segments"]))
                        if abs(sb["params"]["t_start"] - seg["t_start"]) > 0.05
                        or abs(sb["to"] - seg["to"]) > 0.03 * max(1.0, abs(seg["to"]))]
                if miss:
                    blocked.append(f"{tid}.{name}: segments {miss} not recovered on the reproduction; nothing applied")
                    continue
                for k, (sa, sb, seg) in enumerate(zip(ca["segments"], cb["segments"], ch["segments"])):
                    pairs = [("t_start", sa["params"]["t_start"], sb["params"]["t_start"]),
                             ("from", sa["from"], sb["from"]), ("to", sa["to"], sb["to"])]
                    if seg["model"] == "spring":
                        pairs += [("omega0", sa["params"]["omega0"], sb["params"]["omega0"]), ("zeta", sa["params"]["zeta"], sb["params"]["zeta"])]
                    else:
                        pairs.append(("duration", sa["params"]["duration"], sb["params"]["duration"]))
                    for key, va, vb in pairs:
                        d = va - vb
                        changes.append({"param": f"{tid}.{name}.seg{k}.{key}", "old": seg[key], "new": seg[key] + d, "delta": d})
                        seg[key] += d
                ch["initial"] = ch["segments"][0]["from"]
    return new, changes, blocked


def verify(ref, spec, workdir, thr=None, tag="it0", apply=False, ssim=True, tracks=None):
    """One round. Returns dict(report, spec_next, changes, blocked, video)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"{tag}.mp4"
    render.render(spec, out, times=probe.pts(ref), fps=ref.get("probe")["video"]["fps"]["v"])
    test = Store(out)
    probe.run(test)
    scenes.run(test)
    rep = cmp.compare(ref, test, thr=thr, label=tag, ssim=ssim, tracks=tracks or [o["id"] for o in spec["objects"]])
    res = {"report": rep, "video": str(out), "spec_next": None, "changes": [], "blocked": [], "test_store": test}
    if apply:
        res["spec_next"], res["changes"], res["blocked"] = update_spec(spec, ref, test)
    return res
