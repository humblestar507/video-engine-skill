"""Scene spec: the bridge from video.json measurements to something that can be
rendered, exported and verified. Only parameters are copied, never pixels beyond
a flat colour per object and the background, so graphics/assets of the source are
not reproduced (non-flat objects are refused for rendering).

Every channel keeps a `provenance` note saying which curves/track entry it came
from and whether it is a model fit or a replay of measured samples."""
import json

import numpy as np

from . import media, probe
from .curves import arr


def _grab_full(store, t):
    geom, org = probe.geometry(store), probe.origin(store)
    ta, fr = media.grab(store.video, geom, org, t)
    if fr is None:
        raise RuntimeError(f"no frame at {t}")
    return ta, fr


def probe_object(frame, c_px, tw, th):
    """Colour, shape and size of a flat object from one full-res frame (pixel-centre coords)."""
    H, W = frame.shape[:2]
    half = 1.3 * max(tw, th)
    x0, x1 = int(max(0, c_px[0] - half)), int(min(W, c_px[0] + half + 1))
    y0, y1 = int(max(0, c_px[1] - half)), int(min(H, c_px[1] + half + 1))
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    ring = np.concatenate([crop[:2].reshape(-1, 3), crop[-2:].reshape(-1, 3),
                           crop[:, :2].reshape(-1, 3), crop[:, -2:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    dist = np.mean(np.abs(crop - bg), axis=2)
    cx, cy = c_px[0] - x0, c_px[1] - y0
    hw, hh = max(1, int(0.15 * tw)), max(1, int(0.15 * th))
    sl = (slice(int(max(0, cy - hh)), int(cy + hh + 1)), slice(int(max(0, cx - hw)), int(cx + hw + 1)))
    core = crop[sl].reshape(-1, 3)
    contrast = float(np.median(dist[sl]))
    if contrast < 6:
        return None
    mask = dist > 0.5 * contrast
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or xs.min() == 0 or ys.min() == 0 or xs.max() == mask.shape[1] - 1 or ys.max() == mask.shape[0] - 1:
        return None
    w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    fill = float(mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean())
    return {"color": [float(v) for v in np.median(core, axis=0)], "bg": [float(v) for v in bg],
            "flat_std": float(np.mean(np.std(core, axis=0))), "w_px": w, "h_px": h, "fill": fill,
            "shape": "disc" if fill < 0.9 else "rect"}


def _segments_channel(c, src):
    segs = []
    for s in c["segments"]:
        p = s["params"]
        d = {"model": s["model"], "t_start": p["t_start"], "from": s["from"], "to": s["to"]}
        if s["model"] == "spring":
            d.update(omega0=p["omega0"], zeta=p["zeta"])
        else:
            d["duration"] = p["duration"]
            if s["model"] == "bezier":
                d["bezier"] = p["bezier"]
        segs.append(d)
    return {"type": "segments", "initial": c["segments"][0]["from"], "segments": segs,
            "provenance": f"model fit from {src}", "src": "derived"}


def build(store, tid):
    tid = str(tid)
    tr = store.get("tracks", tid)
    if tr is None:
        raise KeyError(f"no track {tid}")
    prm = tr["_meta"]["params"]
    S = tr["samples"]
    t, x, y = arr(S["t"]), arr(S["x"]), arr(S["y"])
    ok = np.isfinite(x) & (arr(S["lost"]) == 0)
    pv = store.get("probe")["video"]
    W, H = pv["width"], pv["height"]
    ta, frame = _grab_full(store, tr["t_init"])
    i0 = int(np.argmin(np.abs(t - tr["t_init"])))
    c_px = (x[i0] * H - 0.5, y[i0] * H - 0.5)
    box = tr["box0_fh"]
    obj = probe_object(frame, c_px, int(round(box[2] * H)), int(round(box[3] * H)))
    warnings = []
    renderable = obj is not None and obj["flat_std"] < 20
    if obj is None:
        warnings.append("object not separable from its surroundings at t_init: positions only, no render")
    elif obj["flat_std"] >= 20:
        warnings.append(f"object is textured (core std {obj['flat_std']:.0f}); render would be a flat guess, refused")
    curves = store.get("curves") or {}
    ch = {}
    # y / x
    bal = curves.get(f"ballistic_{tid}")
    if bal and bal["axis"] == "y":
        f0 = bal["flights"][0]
        gr = float(np.mean([i["y_fh"] for i in bal["impacts"]])) if bal["impacts"] else None
        e = bal["restitution_mean"]["v"] if bal["restitution_mean"] else 0.0
        ch["y"] = {"type": "ballistic", "t_ref": f0["tref"], "y0": f0["y0"], "v0": f0["v0"], "g": bal["g"]["v"],
                   "ground": gr, "restitution": e, "provenance": f"curves.ballistic_{tid}", "src": "derived"}
        vx = bal["vx_mean_fh_s"]["v"]
        ch["x"] = {"type": "linear", "x0": float(np.median(x[ok] - vx * t[ok])), "vx": vx,
                   "provenance": f"curves.ballistic_{tid} (mean horizontal speed)", "src": "derived"}
    for name, arr_ in (("x", x), ("y", y)):
        if name in ch:
            continue
        c = curves.get(f"{tid}_{name}")
        if c and c["segments"]:
            ch[name] = _segments_channel(c, f"curves.{tid}_{name}")
        elif c or np.ptp(arr_[ok]) < 0.5 / tr["analysis_height_px"]:
            ch[name] = {"type": "const", "v": float(np.median(arr_[ok])), "src": "derived",
                        "provenance": "no motion intervals" if c else "range below 0.5 px: treated as constant (median)"}
        else:
            ch[name] = {"type": "samples", "t": [float(v) for v in t[ok]], "v": [float(v) for v in arr_[ok]],
                        "provenance": "replay of measured track samples (no model fitted)", "src": "measured"}
    for name in ("scale", "opacity"):
        c = curves.get(f"{tid}_{name}")
        if c and c["segments"]:
            ch[name] = _segments_channel(c, f"curves.{tid}_{name}")
        else:
            ch[name] = {"type": "const", "v": 1.0, "provenance": "assumed 1.0 (not analysed)", "src": "assumed"}
    ground = None
    if ch["y"]["type"] == "ballistic" and ch["y"]["ground"] is not None and obj:
        top = ch["y"]["ground"] + obj["h_px"] / H / 2
        row = int(round(top * H)) + 3
        if row < H - 2:
            gcol = np.median(frame[row:, :].reshape(-1, 3), axis=0)
            ground = {"top_fh": top, "color": [float(v) for v in gcol],
                      "provenance": "surface under the impacts; colour sampled below it. Decoration on it is not modelled"}
    dur = pv and store.get("probe")["video"]["duration"]["v"]
    spec = {
        "version": 1, "source": str(store.video), "track": tid, "name": tr.get("name", ""),
        "stage": {"w": W, "h": H, "units": "fh = fraction of stage height; x right, y down"},
        "fps": store.get("probe")["video"]["fps"]["v"], "duration": dur,
        "bg": obj["bg"] if obj else [0, 0, 0], "ground": ground,
        "objects": [{
            "id": tid, "name": tr.get("name", ""), "renderable": renderable,
            "shape": obj["shape"] if obj else None, "color": obj["color"] if obj else None,
            "size_fh": [obj["w_px"] / H, obj["h_px"] / H] if obj else None,
            "channels": ch,
            "reference": tr["ref"],
            "track_params": {k: prm[k] for k in ("box", "t_init", "t0", "t1", "ref", "update", "measure", "name", "center")},
        }],
        "warnings": warnings,
        "labels": "channel.src = derived (model fit) | measured (replayed samples) | assumed; see provenance",
    }
    return spec


def save(spec, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
