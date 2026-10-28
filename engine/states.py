"""Layer 2: which pose/look an object has when, as a held-state timeline.

A window of fixed size follows the object (a track, or a fixed box), is cut out
with sub-pixel interpolation each frame, and compared with a shift-tolerant
normalised MSE. Change points split the clip into held segments; segments are
merged into a palette of distinct poses; repeating pose sequences are reported
as cycles with per-frame hold times (the thing animators call "on 4s").

Limitation (kept visible in the output): everything inside the window counts as
appearance, including background that changes behind the object."""
import hashlib
import json
import time

import cv2
import numpy as np

from . import track as trk
from .store import D, M

VERSION = "3"
BLUR = 1.0  # px, before comparing
FLOOR = 4e-4  # normalised MSE below which two crops are the same (sensor/compression noise)


def _crops(F, centers, w, h, max_side):
    out = []
    for fr, c in zip(F, centers):
        cr = cv2.getRectSubPix(fr, (w, h), (float(c[0]), float(c[1])))
        # light blur: sub-pixel misalignment and anti-aliasing must not read as a pose change
        out.append(cv2.GaussianBlur(cr, (0, 0), BLUR))
    arr = np.array(out, np.float32)
    s = 1
    while max(w, h) / s > max_side:
        s += 1
    if s > 1:
        arr = np.array([cv2.resize(a, (w // s, h // s), interpolation=cv2.INTER_AREA) for a in arr], np.float32)
    return arr, s


def dist(a, b, tol=1):
    """min over +-tol px shifts of mean squared difference, 0..1 (images in 0..255)."""
    H, W = a.shape[:2]
    m = tol
    ref = a[m:H - m, m:W - m]
    best = 1e9
    for dy in range(-tol, tol + 1):
        for dx in range(-tol, tol + 1):
            cand = b[m + dy:H - m + dy, m + dx:W - m + dx]
            d = float(np.mean((ref - cand) ** 2))
            if d < best:
                best = d
    return best / 255.0 ** 2


def cycles_of(seq, min_rep=2):
    """Repeating subsequences: [{period, pattern, repeats, i0, i1}] without overlaps."""
    found = []
    n = len(seq)
    for p in range(2, min(9, n // 2 + 1)):
        eq = [seq[i] == seq[i + p] for i in range(n - p)]
        i = 0
        while i < len(eq):
            if eq[i]:
                j = i
                while j + 1 < len(eq) and eq[j + 1]:
                    j += 1
                L = j - i + 1
                if L >= p and len(set(seq[i:i + p])) > 1:
                    found.append((L + p, p, i, i + L + p - 1))
                i = j + 1
            else:
                i += 1
    found.sort(key=lambda f: (-f[0], f[1]))
    used = np.zeros(n, bool)
    out = []
    for cover, p, i0, i1 in found:
        if used[i0:i1 + 1].any():
            continue
        used[i0:i1 + 1] = True
        out.append({"period": p, "pattern": seq[i0:i0 + p], "repeats": (i1 - i0 + 1) / p, "i0": i0, "i1": i1})
    return sorted(out, key=lambda c: c["i0"])


def run(store, sid, box_fh, track_id=None, t_init=0.0, t0=None, t1=None, thr=None, max_side=96, force=False):
    sid = str(sid)
    params = {"box": [round(b, 5) for b in box_fh], "track": track_id, "t_init": t_init, "t0": t0, "t1": t1,
              "thr": thr, "max_side": max_side}
    deps = [("probe",)] + ([("tracks", str(track_id))] if track_id is not None else [])
    if not force and store.fresh(("states", sid), VERSION, params):
        return store.get("states", sid), True
    t_start = time.time()
    dur = store.get("probe")["video"]["duration"]["v"]
    t0 = 0.0 if t0 is None else t0
    t1 = dur if t1 is None else t1
    T, F, H = trk.load_range(store, t0, t1)
    Ff = [f.astype(np.float32) for f in F]
    x, y, w, h = [b * H for b in box_fh]
    w, h = max(8, int(round(w))), max(8, int(round(h)))
    c_box = np.array([x + w / 2 - 0.5, y + h / 2 - 0.5])
    warnings = []
    if track_id is not None:
        tr = store.get("tracks", str(track_id))
        S = tr["samples"]
        tt = trk_arr(S["t"])
        tx, ty = trk_arr(S["x"]) * H - 0.5, trk_arr(S["y"]) * H - 0.5
        i0 = int(np.argmin(np.abs(tt - t_init)))
        off = c_box - np.array([tx[i0], ty[i0]])
        centers = np.stack([np.interp(T, tt, tx), np.interp(T, tt, ty)], axis=1) + off
        if tr["quality"]["lost"]:
            warnings.append(f"track has {tr['quality']['lost']} lost frames; their anchors are predictions")
        anchor = {"kind": "track", "track": str(track_id)}
    else:
        centers = np.tile(c_box, (len(T), 1))
        anchor = {"kind": "fixed box"}
    crops, scale = _crops(Ff, centers, w, h, max_side)
    n = len(T)
    d = np.array([0.0] + [dist(crops[i - 1], crops[i]) for i in range(1, n)])
    p20 = float(np.percentile(d[1:], 20)) if n > 5 else 0.0

    def analyse(thr_):
        starts = [0] + [i for i in range(1, n) if d[i] > thr_]
        segs = [(starts[k], (starts[k + 1] if k + 1 < len(starts) else n) - 1) for k in range(len(starts))]
        reps = [np.median(crops[a:b + 1], axis=0) for a, b in segs]
        pose_of, poses, margins = [], [], []
        for rep in reps:
            ds = [dist(rep, pr) for pr in poses]
            if ds and min(ds) < thr_:
                pose_of.append(int(np.argmin(ds)))
                margins.append(min(ds))
            else:
                poses.append(rep)
                pose_of.append(len(poses) - 1)
        return starts, segs, pose_of, poses

    method, plateau = "user", None
    if thr is not None:
        thr_used = thr
    else:
        # the natural scale is where the result stops changing: longest plateau of (poses, intervals) over thresholds
        cands = np.geomspace(2e-4, 6e-3, 18)
        res = []
        for c in cands:
            st_, sg_, po_, pz_ = analyse(c)
            res.append((len(pz_), len(sg_)))
        best, k = (0, 0, 0), 0
        while k < len(cands):
            j = k
            while j + 1 < len(cands) and res[j + 1] == res[k]:
                j += 1
            cand = (j - k + 1, res[k][0], k)
            if cand[:2] > best[:2]:
                best = cand
            k = j + 1
        length, _, k0 = best
        lo, hi = cands[k0], cands[k0 + length - 1]
        thr_used = float(np.sqrt(lo * hi))
        method = "plateau"
        plateau = {"from": float(lo), "to": float(hi), "candidates": int(length), "of": len(cands), "poses": res[k0][0]}
        if length < 3:
            warnings.append(f"no stable threshold: pose count changes with every candidate (plateau of {length}); "
                            "result is a guess, set --thr and check the palette image")
    starts, segs, pose_of, poses = analyse(thr_used)
    dtm = float(np.median(np.diff(T))) if n > 1 else 1 / 30
    timeline = []
    for (a, b), k in zip(segs, pose_of):
        ta = float(T[a])
        tb = float(T[b + 1]) if b + 1 < n else float(T[b] + dtm)
        timeline.append({"pose": f"P{k}", "t0": ta, "t1": tb, "frames": int(b - a + 1), "dur": tb - ta,
                         "f0": int(a), "f1": int(b)})
    # tiny held segments right at the boundary of two others can be blends: flag, do not delete
    for e in timeline:
        if e["frames"] == 1 and 0 < e["f0"] < n - 1:
            e["note"] = "single frame: transition blend or a genuine 1-frame pose"
    seq = [e["pose"] for e in timeline]
    cyc = []
    for c in cycles_of(seq):
        a, b = timeline[c["i0"]], timeline[c["i1"]]
        hold = [e["frames"] for e in timeline[c["i0"]:c["i1"] + 1]]
        cyc.append({"pattern": c["pattern"], "repeats": c["repeats"], "t0": a["t0"], "t1": b["t1"],
                    "hold_frames": hold[:c["period"]], "hold_frames_median": float(np.median(hold)),
                    "hold_s_median": float(np.median([e["dur"] for e in timeline[c["i0"]:c["i1"] + 1]])),
                    "cycle_s": float(sum(e["dur"] for e in timeline[c["i0"]:c["i0"] + c["period"]])),
                    "src": "derived"})
    palette = []
    for k, rep in enumerate(poses):
        use = [e for e in timeline if e["pose"] == f"P{k}"]
        palette.append({"id": f"P{k}", "intervals": len(use), "frames": int(sum(e["frames"] for e in use)),
                        "time_s": float(sum(e["dur"] for e in use)), "first_t": use[0]["t0"]})
    # palette image for Claude to look at (one image, nearest-neighbour upscaled)
    tag = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    img_path = store.file(f"views/poses-{sid}-{tag}.png")
    zoom = max(1, int(np.ceil(160 / max(rep.shape[:2]))))
    tiles = []
    for k, rep in enumerate(poses):
        im = cv2.resize(np.clip(rep, 0, 255).astype(np.uint8), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        bar = np.full((22, im.shape[1], 3), 24, np.uint8)
        cv2.putText(bar, f"P{k} x{palette[k]['frames']}f", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([im, bar]))
    hh = max(t.shape[0] for t in tiles)
    row = [np.pad(t, ((0, hh - t.shape[0]), (3, 3), (0, 0)), constant_values=48) for t in tiles]
    cv2.imwrite(str(img_path), np.hstack(row))
    out = {
        "anchor": anchor, "window_px": [w, h], "crop_downscale": scale, "frames": n,
        "palette": palette, "timeline": timeline, "cycles": cyc, "palette_image": str(img_path),
        "threshold": {"used": thr_used, "noise_p20": p20, "method": method, "plateau": plateau,
                      "src": "user" if thr is not None else "auto",
                      "note": "normalised MSE, shift-tolerant +-1px; poses closer than this are merged"},
        "separation": D(float(min([d[s] for s in starts[1:]] + [1.0])) / thr_used if len(starts) > 1 else None, 0.6,
                        note="smallest accepted change / threshold; close to 1 means a marginal pose difference"),
        "summary_line": f"{len(poses)} poses, {len(timeline)} intervals, {len(cyc)} cycle(s)"
                        + ("; " + ", ".join(f"{'-'.join(c['pattern'])} x{c['repeats']:.0f} hold {c['hold_frames_median']:.0f}f" for c in cyc) if cyc else ""),
        "limitations": "everything inside the window is appearance, including a changing background",
        "warnings": warnings,
    }
    return store.put(("states", sid), out, VERSION, params, deps=deps, t_start=t_start), False


def trk_arr(x):
    return np.array([np.nan if v is None else v for v in x], float)
