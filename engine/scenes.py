"""Layer 0: cuts, gradual transitions, fades, flashes, per-scene keyframes and
a cheap activity signal.

Per-frame signals (t, pixel diff, changed-pixel fraction, histogram distance,
luma) go to signals_l0.npz; video.json keeps only scenes and events."""
import time

import cv2
import numpy as np

from . import media, probe
from .store import D, M

VERSION = "5"
ANALYSIS_W = 240
CHG_MOVING = 0.0005  # changed-pixel fraction that counts as "something moves"


def _local_median(x, i, half):
    lo, hi = max(0, i - half), min(len(x), i + half + 1)
    w = np.concatenate([x[lo:i], x[i + 1:hi]])
    return float(np.median(w)) if len(w) else 0.0


def _ncc(a, b):
    """Brightness/contrast invariant similarity of two gray frames (1 = same picture)."""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    sa, sb = a.std(), b.std()
    if sa < 2 and sb < 2:
        return 1.0  # both flat (e.g. black)
    if sa < 2 or sb < 2:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def compute_signals(store):
    geom, org = probe.geometry(store), probe.origin(store)
    ts, dpix, dhist, luma, ncc, hists, dh64 = [], [], [], [], [], [], []
    prev_g = None
    for t, fr in media.iter_frames(store.video, geom, org, width=ANALYSIS_W):
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        ts.append(t)
        luma.append(g.mean() / 255.0)
        if prev_g is None:
            dpix.append(0.0)
            dhist.append(0.0)
            ncc.append(1.0)
            dh64.append(np.zeros(64, np.float32))
        else:
            d = cv2.absdiff(g, prev_g)
            dpix.append(float(np.mean(d)) / 255.0)
            dhist.append(float(cv2.compareHist(hists[-1], hist, cv2.HISTCMP_BHATTACHARYYA)))
            ncc.append(_ncc(g, prev_g))
            # distribution of |diff| (4-level bins) to set the noise floor afterwards
            dh64.append(np.bincount(d.ravel() >> 2, minlength=64).astype(np.float32) / d.size)
        prev_g = g
        hists.append(hist)
    sig = {k: np.array(v, np.float64) for k, v in
           dict(t=ts, dpix=dpix, dhist=dhist, luma=luma, ncc=ncc).items()}
    sig["hist"] = np.array(hists, np.float32)
    sig["diffhist"] = np.array(dh64, np.float32)
    sig["dchg"], sig["noise_thr"] = changed_fraction(sig["diffhist"])
    np.savez_compressed(store.file("signals_l0.npz"), **sig)
    return sig


def changed_fraction(diffhist):
    """Fraction of pixels whose |diff| is above the video-wide noise floor.
    Floor = 4 x robust sigma from the quietest 10% of frames, at least 10/255."""
    if len(diffhist) < 2:
        return np.zeros(len(diffhist)), 10.0
    cdf = np.cumsum(diffhist[1:], axis=1)
    med_bin = np.argmax(cdf >= 0.5, axis=1) * 4 + 2  # per-frame median |diff|
    sigma = 1.4826 * float(np.percentile(med_bin, 10))
    thr = max(10.0, 4 * sigma)
    b = int(np.ceil(thr / 4))
    out = np.concatenate([[0.0], diffhist[1:, b:].sum(axis=1)])
    return out, thr


def _bhatt(a, b):
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def detect(sig, fps, min_scene_s=0.2):
    """Returns (boundary frame indices, events)."""
    t, dp, dh, lu, H = sig["t"], sig["dpix"], sig["dhist"], sig["luma"], sig["hist"]
    n = len(t)
    events = []
    half = max(3, int(round(0.5 * fps)))

    nc = sig["ncc"]
    # 0. brightness ramps with unchanged structure: fades (to/from black or any level)
    dl = np.diff(lu, prepend=lu[0])
    ramp = (np.abs(dl) > 0.003) & ((nc > 0.9) | (lu < 0.06))  # near black NCC is quantization noise
    fade = np.zeros(n, bool)
    i = 1
    while i < n:
        if ramp[i]:
            j = i
            while j + 1 < n and ramp[j + 1] and np.sign(dl[j + 1]) == np.sign(dl[i]):
                j += 1
            if j - i + 1 >= 3:
                fade[i:j + 1] = True
                down = dl[i] < 0
                kind = ("fade_out" if lu[j] < 0.05 else "dim") if down else ("fade_in" if lu[i - 1] < 0.05 else "brighten")
                events.append({"type": kind, "t": float(t[i - 1]), "t_end": float(t[j]), "conf": 0.8,
                               "src": "measured", "luma": [round(float(lu[i - 1]), 3), round(float(lu[j]), 3)]})
            i = j + 1
        else:
            i += 1

    # 1. sustained runs of palette change (>= 3 frames): dissolves, not hard cuts
    elevated = (dh > 0.1) & ~fade  # histogram only: motion (pans) changes pixels, not the palette
    gradual = np.zeros(n, bool)
    bounds = []
    i = 1
    while i < n:
        if elevated[i]:
            j = i
            while j + 1 < n and (elevated[j + 1] or (j + 2 < n and elevated[j + 2])):
                j += 1
            if j - i + 1 >= 3:
                gradual[i:j + 1] = True
                lum = lu[i - 1:j + 2]
                trend = np.sign(np.diff(lum))
                mono = abs(trend.sum()) >= 0.8 * len(trend)
                before, after = H[i - 1], H[min(n - 1, j + 1)]
                to_black = lu[min(n - 1, j + 1)] < 0.03
                from_black = lu[i - 1] < 0.03
                kind = "fade_out" if mono and to_black else "fade_in" if mono and from_black else                     "dissolve" if mono else "gradual"
                d_ba = _bhatt(before, after)
                ev = {"type": kind, "t": float(t[i]), "t_end": float(t[j]), "conf": 0.6,
                      "src": "derived", "content_change": round(d_ba, 3)}
                if kind in ("dissolve", "gradual") and d_ba > 0.35:
                    mid = (i + j) // 2
                    bounds.append((mid, 0.6))
                    ev["boundary_at"] = float(t[mid])
                elif kind in ("dissolve", "gradual"):
                    ev["note"] = "sustained change but same content before/after (fast motion?)"
                events.append(ev)
            i = j + 1
        else:
            i += 1

    # 2. isolated spikes: hard cuts and flashes
    cands = []
    for i in range(1, n):
        if gradual[i] or fade[i] or nc[i] > 0.9:
            continue  # same structure: lighting/fade change, not a cut
        mh, mp = _local_median(dh, i, half), _local_median(dp, i, half)
        r_h = dh[i] / (mh + 0.02)
        r_p = dp[i] / (mp + 0.01)
        if (dh[i] > 0.25 and r_h > 3.0) or (dp[i] > 0.12 and r_p > 4.0 and dh[i] > 0.08):
            cands.append((i, max(r_h / 3.0, r_p / 4.0)))
    min_gap = max(2, int(round(min_scene_s * fps)))
    k = 0
    while k < len(cands):
        i, s = cands[k]
        j = cands[k + 1][0] if k + 1 < len(cands) else None
        if j is not None and j - i <= min_gap:
            if _bhatt(H[i - 1], H[j]) < 0.15:  # picture returns: flash / glitch frames
                events.append({"type": "flash", "t": float(t[i]), "t_end": float(t[j - 1]),
                               "conf": 0.75, "src": "measured"})
            else:  # two jumps too close: keep the stronger as the cut
                bi, bs = max(cands[k], cands[k + 1], key=lambda c: c[1])
                bounds.append((bi, min(1.0, 0.3 + 0.15 * bs)))
                events.append({"type": "cut", "t": float(t[bi]), "frame": int(bi),
                               "conf": round(min(1.0, 0.3 + 0.15 * bs), 2), "src": "measured",
                               "note": "second jump within min_scene_s merged"})
            k += 2
            continue
        bounds.append((i, min(1.0, 0.4 + 0.15 * s)))
        events.append({"type": "cut", "t": float(t[i]), "frame": int(i),
                       "conf": round(min(1.0, 0.4 + 0.15 * s), 2), "src": "measured",
                       "dhist": round(float(dh[i]), 3), "dpix": round(float(dp[i]), 3)})
        k += 1

    # 3. black intervals (start of a black run inside the video splits a scene)
    black = lu < 0.03
    i = 0
    while i < n:
        if black[i]:
            j = i
            while j + 1 < n and black[j + 1]:
                j += 1
            events.append({"type": "black", "t": float(t[i]), "t_end": float(t[j]),
                           "conf": 0.9, "src": "measured"})
            if 0 < i and j < n - 1 and not any(abs(b - i) <= min_gap for b, _ in bounds):
                bounds.append((i, 0.7))
            i = j + 1
        else:
            i += 1
    events.sort(key=lambda e: e["t"])
    return sorted(set(b for b, _ in bounds)), events


def keyframes(t, act, i0, i1, max_k=8):
    """Adaptive keyframes: quantiles of a blend of cumulative activity and time,
    so moving parts get more frames and static parts still get some."""
    seg_t = t[i0:i1]
    if len(seg_t) <= 2:
        return [float(x) for x in seg_t]
    dur = seg_t[-1] - seg_t[0]
    k = int(np.clip(3 + dur / 4.0, 3, max_k))
    a = act[i0:i1].copy()
    a[0] = 0
    ca = np.cumsum(a)
    ca = ca / ca[-1] if ca[-1] > 0 else np.zeros_like(ca)
    ct = (seg_t - seg_t[0]) / dur if dur > 0 else np.zeros_like(seg_t)
    c = 0.6 * ca + 0.4 * ct
    qs = np.linspace(0, 1, k)
    idx = np.unique(np.clip(np.searchsorted(c, qs), 0, len(seg_t) - 1))
    return [float(seg_t[j]) for j in idx]


def run(store, force=False, min_scene_s=0.2):
    params = {"w": ANALYSIS_W, "min_scene_s": min_scene_s}
    if not force and store.fresh("scenes", VERSION, params):
        return store.get("scenes"), True
    t_start = time.time()
    pr = store.get("probe")
    sig = compute_signals(store)
    t = sig["t"]
    warnings = []
    n_probe = pr["video"]["frames"]["v"]
    if len(t) != n_probe:
        warnings.append(f"decoded {len(t)} frames but container lists {n_probe}; using decoded pts")
        np.save(store.file("pts.npy"), t)
    fps = pr["video"]["fps"]["v"] or 30
    cut_idx, events = detect(sig, fps, min_scene_s)
    dur = pr["video"]["duration"]["v"]
    bounds = [0] + [b for b in cut_idx if 0 < b < len(t)] + [len(t)]
    chg = sig["dchg"]
    noise_thr = float(sig["noise_thr"])
    scenes = []
    for s in range(len(bounds) - 1):
        i0, i1 = bounds[s], bounds[s + 1]
        if i1 <= i0:
            continue
        t0 = float(t[i0])
        t1 = float(t[i1]) if i1 < len(t) else float(dur)
        inner = chg[i0 + 1:i1]
        scenes.append({
            "id": len(scenes), "t0": t0, "t1": t1, "dur": t1 - t0, "frames": int(i1 - i0),
            "activity": M(float(inner.mean()) if len(inner) else 0.0, 0.9,
                          note="mean changed-pixel fraction per frame"),
            "static": D(bool(len(inner) == 0 or inner.max() < CHG_MOVING), 0.8),
            "luma": float(sig["luma"][i0:i1].mean()),
            "keyframes": keyframes(t, chg, i0, i1),
        })
    # activity segments (where something moves), never spanning a boundary
    moving = chg > CHG_MOVING
    bset = set(cut_idx)
    segs, i = [], 0
    while i < len(t):
        if moving[i] and i not in bset:
            j = i
            while j + 1 < len(t) and moving[j + 1] and (j + 1) not in bset:
                j += 1
            if t[j] - t[i] >= 0.1:
                segs.append({"t0": float(t[i - 1]), "t1": float(t[j]),
                             "peak": float(chg[i:j + 1].max()), "mean": float(chg[i:j + 1].mean())})
            i = j + 1
        else:
            i += 1
    out = {"count": len(scenes), "scenes": scenes, "events": events,
           "activity_segments": {"thr": CHG_MOVING, "items": segs[:200], "total": len(segs),
                                 "note": "changed-pixel fraction > thr; t0 is the frame before motion starts",
                                 "noise_floor_diff": noise_thr},
           "signals_file": "signals_l0.npz", "warnings": warnings}
    return store.put("scenes", out, VERSION, params, deps=[("probe",)], t_start=t_start), False
