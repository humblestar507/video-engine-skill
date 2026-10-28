"""Layer 2: follow one object the user confirmed, in fh units, on PTS time.

Method (classical, no weights): normalized cross-correlation of the user's box
against a window predicted by constant velocity, with parabolic sub-pixel
refinement. NCC is invariant to brightness/contrast scaling, so fades do not
break it. The template can be blended with the current look (`update`) for
objects that change pose. Per frame it also measures an area-equivalent size
and a contrast against the local background (valid for flat-coloured objects).

Coordinates: pixel-edge convention (pixel i covers [i, i+1)), x and y both in fh
(fractions of frame height), y down. Fixed camera assumed; a warning is added
if the camera moved during the track."""
import time

import cv2
import numpy as np

from . import media, probe
from .store import A, D, M

VERSION = "2"
MAX_BYTES = 1.2e9
LOST_SCORE = 0.5


def load_range(store, t0, t1, max_width=960):
    """Frames in [t0, t1] as BGR uint8 at <= max_width (memory capped). Returns (t, frames, H)."""
    geom, org = probe.geometry(store), probe.origin(store)
    pts = probe.pts(store)
    n = int(np.sum((pts >= t0 - 1e-6) & (pts <= t1 + 1e-6)))
    width = min(geom.w, max_width)
    while n * width * (geom.h * width / geom.w) * 3 > MAX_BYTES and width > 320:
        width = int(width * 0.8)
    if n * width * (geom.h * width / geom.w) * 3 > MAX_BYTES:
        raise RuntimeError(f"{n} frames do not fit in memory; narrow --t0/--t1")
    T, F = [], []
    for t, fr in media.iter_frames(store.video, geom, org, width=width, t0=t0 if t0 > 0 else None, t1=t1 + 1e-6):
        T.append(t)
        F.append(fr.copy())
    if not F:
        raise RuntimeError("no frames in range")
    return np.array(T), F, F[0].shape[0]


def _subpix(res, loc):
    x, y = loc
    dx = dy = 0.0
    h, w = res.shape
    if 0 < x < w - 1:
        l, c, r = res[y, x - 1], res[y, x], res[y, x + 1]
        den = l - 2 * c + r
        if den < -1e-9:
            dx = float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
    if 0 < y < h - 1:
        u, c, d = res[y - 1, x], res[y, x], res[y + 1, x]
        den = u - 2 * c + d
        if den < -1e-9:
            dy = float(np.clip(0.5 * (u - d) / den, -0.5, 0.5))
    return dx, dy


def _match(frame, templ, pred_c, radius):
    """Best template centre near pred_c (pixel-centre coords). Returns (cx, cy, score) or None."""
    th, tw = templ.shape[:2]
    H, W = frame.shape[:2]
    tlx, tly = pred_c[0] - (tw - 1) / 2, pred_c[1] - (th - 1) / 2
    x0 = int(max(0, np.floor(tlx - radius)))
    y0 = int(max(0, np.floor(tly - radius)))
    x1 = int(min(W, np.ceil(tlx + radius) + tw))
    y1 = int(min(H, np.ceil(tly + radius) + th))
    if x1 - x0 < tw or y1 - y0 < th:
        return None
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, mx, _, loc = cv2.minMaxLoc(res)
    dx, dy = _subpix(res, loc)
    return x0 + loc[0] + dx + (tw - 1) / 2, y0 + loc[1] + dy + (th - 1) / 2, float(mx)


def _run_dir(F, T, i0, step, c0, templ, update):
    """Track from index i0 in direction step (+1/-1); returns {index: (cx, cy, score, lost)}."""
    out = {}
    tw, th = templ.shape[1], templ.shape[0]
    big = max(tw, th)
    c_prev, t_prev = np.array(c0, float), T[i0]
    v = np.zeros(2)  # px/s
    lost_run = 0
    i = i0 + step
    while 0 <= i < len(F):
        dt = abs(T[i] - t_prev)
        pred = c_prev + v * dt * step
        radius = max(10.0, 0.6 * big) + 1.5 * np.linalg.norm(v) * dt
        m = _match(F[i], templ, pred, radius)
        if m is None or m[2] < LOST_SCORE:
            m2 = _match(F[i], templ, pred, radius * 3 + big)
            if m2 is not None and (m is None or m2[2] > m[2]):
                m = m2
        if m is None or m[2] < LOST_SCORE:
            lost_run += 1
            sc = 0.0 if m is None else m[2]
            out[i] = (float(pred[0]), float(pred[1]), sc, True)
            c_new = pred
            if lost_run > 8:  # stop pretending
                v = np.zeros(2)
        else:
            lost_run = 0
            c_new = np.array(m[:2])
            if dt > 0:
                v_new = (c_new - c_prev) / dt * step
                v = 0.5 * v + 0.5 * v_new if np.any(v) else v_new
            out[i] = (float(c_new[0]), float(c_new[1]), m[2], False)
            if update > 0 and m[2] > 0.85:
                patch = cv2.getRectSubPix(F[i].astype(np.float32), (tw, th), (float(c_new[0]), float(c_new[1])))
                templ[:] = (1 - update) * templ + update * patch
        c_prev, t_prev = c_new, T[i]
        i += step
    return out


def measure(frame, c, tw, th):
    """Area-equivalent size (px) and contrast of a flat object against its local background."""
    H, W = frame.shape[:2]
    half = 1.3 * max(tw, th)
    x0, x1 = int(max(0, c[0] - half)), int(min(W, c[0] + half + 1))
    y0, y1 = int(max(0, c[1] - half)), int(min(H, c[1] + half + 1))
    crop = frame[y0:y1, x0:x1].astype(np.float32)
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    ring = np.concatenate([crop[:2].reshape(-1, 3), crop[-2:].reshape(-1, 3),
                           crop[:, :2].reshape(-1, 3), crop[:, -2:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    ring_std = float(np.mean(np.std(ring, axis=0)))
    dist = np.mean(np.abs(crop - bg), axis=2)
    cx, cy = c[0] - x0, c[1] - y0
    hw, hh = max(1, int(0.15 * tw)), max(1, int(0.15 * th))
    core = dist[int(max(0, cy - hh)):int(cy + hh + 1), int(max(0, cx - hw)):int(cx + hw + 1)]
    contrast = float(np.median(core)) if core.size else 0.0
    if contrast < 6.0:
        nan = float("nan")
        return {"contrast": contrast, "size": nan, "w": nan, "h": nan, "bg_std": ring_std, "cx": nan, "cy": nan}
    wgt = np.clip(dist / contrast, 0, 1)
    mask = wgt > 0.5
    ys, xs = np.nonzero(mask)
    touches = xs.min() == 0 or ys.min() == 0 or xs.max() == mask.shape[1] - 1 or ys.max() == mask.shape[0] - 1
    # soft area counts only the connected neighbourhood of the core, via the thresholded blob's bbox
    box = wgt[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    nan = float("nan")
    if touches:
        return {"contrast": contrast, "size": nan, "w": nan, "h": nan, "bg_std": ring_std, "cx": nan, "cy": nan}
    gy, gx = np.mgrid[0:box.shape[0], 0:box.shape[1]]
    tot = float(box.sum())
    return {"contrast": contrast, "size": float(np.sqrt(tot)),
            "w": float(xs.max() - xs.min() + 1), "h": float(ys.max() - ys.min() + 1), "bg_std": ring_std,
            # soft-weighted centroid in pixel-centre coords of the full frame
            "cx": float(x0 + xs.min() + (box * gx).sum() / tot), "cy": float(y0 + ys.min() + (box * gy).sum() / tot)}


def run(store, tid, box_fh, t_init, name="", ref_fh=None, t0=None, t1=None, update=0.0, do_measure=True, center="ncc", force=False):
    tid = str(tid)
    dur = store.get("probe")["video"]["duration"]["v"]
    t0 = 0.0 if t0 is None else max(0.0, t0)
    t1 = dur if t1 is None else min(dur, t1)
    params = {"box": [round(b, 5) for b in box_fh], "t_init": round(t_init, 5), "t0": t0, "t1": t1,
              "ref": ref_fh, "update": update, "measure": do_measure, "name": name, "center": center}
    if not force and store.fresh(("tracks", tid), VERSION, params):
        return store.get("tracks", tid), True
    t_start = time.time()
    T, F, H = load_range(store, t0, t1)
    Hh, Ww = F[0].shape[:2]
    x, y, w, h = [b * H for b in box_fh]  # fh -> analysis px (x also in fh = H units)
    c0 = (x + w / 2 - 0.5, y + h / 2 - 0.5)  # pixel-centre coords
    tw, th = max(8, int(round(w))), max(8, int(round(h)))
    i0 = int(np.argmin(np.abs(T - t_init)))
    templ = cv2.getRectSubPix(F[i0].astype(np.float32), (tw, th), (float(c0[0]), float(c0[1])))
    var = float(templ.std())
    warnings = []
    if var < 3.0:
        warnings.append("template is nearly flat (std %.1f); tracking may slide - box a part with edges" % var)
    res = {i0: (c0[0], c0[1], 1.0, False)}
    res.update(_run_dir(F, T, i0, +1, c0, templ.copy(), update))
    res.update(_run_dir(F, T, i0, -1, c0, templ.copy(), update))
    idx = sorted(res)
    cx = np.array([res[i][0] for i in idx]) + 0.5
    cy = np.array([res[i][1] for i in idx]) + 0.5
    sc = np.array([res[i][2] for i in idx])
    lost = np.array([res[i][3] for i in idx], int)
    Tt = T[idx]
    size = np.full(len(idx), np.nan)
    contrast = np.full(len(idx), np.nan)
    bgstd = np.full(len(idx), np.nan)
    bx_ = np.full(len(idx), np.nan)
    by_ = np.full(len(idx), np.nan)
    if center == "centroid":
        do_measure = True
    if do_measure:
        for k, i in enumerate(idx):
            if lost[k]:
                continue
            m = measure(F[i], (cx[k] - 0.5, cy[k] - 0.5), tw, th)
            if m:
                size[k], contrast[k], bgstd[k] = m["size"], m["contrast"], m["bg_std"]
                bx_[k], by_[k] = m["cx"] + 0.5, m["cy"] + 0.5
    ncc_x, ncc_y = cx.copy(), cy.copy()
    both = np.isfinite(bx_) & (lost == 0)
    if both.sum() >= 5:
        gap = float(np.median(np.hypot(bx_[both] - ncc_x[both], by_[both] - ncc_y[both])))
        if center == "ncc" and gap > 1.5 and gap > 0.04 * max(tw, th):
            warnings.append(f"NCC centre and blob centroid differ by {gap:.1f}px: expected for sprites/textured objects; "
                            "for flat shapes that change size try --center centroid")
    if center == "centroid":
        use = np.isfinite(bx_) & (lost == 0)
        cx = np.where(use, bx_, cx)
        cy = np.where(use, by_, cy)
        if (~use & (lost == 0)).any():
            warnings.append(f"centroid unavailable in {int((~use & (lost == 0)).sum())} frames; NCC centre used there")
    # jitter: robust residual against a local quadratic (savgol); only a noise indicator
    jit = None
    ok = lost == 0
    if ok.sum() >= 11:
        from scipy.signal import savgol_filter
        r = np.concatenate([cx[ok] - savgol_filter(cx[ok], 7, 2), cy[ok] - savgol_filter(cy[ok], 7, 2)])
        jit = float(1.4826 * np.median(np.abs(r - np.median(r))) / H)
    mo = store.get("motion")
    if mo:
        for cs in mo["camera_segments"]:
            if cs["t0"] < Tt[-1] and cs["t1"] > Tt[0]:
                warnings.append(f"camera moves {cs['t0']:.2f}-{cs['t1']:.2f}s; positions are screen-space there")
    if lost.sum():
        warnings.append(f"{int(lost.sum())} frames lost (score < {LOST_SCORE}); predicted positions, flagged lost=1")
    ref_src = "measured" if ref_fh else "assumed"
    ref_val = ref_fh if ref_fh else box_fh[3]
    out = {
        "name": name, "t0": float(Tt[0]), "t1": float(Tt[-1]), "t_init": float(T[i0]), "mode": "template-ncc",
        "update": update, "box0_fh": [float(b) for b in box_fh],
        "template_px": [tw, th], "analysis_height_px": H,
        "ref": {"fh": float(ref_val), "src": ref_src, "conf": 1.0 if ref_fh else 0.4,
                "what": "size of the reference object in fh" if ref_fh else "assumed: height of the initial box"},
        "center": center,
        "samples": {"t": Tt, "x": cx / H, "y": cy / H, "x_ncc": ncc_x / H, "y_ncc": ncc_y / H, "score": sc, "lost": lost,
                    "size": size / H, "contrast": contrast, "bg_std": bgstd},
        "quality": {"frames": int(len(idx)), "lost": int(lost.sum()), "mean_score": float(sc[ok].mean()) if ok.any() else 0.0,
                    "jitter_fh": D(jit, 0.5, "fh", "robust residual vs local quadratic; an upper bound on noise") if jit is not None else None},
        "units": "x,y centre in fh (pixel-edge convention, y down); size = area-equivalent side in fh; contrast 0..255",
        "warnings": warnings,
    }
    return store.put(("tracks", tid), out, VERSION, params, deps=[("probe",)], t_start=t_start), False
