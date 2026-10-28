"""Layer 1: where and when the picture moves, and how the camera moves.

Per frame pair:
  1. camera: sparse LK features + RANSAC similarity (shift, scale, rotation).
     Trusted only when the inliers cover the frame; on flat/untextured frames
     the camera is *assumed* static (label `assumed`, low confidence) because
     a lone moving object would otherwise be mistaken for camera motion.
  2. residual: previous frame warped by the camera, minus current frame,
     above the video-wide noise floor -> blobs (moving objects), 8x14 heat grid.
Per-frame arrays go to signals_l1.npz; video.json keeps camera segments,
object-motion segments (bbox, path, speed) and scene summaries."""
import time

import cv2
import numpy as np

from . import media, probe
from .store import D, M

VERSION = "3"
WIDTH = 320
GRID = (8, 14)  # rows, cols
MIN_BLOB = 0.0004  # blob area as a fraction of the frame
CAM_MIN_PX = 0.25  # camera shift per frame (analysis px) that counts as movement
COVER = 0.4  # inliers must span this fraction of width and height to trust the camera


def estimate_camera(prev, cur):
    """Returns (M 2x3 or None, conf, n_inliers). M maps prev points to cur."""
    h, w = prev.shape
    p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 6)
    if p0 is None or len(p0) < 15:
        return None, 0.0, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 15:
        return None, 0.0, int(ok.sum())
    a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
    Mx, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=0.8, maxIters=500)
    if Mx is None:
        return None, 0.0, 0
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    if n < 15:
        return None, 0.0, n
    pts = a[inl]
    cover_x = (pts[:, 0].max() - pts[:, 0].min()) / w
    cover_y = (pts[:, 1].max() - pts[:, 1].min()) / h
    if cover_x < COVER or cover_y < COVER:
        return None, 0.0, n
    conf = float(min(1.0, n / len(a) * 1.2) * min(1.0, min(cover_x, cover_y) / 0.8))
    return Mx, conf, n


def cam_params(Mx, w, h):
    """Shift of the image at the frame centre (px), log scale, rotation (rad)."""
    cx, cy = w / 2, h / 2
    mx, my = Mx @ np.array([cx, cy, 1.0])
    return mx - cx, my - cy, float(np.log(np.hypot(Mx[0, 0], Mx[1, 0]))), float(np.arctan2(Mx[1, 0], Mx[0, 0]))


def _blobs(mask, min_px, raw_mask=None):
    """Moving blobs from a binary residual mask. Slow objects only leave thin
    leading/trailing slivers, so no opening: dilate to merge them, then keep
    components that hold at least `min_px` raw changed pixels.
    Tuple: (x0, y0, x1, y1, cx, cy, raw_px, screen_fixed)."""
    m = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    raw_cnt = np.bincount(lab[mask > 0], minlength=n)
    out = []
    for i in range(1, n):
        if raw_cnt[i] < min_px:
            continue
        x, y, bw, bh, _ = stats[i]
        fixed = False
        if raw_mask is not None:
            sel = (lab == i) & (mask > 0)
            # changed after camera compensation but not changed on screen -> fixed to the screen
            fixed = float(np.mean(~raw_mask[sel])) > 0.35
        # bbox from the raw changed pixels, not the dilated component (dilation only merges)
        ys, xs = np.nonzero((lab[y:y + bh, x:x + bw] == i) & (mask[y:y + bh, x:x + bw] > 0))
        out.append((x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1,
                    float(cents[i][0]), float(cents[i][1]), int(raw_cnt[i]), fixed))
    return sorted(out, key=lambda b: -b[6])


def compute(store):
    geom, org = probe.geometry(store), probe.origin(store)
    sc = store.get("scenes")
    noise = max(10.0, float(sc["activity_segments"].get("noise_floor_diff", 10.0)))
    l0 = np.load(store.file("signals_l0.npz"))
    ncc, luma = l0["ncc"], l0["luma"]
    v = store.get("probe")["video"]
    W, H = geom.scaled(WIDTH)
    rows, cols = GRID
    min_px = max(12, int(MIN_BLOB * W * H))

    T, cam, camc, camsrc, resf, kind, heat, blob = [], [], [], [], [], [], [], []
    prev = None
    ones = np.full((H, W), 255, np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    for i, (t, fr) in enumerate(media.iter_frames(store.video, geom, org, width=WIDTH, gray=True)):
        fr = fr.copy()
        T.append(t)
        if prev is None:
            cam.append((0, 0, 0, 0)); camc.append(0.0); camsrc.append(0); resf.append(0.0)
            kind.append(0); heat.append(np.zeros(GRID, np.float32)); blob.append([])
            prev = fr
            continue
        # transitions (cut, flash, strong fade) are not motion
        jump = i < len(ncc) and (ncc[i] < 0.6 or abs(luma[i] - luma[i - 1]) > 0.08)
        if jump:
            cam.append((0, 0, 0, 0)); camc.append(0.0); camsrc.append(0); resf.append(0.0)
            kind.append(2); heat.append(np.zeros(GRID, np.float32)); blob.append([])
            prev = fr
            continue
        Mx, conf, n_in = estimate_camera(prev, fr)
        if Mx is None:
            Mx = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
            cam.append((0, 0, 0, 0)); camc.append(0.3); camsrc.append(0)
        else:
            cam.append(cam_params(Mx, W, H)); camc.append(conf); camsrc.append(1)
        moved = abs(cam[-1][0]) > 0.05 or abs(cam[-1][1]) > 0.05 or abs(cam[-1][2]) > 1e-4
        ref = cv2.warpAffine(prev, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if moved else prev
        gain = float(fr.mean()) / max(1.0, float(ref.mean()))
        if abs(gain - 1) > 0.01:  # fade / exposure change: compensate global brightness
            ref = np.clip(ref.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        d = cv2.absdiff(ref, fr)
        valid = np.full((H, W), 255, np.uint8)
        if moved:
            valid = cv2.erode(cv2.warpAffine(ones, Mx, (W, H), flags=cv2.INTER_NEAREST), kernel)
        mask = ((d > noise) & (valid > 0)).astype(np.uint8)
        frac = float(mask.mean())
        resf.append(frac)
        if frac > 0.6:  # almost everything changed: lighting/transition, not an object
            kind.append(2); heat.append(np.zeros(GRID, np.float32)); blob.append([])
        else:
            raw_mask = (cv2.absdiff(prev, fr) > noise) if moved else None
            bl = _blobs(mask, min_px, raw_mask)
            kind.append(1 if any(not b[7] for b in bl) else 0)
            g = cv2.resize(mask.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
            heat.append(g.astype(np.float32))
            blob.append(bl)
        prev = fr
    return dict(T=np.array(T), cam=np.array(cam, np.float64), camc=np.array(camc), camsrc=np.array(camsrc),
                resf=np.array(resf), kind=np.array(kind), heat=np.array(heat), blob=blob, W=W, H=H, noise=noise)


def _runs(flag, max_gap):
    """Index ranges of True runs, bridging gaps up to max_gap."""
    out, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            gap = 0
            k = i
            while k + 1 < n and (flag[k + 1] or gap < max_gap):
                k += 1
                if flag[k]:
                    j = k
                    gap = 0
                else:
                    gap += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def build(store, r):
    T, H, W = r["T"], r["H"], r["W"]
    cam = r["cam"]
    dts = np.diff(T, prepend=T[0] - (T[1] - T[0] if len(T) > 1 else 1 / 30))
    scenes = store.get("scenes")["scenes"]
    sid = np.zeros(len(T), int)
    for s in scenes:
        sid[(T >= s["t0"] - 1e-9) & (T < s["t1"] - 1e-9)] = s["id"]

    # --- camera segments (measured frames only)
    mag = np.hypot(cam[:, 0], cam[:, 1])
    moving = (r["camsrc"] == 1) & ((mag > CAM_MIN_PX) | (np.abs(cam[:, 2]) > 0.002))
    cam_segs = []
    for a, b in _runs(moving, 1):
        if b - a + 1 < 3:
            continue
        if len(set(sid[a:b + 1])) > 1:  # never across a scene boundary
            for s in sorted(set(sid[a:b + 1])):
                idx = np.where(sid[a:b + 1] == s)[0] + a
                if len(idx) >= 3:
                    cam_segs.append(_cam_seg(T, dts, cam, r, idx[0], idx[-1], H))
            continue
        cam_segs.append(_cam_seg(T, dts, cam, r, a, b, H))

    # --- overlays: changed after camera compensation but static on screen (HUD, logos)
    overlays = []
    for sc_ in scenes:
        fr_idx = [i for i in range(len(T)) if sid[i] == sc_["id"] and r["kind"][i] != 2
                  and any(bb[7] for bb in r["blob"][i])]
        if len(fr_idx) >= 3:
            fb = [bb for i in fr_idx for bb in r["blob"][i] if bb[7]]
            overlays.append({
                "scene": sc_["id"], "frames": len(fr_idx),
                "bbox_fh": [float(min(b[0] for b in fb) / H), float(min(b[1] for b in fb) / H),
                            float((max(b[2] for b in fb) - min(b[0] for b in fb)) / H),
                            float((max(b[3] for b in fb) - min(b[1] for b in fb)) / H)],
                "src": "derived", "conf": 0.5,
                "note": "region fixed on screen while the image moves (HUD/overlay); bbox spans the changed edge pixels"})

    # --- object motion segments
    local = r["kind"] == 1
    segs = []
    for a, b in _runs(local, 3):
        if T[b] - T[max(a - 1, 0)] < 0.1:
            continue
        for s in sorted(set(sid[a:b + 1])):
            idx = np.where((sid[a:b + 1] == s) & local[a:b + 1])[0] + a
            if len(idx) < 2:
                continue
            t_lo = max(T[idx[0] - 1], next(x["t0"] for x in scenes if x["id"] == s))
            if T[idx[-1]] - t_lo < 0.1:
                continue
            segs.append(_obj_seg(store, T, r, idx[0], idx[-1], H, W, int(s)))
    for k, sgm in enumerate(segs):
        sgm["id"] = k

    # --- per-scene summary
    per_scene = []
    for s in scenes:
        m = (sid == s["id"])
        measured = m & (r["camsrc"] == 1)
        c_here = [c for c in cam_segs if c["scene"] == s["id"]]
        o_here = [o for o in segs if o["scene"] == s["id"]]
        per_scene.append({
            "scene": s["id"],
            "camera": ("moving" if c_here else "static" if measured.sum() > 0.3 * max(1, m.sum()) else "unknown"),
            "camera_src": "measured" if measured.sum() > 0.3 * max(1, m.sum()) else "assumed",
            "object_segments": [o["id"] for o in o_here],
            "moving_frac_time": float(np.sum(dts[m & local]) / max(1e-9, np.sum(dts[m]))),
        })
    return cam_segs, segs, per_scene, overlays


def _cam_seg(T, dts, cam, r, a, b, H):
    sl = slice(a, b + 1)
    t0 = float(T[max(a - 1, 0)])
    dur = float(T[b] - t0)
    dx, dy = cam[sl, 0].sum() / H, cam[sl, 1].sum() / H
    sid = None
    return {
        "t0": t0, "t1": float(T[b]), "dur": dur,
        "image_shift_fh": [float(dx), float(dy)],
        "image_velocity_fh_s": {"v": [float(dx / dur), float(dy / dur)], "src": "measured",
                                "conf": round(float(r["camc"][sl].mean()), 2), "unit": "fh/s",
                                "note": "motion of the image content (opposite to the camera)"},
        "scale_rate_per_s": float(cam[sl, 2].sum() / dur),
        "rot_rate_rad_s": float(cam[sl, 3].sum() / dur),
        "scene": int(_scene_at(r, T[b])),
        "src": "measured",
    }


def _scene_at(r, t):
    return r["_scene_of"](t)


def _obj_seg(store, T, r, a, b, H, W, scene):
    sl = [i for i in range(a, b + 1) if r["kind"][i] == 1]
    for i in sl:  # fixed overlays are reported separately
        r["blob"][i] = [bb for bb in r["blob"][i] if not bb[7]]
    allb = [bb for i in sl for bb in r["blob"][i]]
    x0 = min(bb[0] for bb in allb) / H
    y0 = min(bb[1] for bb in allb) / H
    x1 = max(bb[2] for bb in allb) / H
    y1 = max(bb[3] for bb in allb) / H
    cs = np.array([(T[i], r["blob"][i][0][4] / H, r["blob"][i][0][5] / H) for i in sl if r["blob"][i]])
    t0 = float(max(T[max(a - 1, 0)], next(x["t0"] for x in store.get("scenes")["scenes"] if x["id"] == scene)))
    heat = r["heat"][a:b + 1].mean(axis=0)
    hs = cv2.resize(heat, (8, 6), interpolation=cv2.INTER_AREA)
    hs = hs / hs.max() if hs.max() > 0 else hs
    seg = {
        "t0": t0, "t1": float(T[b]), "scene": scene, "frames": int(b - a + 1),
        "bbox_fh": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
        "peak_frac": float(r["resf"][a:b + 1].max()),
        "blobs_max": int(max(len(r["blob"][i]) for i in sl)),
        "camera_moving_during": bool(np.mean(r["camsrc"][sl] * (np.hypot(r["cam"][sl, 0], r["cam"][sl, 1]) > CAM_MIN_PX)) > 0.5),
        "heat_6x8": np.round(hs, 2).tolist(),
        "src": "measured",
    }
    if len(cs) >= 3:
        vx = np.polyfit(cs[:, 0], cs[:, 1], 1)[0]
        vy = np.polyfit(cs[:, 0], cs[:, 2], 1)[0]
        seg["main_blob"] = {
            "start_fh": [float(cs[0, 1]), float(cs[0, 2])], "end_fh": [float(cs[-1, 1]), float(cs[-1, 2])],
            "mean_velocity_fh_s": D([float(vx), float(vy)], 0.6, "fh/s",
                                    "slope of blob centroid (frame-difference blobs lead/trail the object, so the centroid lags)"),
        }
    return seg


def run(store, force=False):
    params = {"w": WIDTH, "grid": list(GRID), "min_blob": MIN_BLOB, "cam_min_px": CAM_MIN_PX}
    if not force and store.fresh("motion", VERSION, params):
        return store.get("motion"), True
    t_start = time.time()
    r = compute(store)
    sc = store.get("scenes")["scenes"]
    r["_scene_of"] = lambda t: next((s["id"] for s in sc if s["t0"] - 1e-9 <= t < s["t1"] - 1e-9), sc[-1]["id"])
    cam_segs, segs, per_scene, overlays = build(store, r)
    cam = r["cam"]
    np.savez_compressed(store.file("signals_l1.npz"), t=r["T"], cam_dx_px=cam[:, 0], cam_dy_px=cam[:, 1],
                        cam_logscale=cam[:, 2], cam_rot=cam[:, 3], cam_conf=r["camc"], cam_measured=r["camsrc"],
                        res_frac=r["resf"], kind=r["kind"], heat=r["heat"])
    parts = []
    if cam_segs:
        parts.append(f"{len(cam_segs)} camera moves")
    parts.append(f"{len(segs)} object-motion segments")
    out = {
        "analysis": {"width": r["W"], "height": r["H"], "noise_floor_diff": r["noise"], "grid": list(GRID),
                     "units": "fh = fraction of frame height; x right, y down"},
        "camera_segments": cam_segs, "object_segments": segs, "overlays": overlays, "per_scene": per_scene,
        "summary_line": ", ".join(parts) + (" | camera static elsewhere" if cam_segs else ""),
        "signals_file": "signals_l1.npz", "warnings": [],
    }
    unk = [p["scene"] for p in per_scene if p["camera_src"] == "assumed"]
    if unk:
        out["warnings"].append(f"camera assumed static (no texture to measure) in scenes {unk}")
    return store.put("motion", out, VERSION, params, deps=[("probe",), ("scenes",)], t_start=t_start), False
