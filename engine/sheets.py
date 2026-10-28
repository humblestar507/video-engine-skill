"""Contact sheets, film strips and single frames/crops for Claude to look at.

All images land in <cache>/views/ and are cached by their parameters."""
import hashlib
import json

import cv2
import numpy as np

from . import media, probe


def _name(prefix, params):
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return f"views/{prefix}-{h}"


def _label(tile, text, bar=22):
    h, w = tile.shape[:2]
    out = np.full((h + bar, w, 3), 24, np.uint8)
    out[:h] = tile
    cv2.putText(out, text, (5, h + bar - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _grid(tiles, cols, pad=4):
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    W = cols * tw + (cols + 1) * pad
    H = rows * th + (rows + 1) * pad
    img = np.full((H, W, 3), 48, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y, x = pad + r * (th + pad), pad + c * (tw + pad)
        img[y:y + th, x:x + tw] = t
    return img


def scene_of(store, t):
    for s in store.get("scenes")["scenes"]:
        if s["t0"] - 1e-9 <= t < s["t1"] - 1e-9:
            return s["id"]
    return store.get("scenes")["scenes"][-1]["id"]


def grab_many(store, times, width):
    geom, org = probe.geometry(store), probe.origin(store)
    out = []
    for t in times:
        ta, fr = media.grab(store.video, geom, org, t, width=width)
        if fr is not None:
            out.append((ta, fr))
    return out


def pick_times(store, t0=None, t1=None, n=20):
    """Keyframes from the scenes inside [t0, t1], thinned/topped up to n."""
    sc = store.get("scenes")
    t0 = 0.0 if t0 is None else t0
    t1 = store.get("probe")["video"]["duration"]["v"] if t1 is None else t1
    cands = []
    for s in sc["scenes"]:
        if s["t1"] <= t0 or s["t0"] >= t1:
            continue
        kfs = [k for k in s["keyframes"] if t0 <= k <= t1] or [max(t0, s["t0"])]
        cands.append(kfs)
    if not cands:
        return []
    # one per scene first (the first keyframe), then fill by round robin
    chosen = []
    depth = 0
    while len(chosen) < n and any(depth < len(c) for c in cands):
        for c in cands:
            if depth < len(c) and len(chosen) < n:
                chosen.append(c[depth])
        depth += 1
    if len(chosen) < n:
        # top up: quantiles of activity+time over the range, away from frames already chosen
        z = np.load(store.file("signals_l0.npz"))
        t, act = z["t"], z["dchg"]
        m = (t >= t0) & (t <= t1)
        t, act = t[m], act[m]
        if len(t) > len(chosen):
            ca = np.cumsum(act)
            ca = ca / ca[-1] if ca[-1] > 0 else np.zeros_like(ca)
            ct = (t - t[0]) / max(1e-9, t[-1] - t[0])
            c = 0.6 * ca + 0.4 * ct
            gap = (t[-1] - t[0]) / (2.5 * n)
            # coarse-to-fine order (van der Corput) so a partial top-up still covers the range
            vdc = [int(format(k, "b")[::-1], 2) / 2 ** len(format(k, "b")) for k in range(1, 4 * n)]
            for q in vdc:
                if len(chosen) >= n:
                    break
                cand = float(t[min(len(t) - 1, np.searchsorted(c, q))])
                if all(abs(cand - x) > gap for x in chosen):
                    chosen.append(cand)
    return sorted(set(chosen))


def _overlay(store, fr, ta):
    """Draw layer-1 motion on a tile: object-segment bboxes (green), HUD overlays (cyan), camera arrow (orange)."""
    mo = store.get("motion")
    if not mo:
        return fr
    h, w = fr.shape[:2]
    fh = h  # fh units -> tile px (x is also in fh units: x_px = x_fh * h)
    for sg in mo["object_segments"]:
        if sg["t0"] - 1e-6 <= ta <= sg["t1"] + 1e-6:
            x, y, bw, bh = sg["bbox_fh"]
            cv2.rectangle(fr, (int(x * fh), int(y * fh)), (int((x + bw) * fh), int((y + bh) * fh)), (0, 255, 0), 2)
    sid = scene_of(store, ta)
    for ov in mo.get("overlays", []):
        if ov["scene"] == sid:
            x, y, bw, bh = ov["bbox_fh"]
            cv2.rectangle(fr, (int(x * fh), int(y * fh)), (int((x + bw) * fh), int((y + bh) * fh)), (255, 255, 0), 1)
    for cs in mo["camera_segments"]:
        if cs["t0"] - 1e-6 <= ta <= cs["t1"] + 1e-6:
            vx, vy = cs["image_velocity_fh_s"]["v"]
            c = (w // 2, h // 2)
            cv2.arrowedLine(fr, c, (int(c[0] + vx * fh * 0.5), int(c[1] + vy * fh * 0.5)), (0, 140, 255), 3, tipLength=0.3)
    return fr


def contact_sheet(store, t0=None, t1=None, n=20, width=256, cols=5, times=None, overlay=False):
    times = times if times is not None else pick_times(store, t0, t1, n)
    params = {"times": [round(x, 4) for x in times], "w": width, "cols": cols, "ov": bool(overlay)}
    path = store.file(_name("sheet", params) + ".jpg")
    if path.exists():
        return path, times
    frames = grab_many(store, times, width)
    if overlay:
        frames = [(ta, _overlay(store, fr, ta)) for ta, fr in frames]
    tiles = [_label(fr, f"{ta:.3f}s  S{scene_of(store, ta)}") for ta, fr in frames]
    img = _grid(tiles, min(cols, len(tiles)))
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path, [ta for ta, _ in frames]


def _crop_px(store, crop, units):
    v = store.get("probe")["video"]
    W, H, sar = v["width"], v["height"], v.get("sar", 1.0)
    x, y, w, h = crop
    if units == "fh":
        x, w = x * H / sar, w * H / sar
        y, h = y * H, h * H
    elif units == "frac":
        x, w, y, h = x * W, w * W, y * H, h * H
    x0, y0 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
    x1, y1 = int(min(W, np.ceil(x + w))), int(min(H, np.ceil(y + h)))
    return x0, y0, x1, y1


def frame(store, t, crop=None, units="fh", width=None, min_side=384):
    """Full-resolution frame or crop. Small crops are upscaled with nearest
    neighbour (pixel art stays crisp). Returns (path, t_actual)."""
    params = {"t": round(t, 5), "crop": crop, "units": units, "w": width}
    path = store.file(_name("frame", params) + ".png")
    side = path.with_suffix(".t")
    if path.exists() and side.exists():
        return path, float(side.read_text())
    geom, org = probe.geometry(store), probe.origin(store)
    ta, fr = media.grab(store.video, geom, org, t)
    if fr is None:
        raise RuntimeError(f"no frame at t={t}")
    if crop:
        x0, y0, x1, y1 = _crop_px(store, crop, units)
        fr = fr[y0:y1, x0:x1]
        s = max(1, int(np.ceil(min_side / max(1, min(fr.shape[:2])))))
        if s > 1:
            fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    if width and fr.shape[1] > width:
        fr = cv2.resize(fr, (width, int(fr.shape[0] * width / fr.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), fr)
    side.write_text(repr(ta))
    return path, ta


def strip(store, t0, t1, n=8, crop=None, units="fh", width=200, cols=8):
    """Consecutive (or evenly spaced) frames from one decode pass, in one image."""
    params = {"t0": t0, "t1": t1, "n": n, "crop": crop, "units": units, "w": width, "cols": cols}
    path = store.file(_name("strip", params) + ".jpg")
    side = path.with_suffix(".t")
    if path.exists() and side.exists():
        return path, json.loads(side.read_text())
    pts = probe.pts(store)
    sel = pts[(pts >= t0 - 1e-6) & (pts <= t1 + 1e-6)]
    if len(sel) == 0:
        raise RuntimeError("no frames in range")
    want = set(np.round(sel[np.linspace(0, len(sel) - 1, min(n, len(sel))).astype(int)], 6))
    geom, org = probe.geometry(store), probe.origin(store)
    tiles, got = [], []
    box = _crop_px(store, crop, units) if crop else None
    for ta, fr in media.iter_frames(store.video, geom, org, t0=t0, t1=t1 + 1e-6):
        if round(ta, 6) not in want and not any(abs(ta - w) < 1e-4 for w in want):
            continue
        if box:
            x0, y0, x1, y1 = box
            fr = fr[y0:y1, x0:x1]
        interp = cv2.INTER_NEAREST if fr.shape[1] < width else cv2.INTER_AREA
        fr = cv2.resize(fr, (width, max(1, int(fr.shape[0] * width / fr.shape[1]))), interpolation=interp)
        tiles.append(_label(fr, f"{ta:.3f}s"))
        got.append(ta)
    cv2.imwrite(str(path), _grid(tiles, min(cols, len(tiles))), [cv2.IMWRITE_JPEG_QUALITY, 88])
    side.write_text(json.dumps(got))
    return path, got
