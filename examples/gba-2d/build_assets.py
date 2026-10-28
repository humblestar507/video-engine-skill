"""Builds assets.json for the HTML player from the decoded GIF frames (C:/tmp/gif_frames.npy, BGR, 42 frames).

* base: frame 0 as a lossless palette PNG
* comps: every region that ever changes; per region a sprite sheet holding, for each frame, only the pixels that differ
  from the base (alpha elsewhere), so the page replays the exact animation
* buttons: masks and 'pressed' patches (the button shifted one art pixel = 6 px down and darkened) found by pixel analysis
"""
import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

F = np.load(sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/gif_frames.npy")
N, H, W, _ = F.shape
base = F[0]
STEP = 6          # one art pixel in screen pixels
SHIFT = STEP      # press depth


def png_b64(arr_rgba_or_rgb, palette=False):
    im = Image.fromarray(arr_rgba_or_rgb)
    if palette:
        im = im.convert("P", palette=Image.ADAPTIVE, colors=256)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def bgr2rgb(a):
    return a[..., ::-1].copy()


# ---------------------------------------------------------------- animated regions
changed = (F != base).any(3).any(0)
m = cv2.dilate(changed.astype(np.uint8), np.ones((9, 9), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
comps = []
COLS = 7
for i in range(1, n):
    x, y, w, h, a = st[i]
    if not changed[y:y + h, x:x + w].any():
        continue
    rows = (N + COLS - 1) // COLS
    sheet = np.zeros((rows * h, COLS * w, 4), np.uint8)
    for k in range(N):
        sub = F[k, y:y + h, x:x + w]
        diff = (sub != base[y:y + h, x:x + w]).any(2)
        r, c = divmod(k, COLS)
        tile = sheet[r * h:(r + 1) * h, c * w:(c + 1) * w]
        tile[..., :3] = bgr2rgb(sub)
        tile[..., 3] = np.where(diff, 255, 0)
    comps.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "cols": COLS, "sheet": png_b64(sheet)})

# ---------------------------------------------------------------- buttons
hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
grey = ((hsv[..., 1] < 60) & (hsv[..., 2] > 90) & (hsv[..., 2] < 235)).astype(np.uint8)
grey[:560] = 0
grey[850:] = 0
grey[:, :380] = 0
grey[:, 870:] = 0
g2 = cv2.dilate(grey, np.ones((5, 5), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(g2, 8)
found = []
for i in range(1, n):
    if st[i, 4] > 120:
        found.append((i, st[i], cen[i]))
names = {}
for i, s, c in found:
    cx, cy = c
    if 440 < cx < 545 and cy > 740:
        names[i] = "dpad"
    elif 670 < cx < 735 and cy < 680:
        names[i] = "a"
    elif 605 < cx < 670 and 680 < cy < 720:
        names[i] = "b"
    elif cx > 760 and 715 < cy < 770:
        names[i] = "pokeball"
    elif cx < 520 and cy < 680:
        names[i] = "power"
    elif cx > 740 and cy > 780:
        names[i] = "select"
    elif 680 < cx < 730 and cy > 800:
        names[i] = "start"


def body_colour(x, y):
    return base[max(0, y), x]


def make_patch(mask, bbox, darken_all=0.88, arm=None):
    """Return RGBA crop of bbox: the masked sprite shifted down by SHIFT and darkened; vacated pixels get body colour."""
    x0, y0, w, h = bbox
    crop = base[y0:y0 + h, x0:x0 + w].copy()
    mk = mask[y0:y0 + h, x0:x0 + w].astype(bool)
    out = crop.copy()
    ys, xs = np.nonzero(mk)
    # vacate: colour from just above the mask in that column
    for x in np.unique(xs):
        col = np.nonzero(mk[:, x])[0]
        top = col.min()
        src = crop[max(0, top - STEP - 1), x]
        out[col.min():col.min() + min(SHIFT, len(col)), x] = src
    for y, x in zip(ys, xs):
        ny = y + SHIFT
        if ny >= h:
            continue
        p = crop[y, x].astype(float)
        f = darken_all
        if arm is not None and arm[y0 + y, x0 + x]:
            f = 0.62
        out[ny, x] = np.clip(p * f, 0, 255).astype(np.uint8)
    rgba = np.dstack([bgr2rgb(out), np.full((h, w), 255, np.uint8)])
    return rgba


buttons = []
for i, s, c in found:
    name = names.get(i)
    if not name:
        continue
    mask = (lab == i).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8))          # include the dark outline
    mask &= (np.abs(np.arange(W)[None, :] - c[0]) < 90).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    pad = SHIFT + 6
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(W, xs.max() + pad + 1), min(H, ys.max() + pad + SHIFT + 1)
    bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    mk_rgba = np.zeros((bbox[3], bbox[2], 4), np.uint8)
    mk_rgba[..., 3] = (mask[y0:y1, x0:x1] * 255)
    entry = {"name": name, "bbox": bbox, "centre": [float(c[0]), float(c[1])], "mask": png_b64(mk_rgba)}
    if name == "dpad":
        cx, cy = c
        yy, xx = np.mgrid[0:H, 0:W]
        arms = {"nw": (xx < cx) & (yy < cy), "ne": (xx >= cx) & (yy < cy), "sw": (xx < cx) & (yy >= cy), "se": (xx >= cx) & (yy >= cy)}
        entry["patches"] = {k: png_b64(make_patch(mask, bbox, 0.92, arm=(a & (grey > 0)).astype(bool))) for k, a in arms.items()}
    else:
        entry["patches"] = {"down": png_b64(make_patch(mask, bbox, 0.86))}
    buttons.append(entry)

out = {"w": int(W), "h": int(H), "frames": int(N), "fps": 10, "step": STEP,
       "base": png_b64(bgr2rgb(base), palette=True), "comps": comps, "buttons": buttons}
Path(sys.argv[2] if len(sys.argv) > 2 else "assets.json").write_text(json.dumps(out), encoding="utf-8")
print("comps", [(c["x"], c["y"], c["w"], c["h"]) for c in comps])
print("buttons", [(b["name"], b["bbox"]) for b in buttons])
print("bytes", len(json.dumps(out)))
