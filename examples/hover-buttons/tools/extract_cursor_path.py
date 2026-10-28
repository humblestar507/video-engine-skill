"""Measures the mouse-cursor tip per frame in the reference video (design px = video px / 2).

The cursor is the largest dark blob (the button text is coloured, never dark); its tip is the up-left extreme of that blob.
Needs the engine installed as a skill (~/.claude/skills/video-engine) for frame decoding with PTS timing.

python extract_cursor_path.py <video.mp4> <out.json>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.home() / ".claude" / "skills" / "video-engine"))
from engine import media, probe  # noqa: E402
from engine.store import Store  # noqa: E402

video, out = sys.argv[1], sys.argv[2]
st = Store(video)
probe.run(st)
geom, org = probe.geometry(st), probe.origin(st)
T, P = [], []
for t, fr in media.iter_frames(st.video, geom, org):
    gray = cv2.cvtColor(fr[900:1400, 500:3500], cv2.COLOR_BGR2GRAY)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((gray < 70).astype(np.uint8), 8)
    best = None
    for i in range(1, n):
        if stats[i, 4] > 400 and (best is None or stats[i, 4] > stats[best, 4]):
            best = i
    T.append(round(float(t), 4))
    if best is None:
        P.append(None)
        continue
    ys, xs = np.nonzero(lab == best)
    k = np.argmin(xs + ys)
    P.append([round((500 + xs[k]) / 2, 1), round((900 + ys[k]) / 2, 1)])
Path(out).write_text(json.dumps({"t": T, "p": P}))
print("frames", len(T), "with cursor", sum(p is not None for p in P))
