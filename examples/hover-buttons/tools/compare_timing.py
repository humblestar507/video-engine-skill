"""Compares hover events of the reference video and of the replica recording.

Original: the fill front (right edge of the coloured fill) is read on a pixel row above the label text, per button;
a hover is a stretch where the front is more than 60 video px beyond the column edge. Replica: frames where the page's
state is 'hover' and the fill is wider than the column. Events are matched by onset (< 0.4 s apart).

python compare_timing.py <reference video> <btnlog.json> [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / ".claude" / "skills" / "video-engine"))
from engine import media, probe  # noqa: E402
from engine.store import Store  # noqa: E402

video, logf = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "timing.json"
COL = {"AUTHORIZE": ((157, 82, 240), (700, 1700), 825), "DOWNLOAD": ((246, 77, 67), (1650, 2600), 1775), "PUBLISH": ((75, 106, 239), (2540, 3400), 2665)}
st = Store(video)
probe.run(st)
geom, org = probe.geometry(st), probe.origin(st)
T, R = [], {k: [] for k in COL}
for t, fr in media.iter_frames(st.video, geom, org):
    T.append(t)
    for k, (c, (a, b), _) in COL.items():
        m = (np.abs(fr[1000:1007, a:b].astype(int) - np.array(c)).sum(2) < 60).any(0)
        xs = np.nonzero(m)[0]
        R[k].append(a + xs.max() if len(xs) else 0)
T = np.array(T)
rep = json.loads(Path(logf).read_text())["log"]


def runs(mask):
    res, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            res.append((i, j))
            i = j + 1
        else:
            i += 1
    return res


errs, summary = [], {}
for bi, n in enumerate(COL):
    orig = [(T[a], T[b]) for a, b in runs(np.array(R[n]) > COL[n][2] + 60) if b - a >= 3]
    ps = np.array([r[bi][0] == "hover" and (r[bi][2] - r[bi][1]) > 45 for r in rep])
    rr = [(T[a], T[b]) for a, b in runs(ps) if b - a >= 3]
    used, miss = set(), 0
    for a, b in orig:
        k = min(range(len(rr)), key=lambda i: abs(rr[i][0] - a)) if rr else None
        if k is not None and abs(rr[k][0] - a) < 0.4 and k not in used:
            used.add(k)
            errs += [abs(rr[k][0] - a), abs(rr[k][1] - b)]
        else:
            miss += 1
    summary[n] = {"original": len(orig), "replica": len(rr), "matched": len(used), "missed": miss, "extra": len(rr) - len(used)}
    print(n, summary[n])
res = {"per_button": summary, "median_ms": float(np.median(errs) * 1000), "p95_ms": float(np.percentile(errs, 95) * 1000),
       "max_ms": float(np.max(errs) * 1000), "n": len(errs)}
print("timing error (matched onsets + offsets): median %.0f ms, p95 %.0f ms, max %.0f ms, n=%d" % (res["median_ms"], res["p95_ms"], res["max_ms"], res["n"]))
Path(out).write_text(json.dumps(res, indent=1))
