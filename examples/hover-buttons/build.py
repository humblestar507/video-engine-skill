"""Builds index.html from the measured curves (curves_<button>.json from the video analysis) and template.html."""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).parent
tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "curves"


def smooth(a):
    a = np.array(a, float)
    k = np.convolve(np.pad(a, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    k[0], k[-1] = 0.0 if a[0] == 0 else a[0], a[-1]
    return [round(float(x), 4) for x in np.maximum.accumulate(np.clip(k, 0, 1))]


data = {}
for name in ("AUTHORIZE", "DOWNLOAD", "PUBLISH"):
    cv = json.loads((tmp / f"curves_{name.lower()}.json").read_text())
    data[name] = {k: smooth(v) for k, v in cv.items() if k != "dur"}
    data[name]["dur"] = cv["dur"]
html = (here / "template.html").read_text(encoding="utf-8").replace("__CURVES__", json.dumps(data))
(here / "index.html").write_text(html, encoding="utf-8")
print("written", here / "index.html", len(html), "bytes")
