"""Decodes the reference GIF to frames.npy (BGR, N x H x W x 3) using the engine, for build_assets.py.

python extract_frames.py ref/reference.gif frames.npy && python build_assets.py frames.npy assets.json && python build.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / ".claude" / "skills" / "video-engine"))
from engine import media, probe  # noqa: E402
from engine.store import Store  # noqa: E402

st = Store(sys.argv[1])
probe.run(st)
geom, org = probe.geometry(st), probe.origin(st)
frames = [fr.copy() for _, fr in media.iter_frames(st.video, geom, org)]
np.save(sys.argv[2], np.array(frames))
print("frames", len(frames), frames[0].shape)
