# video-engine

Claude Code skill: turns a video (gameplay, motion design, UI) into a compact `video.json` of measurements
(PTS timings, cuts, motion map, audio onsets, tracks, ballistic physics, easing/spring fits, pose timelines),
then reproduces the behaviour (JS / CSS / GDScript / mp4) and compares it with the original in numbers.

2D and fixed camera in v1. Classical methods only (no model weights).

## Examples: reference vs replica

Three reproduction jobs done with the engine. Each shows the **reference** next to the **replica** (a self-contained HTML page).
Full-quality videos, references, pages and measured data are in [`examples/`](examples/README.md).

### Hover buttons
Top: reference screen recording. Bottom: the HTML replica, driven by the cursor path measured from the reference.
13 of 13 hover events reproduced, onset/offset error median 58 ms (max 167 ms).

![hover buttons: reference vs replica](docs/hover-buttons-compare.gif)

### 2D pixel console
Left: reference GIF. Right: the HTML replica (only the 4% of pixels that change are stored per frame). **0 differing pixels over all 42 frames.**

![2D console: reference vs replica](docs/gba-2d-compare.gif)

Clickable console buttons (scripted presses: A, D-pad arms, Start, Select, B, Power, Ball):

![2D console button demo](docs/gba-2d-buttons-demo.gif)

### 3D pixel console
Left: the 2D reference artwork. Right: a stylised three.js version (low-res render + dither) with orbit camera and pressable buttons. Visual comparison only.

![3D console: reference vs 3D replica](docs/pixel-3d-compare.gif)

GIFs are previews; the `.mp4` files in `examples/` are higher quality. The reference materials in `examples/` (a screen recording and a pixel-art GIF) are third-party works included only for side-by-side comparison; their rights belong to their authors. If you are a rights holder and want something removed, open an issue.

## Install as a skill

```bash
git clone <this repo> ~/.claude/skills/video-engine
pip install -r ~/.claude/skills/video-engine/requirements.txt   # needs ffmpeg + ffprobe in PATH
python ~/.claude/skills/video-engine/ve.py -h
```

`SKILL.md` tells Claude how to use it (summary -> contact sheet -> query -> single frames). Format reference: `references/video-json.md`.

## Self-test

```bash
python tests/test_engine.py     # generates synthetic clips with known ground truth, ~6 min, 112 checks
```

All quality numbers were measured on synthetic clips; see `REPORT.md` for what works, what does not, and what is untested
(GDScript export was not run in Godot; real-world footage is the main open risk).
