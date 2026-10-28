# video-engine: final report

Version 1: 2D, fixed camera, classical methods, no model weights. Install it by cloning this repository into `~/.claude/skills/video-engine` (see `README.md`).

## What is there
| Stage | Modules | Criterion | Result |
|---|---|---|---|
| 1 Core | probe, scenes, store, sheets | any video becomes a summary and a contact sheet with no manual steps | met on synthetic clips: 8 formats/containers, VFR, rotation, start offset; 10 min of 720p in ~50 s |
| 2 Motion and audio | motion, audio | the motion map agrees with the contact sheet | met: the `sheet --motion` overlay matched the picture; pan -0.3337 vs -0.3333 fh/s; clicks within 4 ms |
| 3 Tracks, poses, curves | track, curves, states | trajectory and pose timeline match manual annotation | met: ball position 0.10 px, g 0.02%, restitution 0.700, spring omega0/zeta within 0.1%, 18 intervals and 7 poses exactly |
| 4 Reproduction | spec, render, export, compare, verify | the verification loop converges to an agreed threshold | converges (ball: 85.9 -> 18.6 -> 0.30 px in 2 corrections); **the threshold was not agreed with a user** |

Self-test: `python tests/test_engine.py`, 112 checks (16 + 24 + 51 + 21), all pass.

## Answers to the design document's open questions
- The chat does not accept video files directly (the file reader refused to open `.mp4`): pass a path to a file on disk.
- First test clips: a ball with gravity and bounces, a sprite with pose changes, easing/spring/scale/fade, plus noisy, VFR, static and a set of formats. No real clips were used.
- Third-person 3D: not in version 1.
- Exporters: JS, CSS, GDScript. Lottie, After Effects, Unity: not done.
- Token budget: an estimate only, from `width*height/750` (summary ~150-300, 20-frame sheet ~900, 8-frame strip ~300). Billing was not measured; real numbers should be added after the first real clips.

## What works reliably (on synthetic clips)
PTS timings (including VFR, 0 ms error), cuts/flashes/fades, motion map and camera measurement on a textured background, audio hits and tempo, sub-pixel tracking, ballistics (g, bounces, apex, rest), easing/spring fits, pose timeline and cycles, the cache, a verification loop that blocks itself when a measurement is not trustworthy.

## What does not work or is untested
- **All quality was measured on synthetic clips written by the same author as the analysers.** Real recordings (motion blur, rotation, occlusion, heavy compression, UI screen recordings) may behave worse. This is the main risk.
- Tracker: no rotation or occlusion handling; a lost object is flagged `lost` but not smartly re-acquired. Flat shapes that change size need `--center centroid`.
- On a flat background the camera is not measured (`assumed static`).
- Poses: thresholds are chosen by plateau and were tuned on synthetic data; the window includes the background; nearly identical poses merge.
- Ballistics: the surface is assumed static, there are no metres (a physical size for the reference is needed), a short last flight is skipped.
- Rendering: flat shapes on a flat background; textured objects (sprites) are deliberately not reproduced. There is no exporter for poses (state machine or spritesheet).
- The GDScript export was never run in Godot (not available here). CSS was verified in Chrome; the spring through `linear()` has a 0.55 px error in the test.
- Audio: speech and dense music were not tested; `compare` does not compare audio.
- A fading object may be taken by the cut detector for a scene dissolve (no boundary is created).
- The verdict threshold default (0.01 fh) is not agreed with a user. The tests used 0.002 fh (0.72 px at 360p) as the author's suggestion.

## Defects found by the tests (all fixed)
Fades taken for cuts; thin edges of slow objects lost; NCC giving a plateau when a flat shape grows; false motion on static footage from a degenerate noise estimate; false rest and false bounce on VFR; rest corrupting the g estimate; a fixed pose threshold falling inside compression noise; a missing final keyframe in the CSS (only a real Chrome caught it); two tests with wrong ground truth (identical poses run1/run3, ball bottom on the ground).

## What next
1. Run on 2-3 real clips (gameplay with a jump, a motion animation, a UI transition), record real token budgets and fix the thresholds that do not survive.
2. Agree the verify threshold with the user.
3. Stage 5: camera estimation for 3D, neural trackers as an optional improvement, Lottie/After Effects/Unity exporters, pose-timeline export to a state machine.
4. Object rotation and re-acquisition after loss in `track`; audio in `compare`.
