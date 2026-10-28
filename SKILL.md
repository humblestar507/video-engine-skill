---
name: video-engine
description: General video analysis engine (2D, fixed camera). Turns a video (gameplay, motion design, UI) into a compact video.json of measurements - PTS timings, cuts, motion map, audio onsets, object tracks, physics (g, bounces, apex), easing (cubic-bezier, spring), sprite pose timelines - and can reproduce the result (JS, CSS, GDScript, mp4) and compare it with the original in numbers. Use it when you need to analyse a video file on disk, work out jump physics, animation timing and easing, pose changes, or reproduce and verify behaviour seen in a clip. Triggers - "analyze this video/clip/gameplay/screen recording", "what easing is this", "jump physics", "gravity from video", "how many frames does the pose hold", "reproduce this from the video", "break down this animation".
---

# video-engine

The engine sits next to this file. All commands: `python "<skill folder>/ve.py" <command> ...` (called `ve` below).
Results live on disk, not in the chat: **do not page through frames one by one**, follow the funnel.

## Principles (do not break)
1. **Measurements, not descriptions.** Every number in `video.json` carries a label: `measured` (read from the video), `derived` (a fit or inference), `assumed` (an assumption) and a confidence `conf`. When reporting an important number, name its label.
2. **Time comes from PTS only.** Seconds from the first frame. Never compute `frame / fps`. That breaks on VFR; the engine flags such videos.
3. **Units are relative.** `fh` = fraction of the frame height (x and y in the same units, y down). Sizes and speeds are also given in units of the reference object (`ref`). Convert to pixels only on request: `px = fh * height`.
4. **Cache.** Asking again about the same video recomputes nothing (the output says `(cached)`). The cache key is the file content, not its name. Folder: `ve path VIDEO`.
5. **Honesty.** If a method is unsure or a criterion is not met, say so plainly and quote the `warnings` from the output. Do not tune thresholds to get a green result.

## Setup
- Needs: Python 3, `ffmpeg` and `ffprobe` on PATH, `numpy scipy opencv-contrib-python-headless librosa soundfile` (`pip install -r requirements.txt` from the skill folder). Check with `ve -h`.
- The chat **does not accept video files directly**. Ask for the path to a file on disk (`C:\...\clip.mp4`). For links you need `yt-dlp` (it may be missing: offer to install it, do not download without asking).
- No model weights and no GPU: classical methods everywhere. There are no neural trackers yet.

## The funnel (saves tokens)
Cost estimate from the formula `width*height/750` (billing was never measured): a summary is ~150-300 tokens of text, a 20-frame contact sheet at 256 px ~900, a strip of 8 frames ~300, a pose palette ~200.

1. **Summary.** `ve analyze VIDEO` (layers 0-1: probe, scenes, motion, audio; then `ve summary VIDEO`). Read: duration, fps/VFR, cuts, activity, camera and object motions found, audio hits, warnings.
2. **Contact sheet.** `ve sheet VIDEO [--scene N | --t0 a --t1 b] [--n 20] [--motion]`. One sheet with timecodes instead of dozens of frames. `--motion` draws motion boxes (green: the area an object swept during the segment; cyan: an on-screen overlay; orange arrow: image motion).
3. **Query the data.** `ve query VIDEO <path> [--t0 --t1 --max-rows]`. Examples: `scenes.events`, `motion.object_segments`, `motion.camera_segments`, `audio.onsets`, `tracks.1.samples`, `curves.ballistic_1.impacts`, `curves.1_x.segments`, `states.s1.timeline`, `signals.dchg`. Without a path it lists the sections. Tables are cut to 40 rows; narrow the window with `--t0/--t1`.
4. **Targeted look.** Only when the numbers do not answer. `ve frame VIDEO --t 3.5 [--crop x,y,w,h --units px|fh|frac]` (a crop of a small object is upscaled without blur), `ve strip VIDEO --t0 .. --t1 .. [--n 8] [--crop ..]` (consecutive frames in one image).

For long videos: summary and sheet over the whole clip, deep analysis (layer 2) only on the scenes you and the user pick. 10 minutes of 720p: layers 0-1 take ~2 minutes once, then come from the cache.

## Layer 2: only on objects the user confirmed
There is no automatic "hero" detection. Show a frame or a sheet and let the user point at the object, or confirm a box yourself from a crop. Box = the object plus 2-3 px margin on each side, **without** background that moves differently.

```
ve track VIDEO --id 1 --name ball --box x,y,w,h --units px --ref-px 24 [--t-init 0] [--t0 --t1] [--update 0.3] [--center centroid]
ve curves VIDEO --track 1 --ballistic            # physics: g, bounces, apex, restitution
ve curves VIDEO --track 1 --channel x|y|size|scale|opacity   # easing / springs
ve states VIDEO --id s1 --box x,y,w,h --track 2 [--thr ..]    # pose timeline
```
- `--ref-px` is the size of the reference in px (character height, ball diameter). Without it the reference is `assumed` (the box height) and every "reference unit" value is marked as an assumption.
- `--update 0.3` for objects that change pose (sprites). `--center centroid` for **flat shapes that change size** (otherwise correlation gives a plateau and the centre drifts; the engine warns). Default for small texture.
- `states` follows the sprite's track; the window includes the background, so a changing background counts as appearance.

What to pick for which question:
| Question | Commands | What to read |
|---|---|---|
| "What gravity / jump / bounce?" | track + `curves --ballistic` | `g` (fh/s2 and ref/s2), `restitution`, `apex`, `g_up/g_down`, `rests` |
| "What easing / spring?" | track (`--center centroid` for scale) + `curves --channel` | `model`, `nearest_preset`, `params`, `params_sd`, `rmse_over_noise` |
| "How many frames does a pose hold, what is the run cycle?" | track `--update 0.3` + `states` | `timeline`, `cycles.hold_frames`, `palette_image` |
| "Where does anything happen in the video?" | analyze + sheet --motion | `scenes`, `motion.*`, `audio.onsets` |

Report speed/acceleration in fh/s and fh/s2 and in reference units; take seconds and frames from `t` (PTS).

## Reproduction and verification (layer 3)
```
ve spec VIDEO --track 1 --out spec.json          # scene parameters; textured objects are NOT rendered (graphics are not copied)
ve export spec.json --target js|css|godot|all --out DIR
ve verify VIDEO --spec spec.json --thr-track-fh 0.002 [--apply --next-spec spec2.json]
ve compare VIDEO TEST_VIDEO [--tracks 1] [--thr-track-fh ..]   # any second video, e.g. a recording of your implementation
```
The loop: spec -> render -> analyse with the same modules -> compare -> `--apply` corrects only measurable parameters -> repeat. Stop on PASS, on `BLOCKED` (the analyser did not recover the parameters it was given, so the measurement on the reproduction cannot be trusted: do not apply), or when the channel structure diverged (model or segment count: edit by hand).
- **The default threshold 0.01 fh is not agreed with the user.** Ask for a threshold (in px at their resolution) before giving a verdict and pass `--thr-track-fh`. The tests used 0.002 fh (0.72 px at 360p): that is the author's suggestion, not a norm.
- The verdict covers track positions, fitted parameters (tolerances in `compare.py`) and the pose sequence. SSIM is informational only: it also sees everything the spec does not model.
- In the report, `params_ranked` lists numeric errors first by descending `z` (z>1 is out of tolerance), structural ones (a missing bounce, a different model) after.
- Exports: JS verified in node, CSS verified in Chrome (spring = a `linear()` approximation, the error is written in the .css header), **GDScript was never run in Godot**. Tell the user so if they take the Godot file.

## How to report
- Conclusion and number first, then label and confidence: "g = 2.50 fh/s2 (derived, +-0.001; reference set by the user)".
- Retell engine warnings (`warnings`, `camera assumed static`, `lost`, `NCC centre differs`, `no stable threshold`); do not swallow them.
- A single clip cannot reveal hidden mechanics (input buffering, coyote time): ask for more clips.
- For other people's games and films the result is a behaviour analysis for the user's own project; do not reproduce assets.

## Known limitations (v1)
- 2D and a **fixed camera**. Camera motion is measured in `motion`, but tracks are in screen coordinates and warn about it. Third-person 3D and neural trackers are stage 5.
- On a flat background the camera is not measured (`assumed static`, low confidence).
- Tracker: no rotation, no occlusion handling (a lost object is flagged `lost=1`, not "followed").
- A fading object can be mistaken by the cut detector for a scene dissolve (no boundary is created).
- Ballistics: the surface is assumed static; `g` has no metres (a physical size for the reference is needed); a short last flight (<6 frames) is skipped.
- A bezier is not always identifiable: look at `nearest_preset.rmse_vs_free_bezier` (near 1 means the name is meaningful).
- Poses: thresholds were tuned on synthetic clips and chosen by plateau; on real video check `palette_image` and set `--thr` when in doubt. Nearly identical poses merge.
- Audio was tested on clean synthetic clicks and thumps; speech and dense music were not tested.
- All quality was measured on synthetic clips with known ground truth. Real recordings may behave worse.

## Self-test
`python tests/test_engine.py` (skill folder; generates clips in `tests/clips`, ~6 minutes, 112 checks). If anything is red after a dependency or ffmpeg update, do not trust the results until you understand why. File format: `references/video-json.md`.
