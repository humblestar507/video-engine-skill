# Recording the replicas

The pages are deterministic when their clock is faked, so the videos are rendered frame by frame rather than screen-captured.

1. Serve a folder that contains the example pages and accepts frames: `python rec_server.py 8770` (serves the parent of this repo folder; frames go to `frames/<name>/<index>.png`).
   Edit `ROOT` in the script if your layout differs.
2. Open the page through that server, bring the tab to the foreground, and run the matching recorder from the browser console / automation:
   - `hover_buttons_recorder.js` on `hover-buttons/index.html` (needs `cursor_path_tip.json` served at `/rec-tmp/cursor_path_tip.json`)
   - `gba2d_recorder.js` on `gba-2d/index.html`
   - `pixel3d_recorder.js` on `pixel-3d/index.html` with a 800x600 viewport
3. Encode: `ffmpeg -framerate 60 -i frames/btn3/%05d.png -c:v libx264 -crf 17 -pix_fmt yuv420p out.mp4`
   (use 10 fps for `gba_loop`, 20 fps for `gba_demo`, 24 fps for `c3d`; make the height even).

Each recorder replaces `requestAnimationFrame` with a no-op, steps the page's own update code with a fake clock and uploads one PNG per step.
Timing comparison for the buttons: `python ../hover-buttons/tools/compare_timing.py <reference video> btnlog3.json` (needs the engine installed as a skill).
