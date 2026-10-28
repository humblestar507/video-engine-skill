# video.json: format (schema `video.json/1`)

One file per video: `<cache>/<name>-<content hash>/video.json` (`ve path VIDEO`). Next to it: `pts.npy` (frame PTS), `signals_*.npz` (per-frame arrays), `views/` (sheets, frames, palettes), `audio.wav`.

Every section contains `_meta`: `version`, `params`, `deps` (timestamps of the dependencies), `computed_at`. A section is recomputed only if its version, parameters or dependencies changed.

## Value labels
`{"v": value, "src": "measured|derived|assumed", "conf": 0..1, "unit": ..., "note": ...}`. Not every field is labelled: event lists carry `src` and `conf` in each record.

## Units
- time: seconds from the first frame, by PTS;
- positions and sizes: `fh` (fraction of the frame height), x to the right, y down, origin at the top-left corner, pixel-edge convention (pixel i covers [i, i+1));
- speeds: fh/s, accelerations: fh/s2; in reference units by dividing by `tracks.<id>.ref.fh`.

## Sections
| Section | Module | Contents |
|---|---|---|
| `probe` | probe | `video` (frames, duration, fps, `vfr`, `dt_ms`), `audio`, `timeline`, `units`, `warnings` |
| `scenes` | scenes | `scenes[]` (t0, t1, activity, static, keyframes), `events[]` (cut, flash, fade_in/out, dim/brighten, dissolve, black), `activity_segments` |
| `motion` | motion | `camera_segments[]` (image velocity fh/s, scale, rotation), `object_segments[]` (bbox, path, speed), `overlays[]` (fixed on screen), `per_scene` |
| `audio` | audio | `onsets[]` (t, strength, band, `secondary_of`), `tempo`, `loudness`, `sync` |
| `tracks.<id>` | track | `samples` (t, x, y, score, lost, size, contrast, ...), `ref`, `box0_fh`, `quality`, `warnings` |
| `curves.ballistic_<id>` | curves | `g`, `g_ref_units`, `flights[]`, `impacts[]`, `rests[]`, `restitution_mean`, `vx_mean_fh_s` |
| `curves.<id>_<channel>` | curves | `segments[]`: `model` (linear, CSS preset, bezier, spring), `params`, `from`, `to`, `rmse`, `nearest_preset`, `candidates` |
| `states.<id>` | states | `palette`, `timeline[]`, `cycles[]`, `threshold`, `palette_image` |
| `compare.<label>` | compare | `tracks`, `params_ranked`, `states`, `ssim`, `verdict` |

## Queries
`ve query VIDEO scenes.events`, `tracks.1.samples --t0 1 --t1 2`, `curves.ballistic_1.impacts`, `signals.dchg`. Paths are separated by dots; the key `1_x` contains an underscore, not a dot.

## Extending
New modules write their own section through `Store.put(path, payload, version, params, deps)` and do not change existing ones. The format is not tied to 2D: for 3D and neural trackers, add new sections and channels.
