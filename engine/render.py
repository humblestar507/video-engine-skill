"""Spec -> video. Flat shapes on a flat background, supersampled, one frame per
requested timestamp (CFR or the PTS of the source clip, so VFR sources are matched)."""
import subprocess

import cv2
import numpy as np

from .model import eval_object

SS = 4


def frame_at(spec, t, ev=None):
    W, H = spec["stage"]["w"], spec["stage"]["h"]
    img = np.empty((H, W, 3), np.float32)
    img[:] = np.array(spec["bg"], np.float32)
    g = spec.get("ground")
    if g:
        y = int(round(g["top_fh"] * H))
        img[y:] = np.array(g["color"], np.float32)
    for k, obj in enumerate(spec["objects"]):
        if not obj["renderable"]:
            continue
        s = ev[k] if ev else eval_object(obj, np.array([t]))
        cx, cy = float(s["x"][0]) * H, float(s["y"][0]) * H
        sc, al = float(s["scale"][0]), float(np.clip(s["opacity"][0], 0, 1))
        w, h = obj["size_fh"][0] * H * sc, obj["size_fh"][1] * H * sc
        m = np.zeros((H * SS, W * SS), np.uint8)
        if obj["shape"] == "disc":
            cv2.ellipse(m, (int(round(cx * SS)), int(round(cy * SS))), (int(round(w * SS / 2)), int(round(h * SS / 2))),
                        0, 0, 360, 255, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(m, (int(round((cx - w / 2) * SS)), int(round((cy - h / 2) * SS))),
                          (int(round((cx + w / 2) * SS)) - 1, int(round((cy + h / 2) * SS)) - 1), 255, -1)
        a = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0 * al
        img = img * (1 - a) + np.array(obj["color"], np.float32) * a
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def render(spec, out_path, times=None, fps=None, crf=14):
    """times: absolute timestamps (s from first frame); default CFR at spec fps over spec duration."""
    fps = fps or spec["fps"]
    if times is None:
        times = np.arange(int(round(spec["duration"] * fps))) / fps
    times = np.asarray(times, float)
    ev = [eval_object(o, times) for o in spec["objects"]]
    W, H = spec["stage"]["w"], spec["stage"]["h"]
    if len(times) > 1 and not np.allclose(np.diff(times), 1 / fps, atol=1e-4):
        return _render_vfr(spec, ev, times, out_path, crf)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "pipe:0", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(crf), "-pix_fmt", "yuv420p", str(out_path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(len(times)):
        p.stdin.write(frame_at(spec, times[i], [{k: v[i:i + 1] for k, v in e.items()} for e in ev]).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return out_path


def _render_vfr(spec, ev, times, out_path, crf):
    import shutil
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    try:
        lines = []
        for i in range(len(times)):
            fn = d / f"{i:06d}.png"
            cv2.imwrite(str(fn), frame_at(spec, times[i], [{k: v[i:i + 1] for k, v in e.items()} for e in ev]))
            dur = (times[i + 1] - times[i]) if i + 1 < len(times) else (times[-1] - times[-2] if len(times) > 1 else 0.04)
            lines += [f"file '{fn.as_posix()}'", "option framerate 1000", f"duration {dur:.6f}"]
        (d / "list.txt").write_text("\n".join(lines))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(d / "list.txt"),
                        "-fps_mode", "passthrough", "-enc_time_base", "1/1000", "-video_track_timescale", "1000",
                        "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(out_path)], check=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out_path
