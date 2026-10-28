"""Synthetic clips with known ground truth (written next to each clip as *.truth.json).

python tests/synth.py OUTDIR
"""
import json
import math
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

SS = 4  # supersampling for sub-pixel accurate rendering


def write_video(path, frames, fps, audio=None, sr=44100, crf=18, extra=()):
    path = Path(path)
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0"]
    tmp_wav = None
    if audio is not None:
        tmp_wav = path.with_suffix(".tmp.wav")
        write_wav(tmp_wav, audio, sr)
        cmd += ["-i", str(tmp_wav), "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", *extra, str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    if tmp_wav:
        tmp_wav.unlink()


def write_vfr(path, frames, times, crf=18):
    """Variable frame rate: each frame shown from times[i] to times[i+1]."""
    d = Path(tempfile.mkdtemp())
    try:
        lines = []
        for i, f in enumerate(frames):
            fn = d / f"{i:05d}.png"
            cv2.imwrite(str(fn), f)
            dur = (times[i + 1] - times[i]) if i + 1 < len(times) else (times[-1] - times[-2])
            lines += [f"file '{fn.as_posix()}'", "option framerate 1000", f"duration {dur:.6f}"]
        (d / "list.txt").write_text("\n".join(lines))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(d / "list.txt"), "-fps_mode", "passthrough", "-enc_time_base", "1/1000", "-video_track_timescale", "1000",
                        "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)], check=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def write_wav(path, x, sr):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((x * 32767).astype(np.int16).tobytes())


def canvas(w, h, color):
    img = np.empty((h * SS, w * SS, 3), np.uint8)
    img[:] = color
    return img


def down(img):
    h, w = img.shape[0] // SS, img.shape[1] // SS
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def disc(img, cx, cy, r, color):
    cv2.circle(img, (int(round(cx * SS)), int(round(cy * SS))), int(round(r * SS)), color, -1, cv2.LINE_AA)


def rect(img, x0, y0, x1, y1, color):
    cv2.rectangle(img, (int(round(x0 * SS)), int(round(y0 * SS))),
                  (int(round(x1 * SS)) - 1, int(round(y1 * SS)) - 1), color, -1)


# ---------------------------------------------------------------- scenes + audio

def make_scenes(out):
    W, H, fps, N = 640, 360, 30, 180
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (H, W * 4, 3), dtype=np.uint8), (0, 0), 2.0)
    frames = []
    for i in range(N):
        if i < 45:  # S0: red square moving right, one white flash frame
            f = np.full((H, W, 3), (90, 40, 20), np.uint8)
            x = 40 + 3 * i
            cv2.rectangle(f, (x, 150), (x + 60, 210), (40, 40, 220), -1)
            if i == 25:
                f[:] = 250
        elif i < 90:  # S1: checker + bouncing white dot
            f = np.full((H, W, 3), (60, 140, 60), np.uint8)
            for yy in range(0, H, 40):
                for xx in range(0, W, 40):
                    if (xx + yy) // 40 % 2 == 0:
                        f[yy:yy + 40, xx:xx + 40] = (70, 160, 70)
            k = i - 45
            cv2.circle(f, (320, int(200 - 120 * abs(math.sin(k / 45 * 2 * math.pi)))), 14, (255, 255, 255), -1, cv2.LINE_AA)
        elif i < 138:  # S2: camera pans right 4 px/frame over a texture, HUD box fixed
            k = i - 90
            f = tex[:, 4 * k:4 * k + W].copy()
            cv2.rectangle(f, (20, 20), (120, 60), (0, 220, 255), -1)
        else:  # S3: growing purple disc, fade to black from frame 165
            k = i - 138
            f = np.full((H, W, 3), (128, 128, 128), np.uint8)
            cv2.circle(f, (320, 180), 30 + 3 * k, (160, 40, 160), -1, cv2.LINE_AA)
            if i >= 165:
                a = max(0.0, 1 - (i - 165) / 12)
                f = (f * a).astype(np.uint8)
        frames.append(f)
    sr = 44100
    dur = N / fps
    t = np.arange(int(sr * dur)) / sr
    audio = np.zeros_like(t)
    clicks = [0.25 + 0.5 * k for k in range(12)]
    for c in clicks:
        m = (t >= c) & (t < c + 0.03)
        audio[m] += 0.5 * np.sin(2 * np.pi * 1000 * (t[m] - c)) * np.exp(-(t[m] - c) * 60)
    thumps = [2.0, 4.0]
    for c in thumps:
        m = (t >= c) & (t < c + 0.12)
        audio[m] += 0.9 * np.sin(2 * np.pi * 80 * (t[m] - c)) * np.exp(-(t[m] - c) * 25)
    write_video(out / "scenes.mp4", frames, fps, audio=audio, sr=sr)
    truth = {
        "fps": fps, "frames": N, "duration": N / fps,
        "cuts": [45 / fps, 90 / fps, 138 / fps], "flash": [25 / fps],
        "fade_out": {"t0": 165 / fps, "black_from": 177 / fps},
        "camera_pan": {"t0": 90 / fps, "t1": 137 / fps, "content_dx_px_per_frame": -4, "content_dx_fh_per_s": -4 * fps / H},
        "motion_regions": {"S0": "red square moves right, y 150-210 px", "S1": "dot at x=320 bounces",
                           "S2": "whole frame (pan), HUD box static", "S3": "disc grows from center"},
        "audio": {"clicks": clicks, "thumps": thumps, "tempo_bpm": 120},
    }
    (out / "scenes.truth.json").write_text(json.dumps(truth, indent=1))
    return frames, fps, audio, sr


# ---------------------------------------------------------------- bouncing ball

def ball_state(t, p):
    """Exact analytic bounce: returns (x, y) center in px."""
    g, e, r, ground = p["g_px"], p["restitution"], p["r_px"], p["ground_y_px"]
    yc_ground = ground - r
    x = p["x0"] + p["vx"] * t
    y0, vy0, ts = p["y0"], 0.0, 0.0
    bounces = []
    while True:
        # solve y0 + vy0 s + g s^2/2 = yc_ground
        a, b, c = 0.5 * g, vy0, y0 - yc_ground
        s = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        if ts + s >= t or len(bounces) > 50:
            s = t - ts
            return x, y0 + vy0 * s + 0.5 * g * s * s, bounces
        v_imp = vy0 + g * s
        ts += s
        bounces.append(ts)
        y0, vy0 = yc_ground, -e * v_imp


def ball_params():
    return {"W": 640, "H": 360, "g_px": 900.0, "restitution": 0.7, "r_px": 12.0, "ground_y_px": 320.0,
            "x0": 60.0, "y0": 80.0, "vx": 150.0, "duration": 3.0}


def render_ball(p, times):
    W, H = p["W"], p["H"]
    frames = []
    for t in times:
        img = canvas(W, H, (60, 45, 35))
        rect(img, 0, p["ground_y_px"], W, H, (70, 110, 90))
        rect(img, 400, 230, 480, 238, (90, 90, 120))  # static decoration
        x, y, _ = ball_state(t, p)
        disc(img, x, y, p["r_px"], (60, 200, 250))
        frames.append(down(img))
    return frames


def make_ball(out):
    p = ball_params()
    fps = 60
    n = int(p["duration"] * fps)
    times = [i / fps for i in range(n)]
    write_video(out / "ball.mp4", render_ball(p, times), fps)
    # VFR twin: jittered frame times, same physics
    rng = np.random.default_rng(7)
    vt, t = [], 0.0
    while t < p["duration"]:
        vt.append(round(t, 3))
        t += float(rng.choice([1 / 60, 1 / 40, 1 / 30, 1 / 50]))
    write_vfr(out / "ball_vfr.mp4", render_ball(p, vt), vt)
    # heavy compression + noise twin
    noisy = [np.clip(f.astype(np.int16) + rng.normal(0, 6, f.shape), 0, 255).astype(np.uint8)
             for f in render_ball(p, times)]
    write_video(out / "ball_noisy.mp4", noisy, fps, crf=32)
    _, _, bounces = ball_state(p["duration"], p)
    H = p["H"]
    truth = {
        "params": p, "fps": fps,
        "g_fh": p["g_px"] / H, "g_ref": p["g_px"] / (2 * p["r_px"]),
        "ref": "ball diameter", "ref_px": 2 * p["r_px"],
        "vx_fh": p["vx"] / H, "restitution": p["restitution"], "bounces": bounces,
        "start_box_px": [p["x0"] - p["r_px"] - 3, p["y0"] - p["r_px"] - 3, 2 * p["r_px"] + 6, 2 * p["r_px"] + 6],
        "vfr_times": vt,
    }
    (out / "ball.truth.json").write_text(json.dumps(truth, indent=1))


# ---------------------------------------------------------------- sprite poses

POSES = ["idle_a", "idle_b", "run1", "run2", "run3", "run4", "jump"]


def draw_pose(img, cx, base, pose):
    body = (80, 200, 255)
    dark = (40, 90, 160)
    torso_dy = 3 if pose == "idle_b" else 0  # breathing: torso and head 3 px lower
    head_y = base - 70 + torso_dy
    disc(img, cx, head_y, 10, body)
    rect(img, cx - 10, base - 58 + torso_dy, cx + 10, base - 25, body)
    legs = {"idle_a": [(-8, 0), (8, 0)], "idle_b": [(-8, 0), (8, 0)],
            "run1": [(-16, 0), (12, -6)], "run2": [(-4, -10), (4, 0)],
            "run3": [(16, 0), (-12, -6)], "run4": [(6, -8), (-6, 2)], "jump": [(-12, -14), (12, -14)]}[pose]
    for dx, dy in legs:
        cv2.line(img, (int(round(cx * SS)), int(round((base - 25) * SS))), (int(round((cx + dx) * SS)), int(round((base + dy) * SS))), dark, 6 * SS)
    arms = {"idle_a": [(-16, 20), (16, 20)], "idle_b": [(-16, 20), (16, 20)],
            "run1": [(14, 8), (-14, 16)], "run2": [(4, 18), (-4, 18)], "run3": [(-14, 8), (14, 16)],
            "run4": [(8, 14), (-8, 20)], "jump": [(-18, -10), (18, -10)]}[pose]
    for dx, dy in arms:
        cv2.line(img, (int(round(cx * SS)), int(round((base - 52 + torso_dy) * SS))),
                 (int(round((cx + dx) * SS)), int(round((base - 52 + torso_dy + dy) * SS))), dark, 5 * SS)


def make_sprite(out):
    W, H, fps = 480, 270, 30
    seq = [("idle_a", 8), ("idle_b", 8), ("idle_a", 8), ("idle_b", 8)]
    seq += [("run1", 4), ("run2", 4), ("run3", 4), ("run4", 4)] * 3
    seq += [("jump", 14), ("idle_a", 12)]
    frames, timeline, i, x = [], [], 0, 80.0
    xs_truth, base_truth = [], []
    for pose, n in seq:
        timeline.append({"pose": pose, "f0": i, "f1": i + n, "t0": i / fps, "t1": (i + n) / fps, "frames": n})
        for k in range(n):
            base = 220.0
            if pose.startswith("run"):
                x += 4
            if pose == "jump":
                s = k / (n - 1)
                base -= 60 * 4 * s * (1 - s)
                x += 3
            img = canvas(W, H, (50, 35, 30))
            rect(img, 0, 222, W, H, (60, 90, 70))
            draw_pose(img, x, base, pose)
            xs_truth.append(x)
            base_truth.append(base)
            frames.append(down(img))
            i += 1
    write_video(out / "sprite.mp4", frames, fps)
    truth = {"fps": fps, "frames": i, "timeline": timeline, "unique_poses": POSES,
             "start_box_px": [56, 136, 48, 84], "x_center": xs_truth, "base_y": base_truth, "note": "idle_b differs from idle_a only by a 3 px torso/head shift"}
    (out / "sprite.truth.json").write_text(json.dumps(truth, indent=1))


# ---------------------------------------------------------------- easing / spring / fade

def cubic_bezier(x1, y1, x2, y2):
    def f(x):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            u = (lo + hi) / 2
            bx = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u ** 2 * x2 + u ** 3
            if bx < x:
                lo = u
            else:
                hi = u
        u = (lo + hi) / 2
        return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u ** 2 * y2 + u ** 3
    return f


def ease_out_back(p, c1=1.70158):
    c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def spring(t, k, c, m=1.0):
    """Unit step response of m x'' + c x' + k x = k (from rest at 0)."""
    w0 = math.sqrt(k / m)
    z = c / (2 * math.sqrt(k * m))
    wd = w0 * math.sqrt(1 - z * z)
    return 1 - math.exp(-z * w0 * t) * (math.cos(wd * t) + z / math.sqrt(1 - z * z) * math.sin(wd * t))


def make_ease(out):
    W, H, fps, dur = 640, 360, 60, 5.0
    bz = (0.25, 0.1, 0.25, 1.0)
    ease = cubic_bezier(*bz)
    K, C = 120.0, 9.0
    bg, fg = np.array((40, 30, 30), float), np.array((40, 140, 250), float)
    frames, gt = [], []
    for i in range(int(dur * fps)):
        t = i / fps
        x, s, a = 100.0, 1.0, 1.0
        if t >= 0.3:
            x = 100 + 400 * ease(min(1.0, (t - 0.3) / 1.0))
        if t >= 1.7:
            s = 1 + 0.5 * ease_out_back(min(1.0, (t - 1.7) / 0.6))
        if t >= 2.6:
            x = 500 - 300 * spring(t - 2.6, K, C)
        if t >= 3.8:
            a = 1 - 0.8 * min(1.0, (t - 3.8) / 0.8)
        img = canvas(W, H, tuple(int(c) for c in bg))
        half = 30 * s
        col = tuple(int(c) for c in bg + (fg - bg) * a)
        rect(img, x - half, 180 - half, x + half, 180 + half, col)
        frames.append(down(img))
        gt.append({"t": t, "x": x, "scale": s, "alpha": a})
    write_video(out / "ease.mp4", frames, fps)
    truth = {"fps": fps, "segments": [
        {"prop": "x", "t0": 0.3, "t1": 1.3, "from_px": 100, "to_px": 500, "model": "cubic-bezier", "params": bz},
        {"prop": "scale", "t0": 1.7, "t1": 2.3, "from": 1.0, "to": 1.5, "model": "easeOutBack", "params": [1.70158]},
        {"prop": "x", "t0": 2.6, "t1": 3.8, "from_px": 500, "to_px": 200, "model": "spring",
         "params": {"stiffness": K, "damping": C, "mass": 1.0, "zeta": C / (2 * math.sqrt(K)), "omega0": math.sqrt(K)}},
        {"prop": "alpha", "t0": 3.8, "t1": 4.6, "from": 1.0, "to": 0.2, "model": "linear"}],
        "start_box_px": [60, 140, 80, 80], "samples": gt}
    (out / "ease.truth.json").write_text(json.dumps(truth, indent=1))


def make_static(out):
    """Nothing moves; per-frame sensor noise + heavy compression. Expected: zero motion segments."""
    W, H, fps = 480, 270, 30
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 3.0)
    frames = [np.clip(base.astype(np.int16) + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8) for _ in range(90)]
    write_video(out / "static_noisy.mp4", frames, fps, crf=30)
    (out / "static_noisy.truth.json").write_text(json.dumps({"motion_segments": 0, "camera": "static"}))


# ---------------------------------------------------------------- format zoo for "any video"

def make_formats(out):
    src = out / "scenes.mp4"
    jobs = {
        "fmt_vp9.webm": ["-c:v", "libvpx-vp9", "-b:v", "1M", "-c:a", "libopus"],
        "fmt_mjpeg.avi": ["-c:v", "mjpeg", "-q:v", "4", "-an"],
        "fmt_anim.gif": ["-vf", "fps=15,scale=320:-1", "-an"],
        "fmt_rot90.mp4": ["-c", "copy", "-display_rotation", "90"],
        "fmt_offset.mkv": ["-c", "copy", "-output_ts_offset", "1.5"],
        "fmt_tiny.mp4": ["-t", "0.4", "-c:v", "libx264", "-an"],
        "fmt_oddsize.mp4": ["-vf", "scale=333:187", "-c:v", "libx264", "-pix_fmt", "yuv444p", "-an"],
    }
    for name, opts in jobs.items():
        pre = []
        if name == "fmt_rot90.mp4":
            pre = ["-display_rotation", "90"]
            opts = ["-c", "copy"]
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *pre, "-i", str(src), *opts,
                            str(out / name)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"skip {name}: {r.stderr.strip()[:200]}")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "clips")
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2:] or ["scenes", "ball", "sprite", "ease", "static", "formats"]
    for w in which:
        globals()[f"make_{w}"](out)
        print("made", w)
