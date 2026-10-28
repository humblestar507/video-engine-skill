"""ffmpeg/ffprobe access. Frames always come paired with their real PTS
(parsed from the showinfo filter), never with a frame index times 1/fps."""
import json
import queue
import re
import subprocess
import threading

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

_SI = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[0-9.]+(?:e-?\d+)?)")


def ffprobe_json(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:400]}")
    return json.loads(out.stdout)


def packet_pts(path, stream_index):
    """Presentation timestamps of all packets of a stream (no decoding), sorted."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", str(stream_index),
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    vals = []
    for line in out.stdout.splitlines():
        line = line.strip().strip(",")
        try:
            vals.append(float(line))
        except ValueError:
            pass  # N/A
    arr = np.array(vals, dtype=np.float64)
    uniq = np.unique(arr)
    return uniq, len(arr) - len(uniq)


class Geometry:
    """Decoded geometry: storage size after autorotation."""

    def __init__(self, w, h, sar=1.0):
        self.w, self.h, self.sar = int(w), int(h), float(sar)

    def scaled(self, width=None, height=None):
        if width is None and height is None:
            return self.w, self.h
        if width is not None:
            height = max(2, int(round(self.h * width / self.w / 2)) * 2)
        else:
            width = max(2, int(round(self.w * height / self.h / 2)) * 2)
        return int(width), int(height)


def iter_frames(path, geom, t_origin, width=None, gray=False, t0=None, t1=None, vf_extra=None):
    """Yield (t, frame) for every decoded frame, t in seconds from the first
    video frame (PTS based). frame is uint8 HxW (gray) or HxWx3 (BGR)."""
    w, h = geom.scaled(width)
    pix = "gray" if gray else "bgr24"
    ch = 1 if gray else 3
    vf = "showinfo"
    if (w, h) != (geom.w, geom.h):
        vf += f",scale={w}:{h}:flags=area"
    if vf_extra:
        vf += "," + vf_extra
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-loglevel", "info", "-copyts"]
    if t0 is not None and t0 > 0:
        # -ss is relative to the file start; t_origin is the first video pts (absolute)
        cmd += ["-ss", f"{max(0.0, t0 + t_origin['seek_shift']):.6f}"]
    cmd += ["-i", str(path), "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough",
            "-an", "-sn", "-f", "rawvideo", "-pix_fmt", pix, "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 7)
    q = queue.Queue()

    def read_err():
        for raw in p.stderr:
            m = _SI.search(raw.decode("utf-8", "replace"))
            if m and "Parsed_showinfo" in raw.decode("utf-8", "replace"):
                q.put(float(m.group(3)))
        q.put(None)

    th = threading.Thread(target=read_err, daemon=True)
    th.start()
    size = w * h * ch
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            pts = q.get(timeout=60)
            if pts is None:
                break
            t = pts - t_origin["v_start"]
            if t1 is not None and t > t1 + 1e-9:
                break
            if t0 is not None and t < t0 - 1e-6:
                continue
            fr = np.frombuffer(buf, np.uint8).reshape((h, w) if gray else (h, w, 3))
            yield t, fr
    finally:
        if p.poll() is None:
            p.kill()
        p.stdout.close()
        p.wait()


def grab(path, geom, t_origin, t, width=None, gray=False):
    """First frame with pts >= t (minus a quarter frame tolerance). Returns (t_actual, frame)."""
    eps = t_origin.get("dt", 1 / 30) * 0.25
    for ta, fr in iter_frames(path, geom, t_origin, width=width, gray=gray, t0=max(0.0, t - eps)):
        return ta, fr.copy()
    return None, None


def extract_audio(path, out_wav, sr=22050):
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                        "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(out_wav)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:400])
    return out_wav
