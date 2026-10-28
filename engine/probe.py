"""Layer 0: real frame timings (PTS), fps, resolution, VFR, audio presence."""
import time

import numpy as np

from . import media
from .store import D, M

VERSION = "2"


def _rotation(vs):
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"])))
    rot = (vs.get("tags") or {}).get("rotate")
    return int(rot) if rot else 0


def _ratio(s):
    try:
        a, b = s.split(":") if ":" in s else s.split("/")
        return float(a) / float(b) if float(b) else None
    except Exception:
        return None


def run(store, force=False):
    params = {}
    if not force and store.fresh("probe", VERSION, params):
        return store.get("probe"), True
    t_start = time.time()
    info = media.ffprobe_json(store.video)
    streams = info.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"
               and not (s.get("disposition") or {}).get("attached_pic")), None)
    if vs is None:
        raise RuntimeError("no video stream")
    aus = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = info.get("format", {})

    rot = _rotation(vs)
    w, h = int(vs["width"]), int(vs["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    sar = _ratio(vs.get("sample_aspect_ratio", "1:1") or "1:1") or 1.0

    warnings = []
    pts, dups = media.packet_pts(store.video, vs["index"])
    pts_source = "packets"
    if dups:
        warnings.append(f"{dups} packets share a pts with another packet; counted once")
    fmt_start = float(fmt.get("start_time") or 0.0)
    if len(pts) < 2:
        warnings.append("packet pts unavailable; decoding to read frame pts")
        pts_source = "decode"
        geom = media.Geometry(w, h, sar)
        origin = {"v_start": 0.0, "seek_shift": 0.0}
        pts = np.array([t for t, _ in media.iter_frames(store.video, geom, origin, width=32, gray=True)])
    v_start = float(pts[0]) if len(pts) else float(vs.get("start_time") or 0.0)
    rel = pts - v_start
    np.save(store.file("pts.npy"), rel)

    dt = np.diff(rel)
    med = float(np.median(dt)) if len(dt) else None
    n = int(len(rel))
    # time base quantization: tolerate jitter of one tick plus 1%
    tb = _ratio(vs.get("time_base", "1/1000")) or 1e-3
    tol = max(tb * 1.5, (med or 0) * 0.01)
    irregular = int(np.sum(np.abs(dt - med) > tol)) if len(dt) else 0
    vfr = irregular > max(2, 0.01 * len(dt))
    duration = float(rel[-1] + (med or 0)) if n else float(fmt.get("duration") or 0)

    out = {
        "file": {"path": str(store.video), "size_bytes": int(fmt.get("size") or store.video.stat().st_size),
                 "container": fmt.get("format_name"), "bitrate": int(fmt.get("bit_rate") or 0)},
        "video": {
            "codec": vs.get("codec_name"), "pix_fmt": vs.get("pix_fmt"),
            "width": w, "height": h, "sar": sar, "rotation": rot,
            "frames": M(n, 1.0, note=f"pts source: {pts_source}"),
            "duration": M(duration, 1.0, "s"),
            "fps": M((n - 1) / (rel[-1] - rel[0]) if not vfr and n > 1 else (1.0 / med if med else None),
                     0.95 if not vfr else 0.6, "1/s",
                     note="mean rate over the clip" if not vfr else "VFR: 1/median frame interval"),
            "fps_declared": _ratio(vs.get("r_frame_rate", "0/1")),
            "vfr": D(bool(vfr), 0.9, note=f"{irregular} of {len(dt)} intervals deviate > {tol*1000:.2f} ms"),
            "dt_ms": {"median": med * 1000 if med else None,
                      "min": float(dt.min() * 1000) if len(dt) else None,
                      "max": float(dt.max() * 1000) if len(dt) else None},
        },
        "audio": None if aus is None else {
            "codec": aus.get("codec_name"), "sample_rate": int(aus.get("sample_rate") or 0),
            "channels": aus.get("channels"),
            "offset_s": M(float(aus.get("start_time") or 0.0) - v_start, 0.9,
                          "s", note="audio start relative to first video frame"),
        },
        "timeline": {"t0": "first video frame pts", "v_start_abs": v_start,
                     "seek_shift": v_start - fmt_start, "units": "seconds"},
        "units": {"pos": "fh = fraction of frame height (x and y, isotropic; y down)",
                  "time": "s from first frame PTS",
                  "ref": "set by track --ref; positions also given in ref units"},
        "warnings": warnings,
    }
    if sar != 1.0:
        out["warnings"].append(f"non-square pixels sar={sar}; x positions are corrected by sar")
    return store.put("probe", out, VERSION, params, t_start=t_start), False


def origin(store):
    """Timeline helper for media.iter_frames."""
    p = store.get("probe")
    tl = p["timeline"]
    med = (p["video"]["dt_ms"]["median"] or 33.3) / 1000
    return {"v_start": tl["v_start_abs"], "seek_shift": tl["seek_shift"], "dt": med}


def geometry(store):
    v = store.get("probe")["video"]
    return media.Geometry(v["width"], v["height"], v.get("sar", 1.0))


def pts(store):
    return np.load(store.file("pts.npy"))
