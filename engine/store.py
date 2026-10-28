"""Cache directory + video.json for one video.

Every module writes one section of video.json together with a `_meta` block
(module version, params, deps). A module whose meta still matches is skipped,
so repeated questions about the same video never recompute anything.
"""
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

SCHEMA = "video.json/1"


def cache_root() -> Path:
    return Path(os.environ.get("VE_CACHE") or (Path.home() / ".video-engine" / "cache"))


def video_key(path) -> str:
    """Content key: size + first and last MiB. Survives copy/rename-safe moves."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), 2)
            h.update(f.read())
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem)[:40] or "video"
    return f"{stem}-{h.hexdigest()[:12]}"


# ---------- honest labels ----------

def lab(v, src, conf, unit=None, note=None):
    d = {"v": v, "src": src, "conf": round(float(conf), 2)}
    if unit:
        d["unit"] = unit
    if note:
        d["note"] = note
    return d


def M(v, conf=1.0, unit=None, note=None):
    """measured: read directly from the video/stream."""
    return lab(v, "measured", conf, unit, note)


def D(v, conf=0.8, unit=None, note=None):
    """derived: computed from measurements by a model or fit."""
    return lab(v, "derived", conf, unit, note)


def A(v, conf=0.5, unit=None, note=None):
    """assumed: a default or guess not backed by the data."""
    return lab(v, "assumed", conf, unit, note)


def compact(o, sig=6):
    """Round floats recursively to `sig` significant digits to keep json small."""
    if isinstance(o, float):
        if not math.isfinite(o):
            return None
        return float(f"{o:.{sig}g}")
    if isinstance(o, dict):
        return {k: compact(v, sig) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [compact(v, sig) for v in o]
    if hasattr(o, "tolist"):  # numpy scalars/arrays
        return compact(o.tolist(), sig)
    return o


class Store:
    def __init__(self, video):
        self.video = Path(video).resolve()
        if not self.video.exists():
            raise FileNotFoundError(self.video)
        self.key = video_key(self.video)
        self.dir = cache_root() / self.key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.dir / "video.json"
        if self.json_path.exists():
            self.data = json.loads(self.json_path.read_text(encoding="utf-8"))
        else:
            self.data = {"schema": SCHEMA, "source": {"path": str(self.video), "key": self.key}}
        self.data["source"]["path"] = str(self.video)  # latest known location

    def save(self):
        tmp = self.json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(compact(self.data), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.json_path)

    def file(self, name) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ---- sections addressed by a path like ("tracks", "1") ----
    def get(self, *path):
        d = self.data
        for k in path:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def put(self, path, payload, version, params, deps=(), t_start=None):
        if isinstance(path, str):
            path = (path,)
        payload = dict(payload)
        payload["_meta"] = {
            "version": version,
            "params": params,
            "deps": {"/".join(d): self.stamp(d) for d in deps},
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.time() - t_start, 3) if t_start else None,
        }
        d = self.data
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = payload
        self.save()
        return payload

    def stamp(self, path):
        if isinstance(path, str):
            path = (path,)
        meta = (self.get(*path) or {}).get("_meta") or {}
        return meta.get("computed_at")

    def fresh(self, path, version, params):
        """True if the section exists with the same version, params and unchanged deps."""
        if isinstance(path, str):
            path = (path,)
        sec = self.get(*path)
        if not sec or "_meta" not in sec:
            return False
        m = sec["_meta"]
        if m.get("version") != version or m.get("params") != json.loads(json.dumps(params)):
            return False
        for dep, stamp in (m.get("deps") or {}).items():
            if self.stamp(tuple(dep.split("/"))) != stamp:
                return False
        return True
