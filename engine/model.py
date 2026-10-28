"""Channel models shared by render, export and verification.

A channel is a JSON dict describing one scalar over time (seconds):
  {"type": "const", "v": c}
  {"type": "linear", "x0": a, "vx": b}                      a + b*t
  {"type": "samples", "t": [...], "v": [...]}               replay of measured samples
  {"type": "segments", "initial": v, "segments": [seg...]}  eased transitions, in time order
  {"type": "ballistic", "t_ref", "y0", "v0", "g", "ground", "restitution"}
seg = {"model": linear|<css preset>|bezier|spring, "t_start", "duration", "from", "to",
       "bezier": [x1,y1,x2,y2] (model bezier), "omega0", "zeta" (model spring)}
The JS and GDScript exporters implement exactly this file; tests compare them."""
import numpy as np

from .curves import PRESETS, bez_progress, spring_step


def seg_progress(seg, t):
    t = np.asarray(t, float)
    m = seg["model"]
    if m == "spring":
        return spring_step(t - seg["t_start"], seg["omega0"], seg["zeta"])
    p = np.clip((t - seg["t_start"]) / seg["duration"], 0.0, 1.0)
    shape = seg["bezier"] if m == "bezier" else PRESETS[m]
    return bez_progress(p, *shape)


def eval_segments(ch, t):
    t = np.asarray(t, float)
    val = np.full(t.shape, float(ch["initial"]))
    for seg in ch["segments"]:
        mask = t >= seg["t_start"]
        if mask.any():
            p = seg_progress(seg, t[mask])
            val[mask] = seg["from"] + (seg["to"] - seg["from"]) * p
    return val


def flights(ch, t_end):
    """List of free flights [(t0, y0, v0, t1)] until t_end; y down, impact reflects with restitution."""
    g, ground, e = ch["g"], ch.get("ground"), ch.get("restitution", 0.0)
    t0, y0, v0 = ch["t_ref"], ch["y0"], ch["v0"]
    out = []
    for _ in range(80):
        if ground is None or g <= 0:
            out.append((t0, y0, v0, float("inf")))
            break
        disc = v0 * v0 + 2 * g * (ground - y0)
        tau = (-v0 + np.sqrt(max(disc, 0.0))) / g
        t1 = t0 + tau
        out.append((t0, y0, v0, t1))
        if t1 >= t_end or tau < 1e-4:
            break
        v_in = v0 + g * tau
        t0, y0, v0 = t1, ground, -e * v_in
        if abs(v0) < 1e-4:
            out.append((t0, ground, 0.0, float("inf")))
            break
    return out


def eval_ballistic(ch, t):
    t = np.asarray(t, float)
    fl = flights(ch, float(np.max(t)) if t.size else 0.0)
    starts = np.array([f[0] for f in fl])
    idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(fl) - 1)
    out = np.empty(t.shape)
    g = ch["g"]
    for k, f in enumerate(fl):
        m = idx == k
        if m.any():
            tau = np.maximum(t[m] - f[0], 0.0)
            y = f[1] + f[2] * tau + 0.5 * g * tau * tau
            if ch.get("ground") is not None:
                y = np.minimum(y, ch["ground"])
            out[m] = y
    return out


def eval_channel(ch, t):
    t = np.asarray(t, float)
    k = ch["type"]
    if k == "const":
        return np.full(t.shape, float(ch["v"]))
    if k == "linear":
        return ch["x0"] + ch["vx"] * t
    if k == "samples":
        return np.interp(t, ch["t"], ch["v"])
    if k == "segments":
        return eval_segments(ch, t)
    if k == "ballistic":
        return eval_ballistic(ch, t)
    raise ValueError(f"unknown channel type {k}")


def eval_object(obj, t):
    c = obj["channels"]
    return {k: eval_channel(c[k], t) for k in ("x", "y", "scale", "opacity")}


def css_ballistic_flights(ch, t_end):
    """Flights as exact CSS cubic-beziers. A parabola over [0,T] with dy != 0 is
    cubic-bezier(1/3, a, 2/3, a + 1/3), a = v0*T/(3*dy). A bounce starts and ends on the
    ground (dy = 0), so a flight that goes up is split at its apex: up = (1/3,2/3,2/3,1),
    down = (1/3,0,2/3,1/3)."""
    g = ch["g"]
    out = []

    def piece(ta, tb, ya, va):
        T = tb - ta
        if T <= 1e-6:
            return
        yb = ya + va * T + 0.5 * g * T * T
        dy = yb - ya
        if abs(dy) < 1e-9:
            return
        a = va * T / dy / 3
        out.append({"t0": ta, "t1": tb, "from": ya, "to": yb, "bezier": [1 / 3, a, 2 / 3, a + 1 / 3]})

    for (t0, y0, v0, t1) in flights(ch, t_end):
        tb = min(t1, t_end)
        if v0 < 0 and g > 0:
            ta = t0 - v0 / g
            if ta < tb:
                ya = y0 + v0 * (ta - t0) + 0.5 * g * (ta - t0) ** 2
                piece(t0, ta, y0, v0)
                piece(ta, tb, ya, 0.0)
                if ch.get("ground") is not None and t1 <= t_end and out:
                    out[-1]["to"] = ch["ground"]
                continue
        piece(t0, tb, y0, v0)
        if ch.get("ground") is not None and t1 <= t_end and out and out[-1]["t1"] == tb:
            out[-1]["to"] = ch["ground"]
    return out
