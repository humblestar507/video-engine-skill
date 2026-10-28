"""Layer 2: fit physics and easing to tracked signals.

ballistic: free flights with one shared acceleration g, impacts as the exact
           intersection of consecutive parabolas (sub-frame time), restitution,
           apex time/height, separate g on the way up and down.
transitions: any scalar channel is cut into motion intervals; each is fitted by
           linear / CSS preset / free cubic-bezier / damped spring with the start
           time, duration and both levels free. Variable projection: the levels
           are solved linearly for every nonlinear guess.

Everything is expressed in fh (fraction of frame height) and, where the track
has a reference size, in reference units. Fit quality is reported as RMSE and as
a ratio to the measured noise; ambiguous results are flagged, not hidden."""
import time

import numpy as np
from scipy.optimize import least_squares

from .store import D, M

VERSION = "4"

PRESETS = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease-in": (0.42, 0.0, 1.0, 1.0),
    "ease-out": (0.0, 0.0, 0.58, 1.0),
    "ease-in-out": (0.42, 0.0, 0.58, 1.0),
    "easeOutQuad": (0.25, 0.46, 0.45, 0.94),
    "easeInQuad": (0.55, 0.085, 0.68, 0.53),
    "easeOutCubic": (0.215, 0.61, 0.355, 1.0),
    "easeInCubic": (0.55, 0.055, 0.675, 0.19),
    "easeInOutCubic": (0.645, 0.045, 0.355, 1.0),
    "easeOutExpo": (0.16, 1.0, 0.3, 1.0),
    "easeOutBack": (0.34, 1.56, 0.64, 1.0),
    "easeInBack": (0.36, 0.0, 0.66, -0.56),
    "easeInOutBack": (0.68, -0.6, 0.32, 1.6),
}


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def signal(store, tid, channel):
    """(t, y, unit, note) for a track channel; NaN/lost rows dropped."""
    tr = store.get("tracks", str(tid))
    if tr is None:
        raise KeyError(f"no track {tid}; run `ve track` first")
    s = tr["samples"]
    t = arr(s["t"])
    lost = arr(s["lost"]) > 0
    if channel in ("x", "y"):
        y, unit, note = arr(s[channel]), "fh", "track centre"
    elif channel == "size":
        y, unit, note = arr(s["size"]), "fh", "area-equivalent side"
    elif channel in ("scale", "opacity"):
        raw = arr(s["size"] if channel == "scale" else s["contrast"])
        good = np.where(np.isfinite(raw) & ~lost)[0][:5]
        if len(good) == 0:
            raise ValueError(f"no {channel} measurements in track {tid}")
        base = float(np.median(raw[good]))
        y, unit = raw / base, "x first-5-samples"
        note = f"normalised: first samples taken as 1.0 (assumed, base={base:.4g})"
    else:
        raise ValueError("channel must be x, y, size, scale or opacity")
    m = np.isfinite(y) & ~lost
    return t[m], y[m], unit, note, tr


def noise_sigma(t, y):
    d2 = np.diff(y, 2)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)) if len(d2) > 6 else 0.0


# ---------------------------------------------------------------- ballistic

def run_ballistic(store, tid, axis="y", force=False):
    tid = str(tid)
    params = {"axis": axis}
    key = f"ballistic_{tid}"
    if not force and store.fresh(("curves", key), VERSION, params) and \
            (store.get("curves", key)["_meta"]["deps"].get(f"tracks/{tid}") == store.stamp(("tracks", tid))):
        return store.get("curves", key), True
    t_start = time.time()
    t, y, _, _, tr = signal(store, tid, axis)
    _, x, _, _, _ = signal(store, tid, "x" if axis == "y" else "y")
    ref = tr["ref"]
    n = len(t)
    if n < 12:
        raise ValueError("track too short for a ballistic fit")
    dt0, dt1 = np.diff(t)[:-1], np.diff(t)[1:]
    v0, v1 = np.diff(y)[:-1] / dt0, np.diff(y)[1:] / dt1
    a = 2 * (v1 - v0) / (dt0 + dt1)  # at frames 1..n-2
    # rest: >= 6 samples spanning >= 0.1 s within 0.6 px (an apex bends away from this quadratically)
    rest_dev = 0.6 / tr["analysis_height_px"]  # fixed: second differences are not noise on irregular (VFR) times
    rest = np.zeros(n, bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (y[i:j + 2].max() - y[i:j + 2].min()) < rest_dev:
            j += 1
        if j - i + 1 >= 6 and t[j] - t[i] >= 0.1:
            rest[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    rest_a = rest[1:-1] | rest[:-2] | rest[2:]  # accelerations touching a resting sample say nothing about g
    g0 = float(np.median(a[~rest_a])) if (~rest_a).sum() >= 5 else float(np.median(a))
    sig_a = float(1.4826 * np.median(np.abs(a[~rest_a] - g0))) if (~rest_a).sum() >= 5 else float(1.4826 * np.median(np.abs(a - g0)))
    thr = max(8 * sig_a, 0.5 * abs(g0), 1e-6)
    spikes = [i + 1 for i in range(len(a)) if abs(a[i] - g0) > thr and not rest_a[i]]
    # exclusion clusters: spike frames, and resting runs (one sample margin)
    marks = sorted(set(spikes) | set(np.nonzero(rest)[0].tolist()) | set(np.nonzero(np.convolve(rest, [1, 1, 1], "same") > 0)[0].tolist()))
    clusters = []
    for s_ in marks:
        if clusters and s_ - clusters[-1][-1] <= 1:
            clusters[-1].append(s_)
        else:
            clusters.append([s_])
    bounds, start = [], 0
    for c in clusters:
        bounds.append((start, c[0] - 1))  # inclusive index ranges of flights
        start = c[-1] + 1
    bounds.append((start, n - 1))
    rests = []
    i = 0
    while i < n:
        if rest[i]:
            j = i
            while j + 1 < n and rest[j + 1]:
                j += 1
            rests.append({"t0": float(t[i]), "t1": float(t[j]), "y_fh": float(np.median(y[i:j + 1]))})
            i = j + 1
        else:
            i += 1
    flights = [(a0, b0) for a0, b0 in bounds if b0 - a0 + 1 >= 6]
    if not flights:
        raise ValueError("no free-flight segments found (acceleration is not constant anywhere)")
    # joint LSQ: y = y0_k + v_k (t - tr_k) + g/2 (t - tr_k)^2, shared g
    K = len(flights)
    rows, rhs = [], []
    for k, (a0, b0) in enumerate(flights):
        tt = t[a0:b0 + 1] - t[a0]
        for ti, yi in zip(tt, y[a0:b0 + 1]):
            r = np.zeros(1 + 2 * K)
            r[0] = 0.5 * ti * ti
            r[1 + 2 * k] = 1.0
            r[2 + 2 * k] = ti
            rows.append(r)
            rhs.append(yi)
    A, b = np.array(rows), np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ sol - b
    dof = max(1, len(b) - A.shape[1])
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.pinv(A.T @ A)
    g, g_sd = float(sol[0]), float(np.sqrt(cov[0, 0]))
    fl = []
    for k, (a0, b0) in enumerate(flights):
        y0, vk = float(sol[1 + 2 * k]), float(sol[2 + 2 * k])
        tref = float(t[a0])
        off = sum(b_ - a_ + 1 for a_, b_ in flights[:k])
        rm = float(np.sqrt(np.mean(resid[off:off + b0 - a0 + 1] ** 2)))
        e = {"t0": float(t[a0]), "t1": float(t[b0]), "samples": b0 - a0 + 1, "y0": y0, "v0": vk, "tref": tref, "rmse_fh": rm}
        if abs(g) > 1e-9:
            ta = tref - vk / g
            if t[a0] + 1e-3 < ta < t[b0]:
                e["apex"] = {"t": float(ta), "y_fh": float(y0 + vk * (ta - tref) + 0.5 * g * (ta - tref) ** 2)}
                up, dn = (t >= t[a0]) & (t <= ta), (t >= ta) & (t <= t[b0])
                for nm, m in (("g_up", up), ("g_down", dn)):
                    if m.sum() >= 6:
                        c = np.polyfit(t[m] - ta, y[m], 2)
                        e[nm] = D(float(2 * c[0]), 0.5, "fh/s2", "separate quadratic fit on that half of the flight")
        # horizontal speed on this flight
        px = np.polyfit(t[a0:b0 + 1], x[a0:b0 + 1], 1)
        e["vx_fh_s"] = float(px[0])
        fl.append(e)
    # impacts: shared g makes the parabola difference linear in t
    impacts = []
    for k in range(K - 1):
        A_, B_ = fl[k], fl[k + 1]
        cA = (A_["y0"] - A_["v0"] * A_["tref"] + 0.5 * g * A_["tref"] ** 2, A_["v0"] - g * A_["tref"])  # y = cA0 + cA1 t + g/2 t^2
        cB = (B_["y0"] - B_["v0"] * B_["tref"] + 0.5 * g * B_["tref"] ** 2, B_["v0"] - g * B_["tref"])
        d1 = cA[1] - cB[1]
        if abs(d1) < 1e-9:
            continue
        ti = (cB[0] - cA[0]) / d1
        if not (A_["t1"] - 1e-9 <= ti <= B_["t0"] + 1e-9):
            # a kink can lie one frame outside the cleaned windows; accept within one dt but flag
            if not (A_["t1"] - 0.5 * np.median(np.diff(t)) <= ti <= B_["t0"] + 0.5 * np.median(np.diff(t))):
                continue
        v_in = A_["v0"] + g * (ti - A_["tref"])
        v_out = B_["v0"] + g * (ti - B_["tref"])
        if not (v_in > 0 > v_out and 0.02 < -v_out / v_in < 1.5):
            continue  # not an impact: velocity must reverse (a split inside one flight has no such jump)
        yi = A_["y0"] + A_["v0"] * (ti - A_["tref"]) + 0.5 * g * (ti - A_["tref"]) ** 2
        impacts.append({"t": float(ti), "y_fh": float(yi), "v_in_fh_s": float(v_in), "v_out_fh_s": float(v_out),
                        "restitution": D(float(-v_out / v_in) if abs(v_in) > 1e-9 else None, 0.7,
                                         note="-v_out/v_in; assumes a static surface"),
                        "src": "derived"})
    e_vals = [im["restitution"]["v"] for im in impacts if im["restitution"]["v"] is not None]
    vxs = [f["vx_fh_s"] for f in fl]
    refv = ref["fh"]
    out = {
        "axis": axis, "track": tid, "n_samples": n, "flights": fl, "impacts": impacts,
        "g": D(g, round(float(np.clip(1 - 20 * g_sd / max(abs(g), 1e-9), 0.1, 0.95)), 2), "fh/s2",
               f"shared by {K} flights, 1-sigma {g_sd:.3g}; sign: + is toward +{axis} (down for y)"),
        "g_ref_units": D(g / refv, 0.5 if ref["src"] == "assumed" else 0.9, "ref/s2",
                         f"ref = {ref['what']} ({ref['src']})"),
        "restitution_mean": D(float(np.mean(e_vals)), 0.7) if e_vals else None,
        "restitution_sd": float(np.std(e_vals)) if len(e_vals) > 1 else None,
        "vx_mean_fh_s": D(float(np.mean(vxs)), 0.8, "fh/s", f"per-flight spread {np.std(vxs):.3g}"),
        "rmse_fh": float(np.sqrt(s2)),
        "excluded_frames": int(sum(len(c) for c in clusters)),
        "rests": rests,
        "summary_line": f"g={g:.3f} fh/s2 (+-{g_sd:.3f}), {K} flights, {len(impacts)} impacts"
                        + (f", restitution {np.mean(e_vals):.3f}" if e_vals else "") + f", ref src={ref['src']}",
        "warnings": [],
    }
    if g < 0 and axis == "y":
        out["warnings"].append("g is negative on y (y points down): object accelerates upward - check the track")
    return store.put(("curves", key), out, VERSION, params, deps=[("tracks", tid)], t_start=t_start), False


# ---------------------------------------------------------------- transitions

_U = np.linspace(0, 1, 513)


def bez_progress(p, x1, y1, x2, y2):
    u = _U
    bx = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u ** 2 * x2 + u ** 3
    by = 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u ** 2 * y2 + u ** 3
    return np.interp(p, bx, by)


def spring_step(tau, w0, z):
    tau = np.maximum(tau, 0.0)
    if abs(z - 1) < 1e-3:
        return 1 - np.exp(-w0 * tau) * (1 + w0 * tau)
    if z < 1:
        wd = w0 * np.sqrt(1 - z * z)
        return 1 - np.exp(-z * w0 * tau) * (np.cos(wd * tau) + z * w0 / wd * np.sin(wd * tau))
    s = w0 * np.sqrt(z * z - 1)
    r1, r2 = -z * w0 + s, -z * w0 - s
    return 1 - (r2 * np.exp(r1 * tau) - r1 * np.exp(r2 * tau)) / (r2 - r1)


def _vp(t, y, gfun):
    """Variable projection: levels by linear LSQ for progress g(t). Returns (y0, y1, resid)."""
    g = gfun(t)
    A = np.stack([1 - g, g], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol[0], sol[1], A @ sol - y


def _fit_model(t, y, kind, ta, tb, preset=None):
    """Returns dict(model, k, rmse, params, y0, y1, resid) or None. Times are absolute seconds."""
    span = t[-1] - t[0]
    Dg = max(tb - ta, 0.05)
    best = None

    def run(f, p0, lo, hi):
        nonlocal best
        try:
            r = least_squares(f, p0, bounds=(lo, hi), x_scale="jac", max_nfev=300)
        except Exception:
            return
        if best is None or r.cost < best.cost:
            best = r

    if kind in ("linear", "preset"):
        shape = PRESETS["linear"] if kind == "linear" else PRESETS[preset]

        def f(p):
            return _vp(t, y, lambda tt: bez_progress(np.clip((tt - p[0]) / p[1], 0, 1), *shape))[2]
        for s0 in (ta, ta - 0.1 * Dg, ta + 0.1 * Dg):
            for d0 in (Dg, 1.4 * Dg, 0.7 * Dg):
                run(f, [s0, d0], [t[0] - 0.5 * span, 0.04], [t[-1], 3 * span])
        if best is None:
            return None
        p = best.x
        y0, y1, res = _vp(t, y, lambda tt: bez_progress(np.clip((tt - p[0]) / p[1], 0, 1), *shape))
        return dict(model=kind if kind == "linear" else preset, k=4, params={"t_start": p[0], "duration": p[1]},
                    y0=y0, y1=y1, rmse=float(np.sqrt(np.mean(res ** 2))), jac=best.jac, resid=res)
    if kind == "bezier":
        def f(p):
            return _vp(t, y, lambda tt: bez_progress(np.clip((tt - p[0]) / p[1], 0, 1), *p[2:6]))[2]
        for s0 in (ta, ta - 0.1 * Dg):
            for d0 in (Dg, 1.4 * Dg):
                for shp in ((0.25, 0.1, 0.25, 1.0), (0.4, 0.0, 0.6, 1.0), (0.3, 1.3, 0.6, 1.0), (0.1, 0.5, 0.3, 1.0)):
                    run(f, [s0, d0, *shp], [t[0] - 0.5 * span, 0.04, 0, -0.8, 0, -0.8], [t[-1], 3 * span, 1, 2.2, 1, 2.2])
        if best is None:
            return None
        p = best.x
        y0, y1, res = _vp(t, y, lambda tt: bez_progress(np.clip((tt - p[0]) / p[1], 0, 1), *p[2:6]))
        return dict(model="bezier", k=8, params={"t_start": p[0], "duration": p[1], "bezier": [float(v) for v in p[2:6]]},
                    y0=y0, y1=y1, rmse=float(np.sqrt(np.mean(res ** 2))), jac=best.jac, resid=res)
    if kind == "spring":
        def f(p):
            return _vp(t, y, lambda tt: spring_step(tt - p[0], p[1], p[2]))[2]
        for s0 in (ta, ta - 0.05):
            for w0 in (2 * np.pi / max(Dg, 0.1), 4 * np.pi / max(Dg, 0.1), 10.0):
                for z in (0.2, 0.5, 0.9, 1.5):
                    run(f, [s0, w0, z], [t[0] - 0.3 * span, 0.5, 0.03], [t[-1], 200.0, 4.0])
        if best is None:
            return None
        p = best.x
        y0, y1, res = _vp(t, y, lambda tt: spring_step(tt - p[0], p[1], p[2]))
        return dict(model="spring", k=5, params={"t_start": p[0], "omega0": p[1], "zeta": p[2],
                                                 "stiffness_per_mass": p[1] ** 2, "damping_per_mass": 2 * p[2] * p[1]},
                    y0=y0, y1=y1, rmse=float(np.sqrt(np.mean(res ** 2))), jac=best.jac, resid=res)


def _std(fit, sigma):
    try:
        J = fit["jac"]
        cov = max(fit["rmse"], sigma) ** 2 * np.linalg.inv(J.T @ J)
        return np.sqrt(np.diag(cov))
    except Exception:
        return None


def find_intervals(t, y, sigma):
    v = np.gradient(y, t)
    dtm = float(np.median(np.diff(t)))
    sig_v = 0.7 * sigma / dtm
    R = float(y.max() - y.min())
    thr = max(6 * sig_v, 0.02 * R / max(t[-1] - t[0], 1e-6), 1e-9)
    act = np.abs(v) > thr
    segs, i, n = [], 0, len(t)
    while i < n:
        if act[i]:
            j = i
            gap = 0
            k = i
            while k + 1 < n and (act[k + 1] or t[k + 1] - t[j] < 0.12):
                k += 1
                if act[k]:
                    j = k
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return [(t[a], t[b]) for a, b in segs if t[b] - t[a] >= 0.08]


def run_curve(store, tid, channel, force=False):
    tid = str(tid)
    key = f"{tid}_{channel}"
    params = {"channel": channel}
    if not force and store.fresh(("curves", key), VERSION, params) and \
            (store.get("curves", key)["_meta"]["deps"].get(f"tracks/{tid}") == store.stamp(("tracks", tid))):
        return store.get("curves", key), True
    t_start = time.time()
    t, y, unit, note, tr = signal(store, tid, channel)
    sigma = noise_sigma(t, y)
    iv = find_intervals(t, y, sigma)
    refv = tr["ref"]["fh"] if channel in ("x", "y", "size") else None
    # a transition smaller than this is below what the tracker can resolve (0.25 px) or 0.5 % for ratios
    abs_floor = 0.25 / tr["analysis_height_px"] if channel in ("x", "y", "size") else 0.005
    segs = []
    for k, (ta, tb) in enumerate(iv):
        prev_end = iv[k - 1][1] if k else t[0]
        next_start = iv[k + 1][0] if k + 1 < len(iv) else t[-1]
        m = max(0.2, 0.6 * (tb - ta))
        lo = max(t[0], ta - m, 0.5 * (prev_end + ta) if k else t[0])
        hi = min(t[-1], tb + m, 0.5 * (tb + next_start) if k + 1 < len(iv) else t[-1])
        sel = (t >= lo) & (t <= hi)
        tw, yw = t[sel], y[sel]
        if len(tw) < 8:
            continue
        cands = []
        for kind, pre in (("linear", None), *(("preset", p) for p in PRESETS if p != "linear"), ("spring", None), ("bezier", None)):
            f = _fit_model(tw, yw, kind, ta, tb, pre)
            if f:
                cands.append(f)
        floor = max(sigma, 5e-4 * float(y.max() - y.min()), 1e-9)
        best_rmse = min(c["rmse"] for c in cands)
        tol = 1.3 * best_rmse + 0.5 * floor
        presets = [c for c in cands if c["model"] in PRESETS and c["model"] != "linear"]
        lin = next((c for c in cands if c["model"] == "linear"), None)
        spr = next((c for c in cands if c["model"] == "spring"), None)
        bez = next((c for c in cands if c["model"] == "bezier"), None)
        pre = min(presets, key=lambda c: c["rmse"]) if presets else None
        # simplest adequate model: linear < named preset < spring < free bezier
        chosen = None
        for c in (lin, pre, spr, bez):
            if c is not None and c["rmse"] <= tol:
                chosen = c
                break
        chosen = chosen or min(cands, key=lambda c: c["rmse"])
        amp = abs(chosen["y1"] - chosen["y0"])
        if amp < max(8 * sigma, 0.01 * float(y.max() - y.min()), abs_floor):
            continue  # settling ripple / noise, not a transition
        rel = chosen["rmse"] / amp if amp > 0 else 1.0
        conf = float(np.clip(1 - 10 * rel, 0.1, 0.95))
        sd = _std(chosen, sigma)
        p = dict(chosen["params"])
        ent = {"t_window": [float(lo), float(hi)], "t_detect": [float(ta), float(tb)], "model": chosen["model"],
               "k_params": chosen["k"], "from": float(chosen["y0"]), "to": float(chosen["y1"]),
               "rmse": float(chosen["rmse"]), "rmse_over_noise": float(chosen["rmse"] / floor) if floor > 0 else None,
               "params": {kk: (vv if not isinstance(vv, float) else float(vv)) for kk, vv in p.items()},
               "conf": round(conf, 2), "src": "derived",
               "candidates": sorted([{"model": c["model"], "k": c["k"], "rmse": float(c["rmse"])} for c in cands
                                     if c["model"] in ("linear", "spring", "bezier") or c is pre], key=lambda c: c["rmse"])[:5]}
        if sd is not None and chosen["model"] in ("linear",) + tuple(PRESETS):
            ent["params_sd"] = {"t_start": float(sd[0]), "duration": float(sd[1])}
        if chosen["model"] == "spring" and sd is not None:
            ent["params_sd"] = {"t_start": float(sd[0]), "omega0": float(sd[1]), "zeta": float(sd[2])}
        # how well the best named preset does against the free bezier: is the name meaningful?
        if pre is not None and bez is not None:
            ent["nearest_preset"] = {"name": pre["model"], "rmse": float(pre["rmse"]),
                                     "rmse_vs_free_bezier": float(pre["rmse"] / max(bez["rmse"], 1e-12)),
                                     "t_start": float(pre["params"]["t_start"]), "duration": float(pre["params"]["duration"])}
        if refv:
            ent["from_ref_units"] = ent["from"] / refv
            ent["to_ref_units"] = ent["to"] / refv
        segs.append(ent)
    if not segs:
        segs_line = "no motion intervals (signal is a hold)"
    else:
        segs_line = "; ".join(f"{s['t_detect'][0]:.2f}s {s['model']}"
                              + (f"({s['nearest_preset']['name']})" if s["model"] == "bezier" and s.get("nearest_preset") else "")
                              for s in segs)
    out = {"track": tid, "channel": channel, "unit": unit, "note": note, "n_samples": int(len(t)),
           "noise_sigma": D(sigma, 0.6, unit, "robust, from second differences; can be 0 on perfectly static signals, hence absolute floors"),
           "min_transition": abs_floor,
           "segments": segs, "summary_line": f"{channel}: {segs_line}", "warnings": []}
    return store.put(("curves", key), out, VERSION, params, deps=[("tracks", tid)], t_start=t_start), False
