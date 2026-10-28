"""Stage 3: track, curves, states vs ground truth (and negative controls)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402

H_BALL, H_EASE, H_SPR = 360.0, 360.0, 270.0


def run(analyze, truth, check):
    from engine import curves, states, track
    from engine.store import Store
    import tests.test_engine as te

    # ------------------------------------------------------------ ball: track + ballistic
    bt = truth("ball")
    p = bt["params"]
    box = [v / H_BALL for v in bt["start_box_px"]]
    for clip, tol_mean, tol_max in (("ball.mp4", 0.2, 0.5), ("ball_vfr.mp4", 0.2, 0.5), ("ball_noisy.mp4", 0.25, 0.7)):
        st = analyze(clip)
        out, _ = track.run(st, "1", box, 0.0, name="ball", ref_fh=2 * p["r_px"] / H_BALL)
        S = out["samples"]
        t = np.array(S["t"])
        pos = np.stack([np.array(S["x"]), np.array(S["y"])], 1) * H_BALL
        gt = np.array([synth.ball_state(tt, p)[:2] for tt in t])
        e = np.hypot(*(pos - gt).T)
        check(f"track {clip}: position error px", f"mean {e.mean():.3f} max {e.max():.3f}", f"<{tol_mean} / <{tol_max}",
              e.mean() < tol_mean and e.max() < tol_max and out["quality"]["lost"] == 0, f"{len(t)} frames, PTS times")
        c, _ = curves.run_ballistic(st, "1")
        g = c["g"]["v"]
        check(f"ballistic {clip}: g fh/s2", f"{g:.4f}", bt["g_fh"], abs(g - bt["g_fh"]) / bt["g_fh"] < 0.01,
              f"rel err {abs(g - bt['g_fh']) / bt['g_fh'] * 100:.2f}%")
        gr = c["g_ref_units"]["v"]
        check(f"ballistic {clip}: g in ball diameters/s2", f"{gr:.2f}", bt["g_ref"], abs(gr - bt["g_ref"]) / bt["g_ref"] < 0.01)
        imp = c["impacts"]
        es = [i["restitution"]["v"] for i in imp]
        check(f"ballistic {clip}: restitution", [round(v, 3) for v in es], bt["restitution"],
              len(es) >= 3 and all(abs(v - 0.7) < 0.02 for v in es))
        errs = [abs(i["t"] - bt["bounces"][k]) for k, i in enumerate(imp)]
        check(f"ballistic {clip}: impact times", f"{len(imp)} found, max err {max(errs) * 1000:.1f} ms" if errs else "none",
              f"3 of {len(bt['bounces'])}, <5 ms", len(imp) == 3 and max(errs) < 0.005,
              "4th bounce: last flight has < 6 samples, honestly not reported")
        mids = [(bt["bounces"][k] + bt["bounces"][k + 1]) / 2 for k in range(len(bt["bounces"]) - 1)]
        apex = [f["apex"]["t"] for f in c["flights"] if "apex" in f and f["t0"] > 0.1]
        ea = [min(abs(a - m) for m in mids) for a in apex]
        check(f"ballistic {clip}: apex times", [round(a, 3) for a in apex], [round(m, 3) for m in mids[:len(apex)]],
              len(apex) >= 2 and max(ea) < 0.006)
        vx = c["vx_mean_fh_s"]["v"]
        check(f"ballistic {clip}: horizontal speed fh/s", f"{vx:.4f}", f"{bt['vx_fh']:.4f}", abs(vx - bt["vx_fh"]) / bt["vx_fh"] < 0.01)
    _, again = curves.run_ballistic(analyze("ball.mp4"), "1")
    check("cache: ballistic second run", again, True, again is True)

    # ------------------------------------------------------------ ease clip: track + transitions
    et = truth("ease")
    st = analyze("ease.mp4")
    box = [v / H_EASE for v in et["start_box_px"]]
    out, _ = track.run(st, "1", box, 0.0, name="square", ref_fh=60 / H_EASE, center="centroid")
    S = out["samples"]
    x = np.array(S["x"]) * H_EASE
    gx = np.array([g["x"] for g in et["samples"]])[:len(x)]
    e = np.abs(x - gx)
    check("track ease (centroid): x error px", f"mean {e.mean():.3f} max {e.max():.3f}", "<0.15 / <0.5", e.mean() < 0.15 and e.max() < 0.5)
    # the default NCC centre must warn when a flat shape grows (known failure, must not be silent)
    o2, _ = track.run(st, "1n", box, 0.0, name="square-ncc", ref_fh=60 / H_EASE)
    check("track ease (ncc): known drift is flagged", [w[:40] for w in o2["warnings"]], "a warning", bool(o2["warnings"]),
          "NCC plateau on a growing flat square; the warning points to --center centroid")

    cx, _ = curves.run_curve(st, "1", "x")
    segs = cx["segments"]
    check("curves x: two transitions, no ripple segment", [s["model"] for s in segs], ["ease", "spring"],
          len(segs) == 2 and segs[0]["model"] == "ease" and segs[1]["model"] == "spring")
    a, b = segs[0], segs[1]
    pa = a["params"]
    check("curves x#1 ease: start/duration s", [round(pa["t_start"], 4), round(pa["duration"], 4)], [0.3, 1.0],
          abs(pa["t_start"] - 0.3) < 0.01 and abs(pa["duration"] - 1.0) < 0.02)
    check("curves x#1 ease: levels px", [round(a["from"] * H_EASE, 2), round(a["to"] * H_EASE, 2)], [100, 500],
          abs(a["from"] * H_EASE - 100) < 0.5 and abs(a["to"] * H_EASE - 500) < 0.5)
    pr = a.get("nearest_preset") or {}
    check("curves x#1: nearest preset is CSS 'ease'", [pr.get("name"), round(pr.get("rmse_vs_free_bezier", 9), 2)],
          ["ease", "<1.5x free bezier"], pr.get("name") == "ease" and pr.get("rmse_vs_free_bezier", 9) < 1.5)
    sp = et["segments"][2]["params"]
    pb = b["params"]
    check("curves x#2 spring: omega0 / zeta", [round(pb["omega0"], 3), round(pb["zeta"], 4)],
          [round(sp["omega0"], 3), round(sp["zeta"], 4)],
          abs(pb["omega0"] - sp["omega0"]) / sp["omega0"] < 0.02 and abs(pb["zeta"] - sp["zeta"]) / sp["zeta"] < 0.03)
    check("curves x#2 spring: stiffness / damping (m=1)", [round(pb["stiffness_per_mass"], 1), round(pb["damping_per_mass"], 2)],
          [sp["stiffness"], sp["damping"]],
          abs(pb["stiffness_per_mass"] - sp["stiffness"]) / sp["stiffness"] < 0.03 and abs(pb["damping_per_mass"] - sp["damping"]) / sp["damping"] < 0.03)
    check("curves x#2 spring: start s, levels px", [round(pb["t_start"], 3), round(b["from"] * H_EASE, 1), round(b["to"] * H_EASE, 1)],
          [2.6, 500, 200], abs(pb["t_start"] - 2.6) < 0.01 and abs(b["from"] * H_EASE - 500) < 1.5 and abs(b["to"] * H_EASE - 200) < 1.5)

    cs, _ = curves.run_curve(st, "1", "scale")
    s0 = cs["segments"][0] if cs["segments"] else None
    npst = (s0 or {}).get("nearest_preset") or {}
    check("curves scale: one easeOutBack transition", [len(cs["segments"]), s0 and s0["model"]], [1, "easeOutBack"],
          len(cs["segments"]) == 1 and s0["model"] == "easeOutBack")
    if s0:
        check("curves scale: start s / duration s / levels", [round(s0["params"]["t_start"], 3), round(s0["params"]["duration"], 3), round(s0["from"], 3), round(s0["to"], 3)],
              [1.7, 0.6, 1.0, 1.5],
              abs(s0["params"]["t_start"] - 1.7) < 0.01 and abs(s0["params"]["duration"] - 0.6) < 0.03 and abs(s0["from"] - 1) < 0.01 and abs(s0["to"] - 1.5) < 0.01,
              "truth is the exact easeOutBack formula; CSS preset (.34,1.56,.64,1) only approximates it")
    co, _ = curves.run_curve(st, "1", "opacity")
    o0 = co["segments"][0] if co["segments"] else None
    check("curves opacity: linear fade", [len(co["segments"]), o0 and o0["model"]], [1, "linear"],
          len(co["segments"]) == 1 and o0["model"] == "linear")
    if o0:
        check("curves opacity: start s / duration s / end level", [round(o0["params"]["t_start"], 3), round(o0["params"]["duration"], 3), round(o0["to"], 3)],
              [3.8, 0.8, 0.2], abs(o0["params"]["t_start"] - 3.8) < 0.01 and abs(o0["params"]["duration"] - 0.8) < 0.02 and abs(o0["to"] - 0.2) < 0.01)
    _, again = curves.run_curve(st, "1", "x")
    check("cache: curves second run", again, True, again is True)

    # ------------------------------------------------------------ sprite: track + states
    spt = truth("sprite")
    st = analyze("sprite.mp4")
    box = [v / H_SPR for v in spt["start_box_px"]]
    for upd in (0.0, 0.3):
        out, _ = track.run(st, "2", box, 0.0, name="sprite", update=upd)
        xs = np.array(out["samples"]["x"]) * H_SPR
        gt = np.array(spt["x_center"])
        err = np.abs(xs - gt - np.median(xs - gt))
        check(f"track sprite (update {upd}): x error px", f"mean {err.mean():.3f} max {err.max():.3f}", "<0.2 / <0.5",
              err.mean() < 0.2 and err.max() < 0.5 and out["quality"]["lost"] == 0)
        s, _ = states.run(st, "s1", box, track_id="2")
        tl = s["timeline"]
        want = spt["timeline"]
        frames_ok = [e["frames"] for e in tl] == [e["frames"] for e in want]
        mapping, good = {}, True
        for e, w in zip(tl, want):
            good &= mapping.setdefault(e["pose"], w["pose"]) == w["pose"]
        good &= len(set(mapping.values())) == len(mapping)
        check(f"states sprite (update {upd}): intervals/frames exact", f"{len(tl)} intervals, {len(s['palette'])} poses",
              f"{len(want)} intervals, {len(spt['unique_poses'])} poses", frames_ok and len(s["palette"]) == 7)
        check(f"states sprite (update {upd}): pose identity consistent with truth", good, True, good)
        bt0 = [round(e["t0"], 4) for e in tl]
        wt0 = [round(w["t0"], 4) for w in want]
        check(f"states sprite (update {upd}): boundary times from PTS", f"max err {max(abs(a - b) for a, b in zip(bt0, wt0)) * 1000:.2f} ms",
              "0 ms", bt0 == wt0)
        cyc = s["cycles"]
        run_c = [c for c in cyc if len(c["pattern"]) == 4]
        idle_c = [c for c in cyc if len(c["pattern"]) == 2]
        check(f"states sprite (update {upd}): cycles", [(len(c["pattern"]), c["repeats"], c["hold_frames_median"]) for c in cyc],
              [(2, 2.0, 8.0), (4, 3.0, 4.0)],
              len(run_c) == 1 and run_c[0]["repeats"] == 3 and run_c[0]["hold_frames_median"] == 4
              and len(idle_c) == 1 and idle_c[0]["repeats"] == 2 and idle_c[0]["hold_frames_median"] == 8)
    # fixed window instead of a track (poses while the sprite stays put) must still not crash and must label the limitation
    s_fixed, _ = states.run(st, "s_fixed", [v / H_SPR for v in (56, 136, 48, 84)], t0=0.0, t1=1.0)
    check("states fixed window: idle pair found", [len(s_fixed["palette"]), len(s_fixed["timeline"])], [2, 4],
          len(s_fixed["palette"]) == 2 and len(s_fixed["timeline"]) == 4)
    _, again = states.run(st, "s1", box, track_id="2")
    check("cache: states second run", again, True, again is True)

    # ------------------------------------------------------------ negative controls
    st = analyze("static_noisy.mp4")
    out, _ = track.run(st, "9", [100 / 270, 60 / 270, 60 / 270, 60 / 270], 0.0, name="patch")
    c, _ = curves.run_curve(st, "9", "x")
    check("control static_noisy: tracked patch has no motion intervals", len(c["segments"]), 0, len(c["segments"]) == 0)
    s, _ = states.run(st, "s9", [100 / 270, 60 / 270, 60 / 270, 60 / 270])
    check("control static_noisy: one pose, one interval", [len(s["palette"]), len(s["timeline"])], [1, 1],
          len(s["palette"]) == 1 and len(s["timeline"]) == 1)
    st = analyze("scenes.mp4")
    out, _ = track.run(st, "7", [40 / 360, 150 / 360, 60 / 360, 60 / 360], 0.0, name="red-square")
    lost_after = int(np.sum(np.array(out["samples"]["lost"])[np.array(out["samples"]["t"]) > 1.6]))
    check("control scenes: object that is gone is flagged lost, not silently followed",
          f"{lost_after} lost frames after the cut, warnings={len(out['warnings'])}", ">=60 lost + warning",
          lost_after >= 60 and bool(out["warnings"]))
