"""Stage 4: spec, render, export, compare, verify loop."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "tests" / "clips"
WORK = Path(tempfile.gettempdir()) / "ve_stage4"


def node_states(js_path, ts):
    code = f"const m=require({json.dumps(str(js_path))});const ts={json.dumps(ts)};console.log(JSON.stringify(ts.map(t=>m.state(t))));"
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True)
    if r.returncode:
        return None, r.stderr[:200]
    return json.loads(r.stdout), ""


def clone(x):
    return json.loads(json.dumps(x))


def run(analyze, truth, check):
    from engine import compare, curves, export, model, states, track, verify
    from engine import spec as S
    from engine.store import Store
    WORK.mkdir(parents=True, exist_ok=True)
    bt, et = truth("ball"), truth("ease")
    ball_box = [v / 360 for v in bt["start_box_px"]]
    ease_box = [v / 360 for v in et["start_box_px"]]

    # ---------------------------------------------------- analyses of the references
    st_ball = analyze("ball.mp4")
    track.run(st_ball, "1", ball_box, 0.0, name="ball", ref_fh=24 / 360)
    curves.run_ballistic(st_ball, "1")
    st_vfr = analyze("ball_vfr.mp4")
    track.run(st_vfr, "1", ball_box, 0.0, name="ball", ref_fh=24 / 360)
    curves.run_ballistic(st_vfr, "1")
    st_ease = analyze("ease.mp4")
    track.run(st_ease, "1", ease_box, 0.0, name="square", ref_fh=60 / 360, center="centroid")
    for ch in ("x", "scale", "opacity"):
        curves.run_curve(st_ease, "1", ch)

    # ---------------------------------------------------- spec content vs the generator's truth
    sp = S.build(st_ball, "1")
    o = sp["objects"][0]
    col = np.array(o["color"])
    check("spec ball: shape / colour / size px", [o["shape"], [round(v) for v in col], round(o["size_fh"][0] * 360, 1)],
          ["disc", [60, 200, 250], 24],
          o["shape"] == "disc" and np.abs(col - [60, 200, 250]).max() <= 6 and abs(o["size_fh"][0] * 360 - 24) <= 1.5)
    check("spec ball: background colour", [round(v) for v in sp["bg"]], [60, 45, 35], np.abs(np.array(sp["bg"]) - [60, 45, 35]).max() <= 6)
    check("spec ball: ground top fh", round(sp["ground"]["top_fh"], 4) if sp["ground"] else None, round(320 / 360, 4),
          bool(sp["ground"]) and abs(sp["ground"]["top_fh"] - 320 / 360) < 2 / 360)
    spe = S.build(st_ease, "1")
    oe = spe["objects"][0]
    types = [c["type"] for c in oe["channels"].values()]
    check("spec ease: rect, side px, channel types", [oe["shape"], round(oe["size_fh"][0] * 360, 1), types],
          ["rect", 60, ["segments", "const", "segments", "segments"]],
          oe["shape"] == "rect" and abs(oe["size_fh"][0] * 360 - 60) <= 1.5 and types == ["segments", "const", "segments", "segments"])
    st_spr = analyze("sprite.mp4")
    sbox = [v / 270 for v in truth("sprite")["start_box_px"]]
    track.run(st_spr, "2", sbox, 0.0, name="sprite", update=0.3)
    sps = S.build(st_spr, "2")
    check("spec sprite: textured object refused for rendering (not faked)", [sps["objects"][0]["renderable"], bool(sps["warnings"])],
          [False, True], sps["objects"][0]["renderable"] is False and bool(sps["warnings"]), "engine does not copy graphics")

    # ---------------------------------------------------- exporters
    ts = np.sort(np.random.default_rng(5).uniform(0, sp["duration"], 300)).tolist()
    for name, spec_ in (("ball", sp), ("ease", spe)):
        files = export.export_js(spec_, WORK / name, name)
        out, err = node_states(files[0], ts)
        ev = model.eval_object(spec_["objects"][0], np.array(ts))
        d = max(float(np.max(np.abs(np.array([q[k] for q in out]) - ev[k]))) for k in ev) if out else None
        check(f"export js {name}: node vs python model", f"max diff {d:.2e}" if d is not None else err, "<1e-9",
              d is not None and d < 1e-9, "300 random times; the same .js is what the viewer .html loads")
        export.export_css(spec_, WORK / name / f"{name}.css")
        tr = export.css_tracks(spec_, spec_["objects"][0])
        worst = {}
        for prop, dd in tr.items():
            if "kf" in dd:
                v = np.array([export.css_eval(dd["kf"], t, spec_["duration"]) for t in ts])
                worst[prop] = float(np.max(np.abs(v - ev[prop])) / max(float(np.ptp(ev[prop])), 1e-9))
        if name == "ball":
            check("export css ball: ballistic flights exact (cubic-bezier split at apex)", f"max {worst['y']:.1e} of range", "<1e-5",
                  worst["y"] < 1e-5, "CSS timing semantics evaluated in python, not in a browser")
        else:
            check("export css ease: bezier/linear exact, spring via linear()", {k: f"{v:.1e}" for k, v in worst.items()},
                  "x <0.5% (spring approx), scale/opacity <1e-4 of range",
                  worst["x"] < 0.005 and worst["scale"] < 1e-4 and worst["opacity"] < 1e-4, "spring error is stated in the .css header")
    gd = export.export_godot(sp, WORK / "ball" / "ball.gd")
    txt = gd.read_text(encoding="utf-8")
    check("export godot: file generated (NOT executed: no Godot here)", "func state(" in txt and "NOT executed" in txt, True,
          "func state(" in txt and "NOT executed" in txt, "untested in the target engine")

    # ---------------------------------------------------- faithful round trips
    for label, st, spec_, thr in (("ball", st_ball, sp, 0.002), ("ball VFR", st_vfr, S.build(st_vfr, "1"), 0.002),
                                  ("ease", st_ease, spe, 0.003)):
        r = verify.verify(st, spec_, WORK / label.replace(" ", "_"), thr={"track_p95_fh": thr}, tag="faithful")
        rep = r["report"]
        m = rep["tracks"]["1"]
        check(f"verify {label}: faithful spec reproduces the clip",
              f"p95 {m['p95_fh'] * 360:.2f} px, ssim {rep['ssim']['mean']:.4f}, params out {len(rep['params_out_of_tolerance'])}",
              f"p95 < {thr * 360:.2f} px, verdict PASS", rep["verdict"]["pass"], "ssim also sees unmodelled decoration; not in the verdict")

    # ---------------------------------------------------- the loop from a wrong spec
    s0 = clone(sp)
    c = s0["objects"][0]["channels"]
    c["y"]["g"] *= 0.8
    c["y"]["restitution"] = 0.5
    c["x"]["vx"] *= 1.2
    c["x"]["x0"] += 10 / 360
    hist, cur, names0, top = [], s0, None, None
    for it in range(5):
        r = verify.verify(st_ball, cur, WORK / "loop_ball", thr={"track_p95_fh": 0.002}, tag=f"loop{it}", apply=True, ssim=False)
        rep = r["report"]
        hist.append(rep["tracks"]["1"]["p95_fh"])
        if it == 0:
            names0 = rep["params_out_of_tolerance"]
            top = [p["name"] for p in rep["params_ranked"][:4]]
        if rep["verdict"]["pass"] or r["blocked"]:
            break
        cur = r["spec_next"]
    check("loop ball: converges below the agreed threshold", [round(h * 360, 2) for h in hist],
          "px p95: decreasing, last < 0.72", rep["verdict"]["pass"] and all(b < a for a, b in zip(hist, hist[1:])) and len(hist) <= 4,
          f"{len(hist) - 1} corrections")
    check("loop ball: first report names the wrong parameters", sorted(n.split(".")[-1] for n in names0 if n.count(".") == 1),
          "g, restitution, vx among them", {"ballistic_1.g", "ballistic_1.restitution", "ballistic_1.vx"} <= set(names0))
    real = [t for t in top if not t.endswith("flights")]
    check("loop ball: ranking leads with numeric parameter errors", top, "top entries are real parameters",
          all("impact" in t or t.split(".")[-1] in ("g", "vx", "restitution", "y0", "v0") for t in real))
    rr = verify.verify(st_ball, s0, WORK / "loop_ball", thr={"track_p95_fh": 0.002}, tag="rest_case", ssim=False)
    bc = rr["test_store"].get("curves", "ballistic_1")
    g_in = s0["objects"][0]["channels"]["y"]["g"]
    check("analyser recovers a known g on a clip where the ball comes to rest", f"{bc['g']['v']:.4f} (rests: {len(bc['rests'])})",
          f"{g_in:.4f} +-3%, >=1 rest", abs(bc["g"]["v"] - g_in) / g_in < 0.03 and len(bc["rests"]) >= 1,
          "regression: rest used to corrupt g (0.73 instead of 2.0)")

    se = clone(spe)
    c = se["objects"][0]["channels"]
    c["x"]["segments"][1]["omega0"] *= 1.25
    c["x"]["segments"][0]["t_start"] += 0.12
    c["opacity"]["segments"][0]["to"] = 0.5
    hist, cur = [], se
    for it in range(6):
        r = verify.verify(st_ease, cur, WORK / "loop_ease", thr={"track_p95_fh": 0.003}, tag=f"e{it}", apply=True, ssim=False)
        rep = r["report"]
        hist.append((rep["tracks"]["1"]["p95_fh"], len(rep["params_out_of_tolerance"])))
        if rep["verdict"]["pass"] or r["blocked"]:
            break
        cur = r["spec_next"]
    check("loop ease: converges (position AND fitted params)", [(round(a * 360, 2), n) for a, n in hist],
          "px p95, params out -> 0", rep["verdict"]["pass"] and hist[-1][1] == 0 and len(hist) <= 5,
          f"{len(hist) - 1} corrections; blocked={r['blocked']}")

    # ---------------------------------------------------- compare: negative control and robustness
    wrong = clone(sp)
    wrong["objects"][0]["channels"]["y"]["restitution"] = 0.5
    r = verify.verify(st_ball, wrong, WORK / "neg", thr={"track_p95_fh": 0.002}, tag="wrong_e", ssim=False)
    rep = r["report"]
    top3 = [p["name"] for p in rep["params_ranked"][:3]]
    check("compare: one wrong parameter (restitution) is caught and named", [rep["verdict"]["pass"], top3],
          [False, "restitution/impact times lead"],
          (not rep["verdict"]["pass"]) and any("restitution" in n or "impact" in n for n in top3[:2]))

    lowq = WORK / "sprite_lowq.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(CLIPS / "sprite.mp4"), "-c:v", "libx264", "-crf", "38",
                    "-vf", "scale=iw/2:ih/2,scale=iw*2:ih*2", "-pix_fmt", "yuv420p", str(lowq)], check=True)
    states.run(st_spr, "s1", sbox, track_id="2")
    rep = compare.compare(st_spr, Store(lowq), thr={"track_p95_fh": 0.004}, label="lowq", ssim=True)
    ss = rep["states"].get("s1", {})
    check("compare sprite vs crf38 + half-res re-encode: same pose sequence",
          [ss.get("sequence_equal"), round(ss.get("max_boundary_diff_ms", -1), 2)], [True, 0.0],
          ss.get("sequence_equal") is True and ss.get("max_boundary_diff_ms", 99) < 1.0,
          f"track p95 {rep['tracks']['2']['p95_fh'] * 270:.2f} px, ssim {rep['ssim']['mean']:.3f}")
    check("compare sprite vs re-encode: verdict", rep["verdict"]["pass"], True, rep["verdict"]["pass"],
          "threshold 0.004 fh set by the test, not by the user")
