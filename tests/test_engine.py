"""Measured vs ground truth on synthetic clips. Prints a table and exits non-zero on failure.

python tests/test_engine.py [stage ...]      (stages: 1 2 3 4; default: all present)
Clips are generated into tests/clips if missing; cache goes to tests/.cache_test (wiped).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "tests" / "clips"
CACHE = ROOT / "tests" / ".cache_test"
os.environ["VE_CACHE"] = str(CACHE)
sys.path.insert(0, str(ROOT))

from engine import probe, scenes, sheets  # noqa: E402
from engine.store import Store  # noqa: E402

RESULTS = []


def check(name, measured, truth, ok, note=""):
    RESULTS.append((name, measured, truth, ok, note))


def truth(name):
    return json.loads((CLIPS / f"{name}.truth.json").read_text())


def analyze(clip, layers=0):
    st = Store(CLIPS / clip)
    probe.run(st)
    scenes.run(st)
    if layers >= 1:
        from engine import audio, motion
        motion.run(st)
        audio.run(st)
    return st


# ------------------------------------------------------------------ stage 1

def stage1():
    tr = truth("scenes")
    st = analyze("scenes.mp4")
    v = st.get("probe")["video"]
    check("scenes: frames", v["frames"]["v"], tr["frames"], v["frames"]["v"] == tr["frames"])
    check("scenes: duration s", v["duration"]["v"], tr["duration"], abs(v["duration"]["v"] - tr["duration"]) < 1e-6)
    ev = st.get("scenes")["events"]
    cuts = [e["t"] for e in ev if e["type"] == "cut"]
    dt = 1 / tr["fps"]
    err = max((min(abs(c - x) for c in cuts) for x in tr["cuts"]), default=9) if cuts else 9
    check("scenes: cuts", [round(c, 3) for c in cuts], [round(c, 3) for c in tr["cuts"]],
          len(cuts) == len(tr["cuts"]) and err <= dt / 2, f"max err {err * 1000:.1f} ms")
    fl = [e["t"] for e in ev if e["type"] == "flash"]
    check("scenes: flash (not a cut)", fl, tr["flash"], len(fl) == 1 and abs(fl[0] - tr["flash"][0]) < dt / 2)
    fo = [e for e in ev if e["type"] == "fade_out"]
    bl = [e for e in ev if e["type"] == "black"]
    ok = bool(fo) and abs(fo[0]["t"] - tr["fade_out"]["t0"]) <= dt * 1.01 and bl and \
        abs(bl[0]["t"] - tr["fade_out"]["black_from"]) <= dt * 1.01
    check("scenes: fade_out start / black start", [fo[0]["t"] if fo else None, bl[0]["t"] if bl else None],
          [tr["fade_out"]["t0"], tr["fade_out"]["black_from"]], ok, "tolerance 1 frame")
    pth, times = sheets.contact_sheet(st)
    check("scenes: contact sheet", f"{len(times)} tiles", "file exists, every scene present",
          pth.exists() and all(any(s["t0"] <= x < s["t1"] for x in times) for s in st.get("scenes")["scenes"]))

    t0 = time.time()
    st2 = Store(CLIPS / "scenes.mp4")
    _, c1 = probe.run(st2)
    _, c2 = scenes.run(st2)
    el = time.time() - t0
    check("cache: second run recomputes nothing", f"cached={c1 and c2}, {el:.2f}s", "cached=True", c1 and c2 and el < 1.0)

    # variable frame rate: decoded PTS must equal the times the frames were rendered at
    import numpy as np
    bt = truth("ball")
    st = analyze("ball_vfr.mp4")
    pts = probe.pts(st)
    vt = np.array(bt["vfr_times"])
    e = float(np.abs(pts - vt).max() * 1000) if len(pts) == len(vt) else None
    check("vfr: pts vs render times", f"max err {e} ms" if e is not None else f"{len(pts)} vs {len(vt)} frames",
          "0 ms", e is not None and e < 0.5)
    check("vfr: flagged", st.get("probe")["video"]["vfr"]["v"], True, st.get("probe")["video"]["vfr"]["v"] is True)

    # any container/codec: summary + sheet without manual steps
    expect = {"fmt_vp9.webm": 3, "fmt_mjpeg.avi": 3, "fmt_anim.gif": 3, "fmt_rot90.mp4": 3,
              "fmt_offset.mkv": 3, "fmt_tiny.mp4": 0, "fmt_oddsize.mp4": 3}
    for f, ncut in expect.items():
        if not (CLIPS / f).exists():
            check(f"format {f}", "missing", "clip", False)
            continue
        try:
            st = analyze(f)
            cuts = [x["t"] for x in st.get("scenes")["events"] if x["type"] == "cut"]
            pth, _ = sheets.contact_sheet(st, n=8)
            tol = 0.07 if f.endswith(".gif") else 0.02
            good = len(cuts) == ncut and all(min(abs(c - x) for c in cuts) <= tol for x in tr["cuts"][:ncut])
            extra = ""
            if f == "fmt_rot90.mp4":
                v = st.get("probe")["video"]
                good &= (v["width"], v["height"]) == (360, 640)
                extra = f"{v['width']}x{v['height']}"
            check(f"format {f}", f"{len(cuts)} cuts {extra}", f"{ncut} cuts", good and pth.exists())
        except Exception as ex:  # report, do not hide
            check(f"format {f}", f"ERROR {ex}", f"{ncut} cuts", False)


def stage2():
    from engine import audio, motion  # noqa: F401
    import tests.checks_stage2 as c2
    c2.run(analyze, truth, check)


def stage3():
    import tests.checks_stage3 as c3
    c3.run(analyze, truth, check)


def stage4():
    import tests.checks_stage4 as c4
    c4.run(analyze, truth, check)


def main():
    stages = sys.argv[1:] or ["1", "2", "3", "4"]
    if not (CLIPS / "scenes.mp4").exists():
        subprocess.run([sys.executable, str(ROOT / "tests" / "synth.py"), str(CLIPS)], check=True)
    shutil.rmtree(CACHE, ignore_errors=True)
    for s in stages:
        fn = globals()[f"stage{s}"]
        try:
            fn()
        except ModuleNotFoundError as ex:
            print(f"stage {s}: skipped ({ex})")
    w = max(len(r[0]) for r in RESULTS)
    fails = 0
    for name, m, t, ok, note in RESULTS:
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  measured={m}  truth={t}  {note}")
    print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
