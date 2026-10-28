"""Stage 2: motion and audio vs ground truth."""
import numpy as np

H = 360.0


def run(analyze, truth, check):
    tr = truth("scenes")
    st = analyze("scenes.mp4", layers=1)
    mo = st.get("motion")

    # ---- camera
    cams = mo["camera_segments"]
    v = cams[0]["image_velocity_fh_s"]["v"] if cams else [None, None]
    want = tr["camera_pan"]["content_dx_fh_per_s"]
    err = abs(v[0] - want) / abs(want) if cams else 9
    check("motion: pan velocity fh/s", f"{v[0]:.4f}" if cams else None, f"{want:.4f}",
          len(cams) == 1 and err < 0.05 and abs(v[1]) < 0.01, f"rel err {err * 100:.2f}%, {len(cams)} camera segment(s)")
    check("motion: pan span", [round(cams[0]["t0"], 3), round(cams[0]["t1"], 3)] if cams else None,
          [3.0, 4.6], bool(cams) and abs(cams[0]["t0"] - 3.0) < 0.05 and abs(cams[0]["t1"] - 4.6) < 0.07)
    ps = {p["scene"]: p for p in mo["per_scene"]}
    check("motion: flat scenes labelled camera=assumed", [ps[0]["camera_src"], ps[3]["camera_src"]],
          ["assumed", "assumed"], ps[0]["camera_src"] == "assumed" and ps[3]["camera_src"] == "assumed",
          "honest: no texture to measure camera on")

    # ---- object segments per scene
    segs = {s["scene"]: s for s in mo["object_segments"]}
    s0 = segs.get(0)
    bx = s0["bbox_fh"] if s0 else None
    check("motion S0: square y-band", [round(bx[1], 3), round(bx[1] + bx[3], 3)] if bx else None,
          [round(150 / H, 3), round(210 / H, 3)],
          bool(bx) and abs(bx[1] - 150 / H) < 0.03 and abs(bx[1] + bx[3] - 210 / H) < 0.03)
    check("motion S0: x extent", [round(bx[0], 3), round(bx[0] + bx[2], 3)] if bx else None,
          [round(40 / H, 3), round(232 / H, 3)],
          bool(bx) and abs(bx[0] - 40 / H) < 0.04 and abs(bx[0] + bx[2] - 232 / H) < 0.05)
    mb = (s0 or {}).get("main_blob")
    vx = mb["mean_velocity_fh_s"]["v"][0] if mb else None
    check("motion S0: speed fh/s", None if vx is None else round(vx, 3), 0.25,
          vx is not None and abs(vx - 0.25) / 0.25 < 0.15, "blob-centroid slope; tolerance 15%")
    s1 = segs.get(1)
    b1 = s1["bbox_fh"] if s1 else None
    cx = (b1[0] + b1[2] / 2) if b1 else None
    check("motion S1: dot column", None if cx is None else [round(cx, 3), round(b1[2], 3)],
          [round(320 / H, 3), "<0.12 wide"], b1 is not None and abs(cx - 320 / H) < 0.02 and b1[2] < 0.12)
    check("motion S1: dot y range", [round(b1[1], 3), round(b1[1] + b1[3], 3)] if b1 else None,
          [round(66 / H, 3), round(214 / H, 3)],
          b1 is not None and abs(b1[1] - 66 / H) < 0.04 and abs(b1[1] + b1[3] - 214 / H) < 0.04)
    s3 = segs.get(3)
    b3 = s3["bbox_fh"] if s3 else None
    c3 = (b3[0] + b3[2] / 2, b3[1] + b3[3] / 2) if b3 else None
    check("motion S3: disc centre", [round(c3[0], 3), round(c3[1], 3)] if c3 else None, [round(320 / H, 3), 0.5],
          c3 is not None and abs(c3[0] - 320 / H) < 0.03 and abs(c3[1] - 0.5) < 0.03)
    check("motion S2: pan is not reported as object motion", segs.get(2), None, 2 not in segs)
    ov = [o for o in mo["overlays"] if o["scene"] == 2]
    hud = (20 / H, 20 / H, 120 / H, 60 / H)
    if ov:
        x, y, w, h = ov[0]["bbox_fh"]
        inside = x <= hud[0] + 0.01 and y <= hud[1] + 0.01 and x + w >= hud[2] - 0.01 and y + h >= hud[3] - 0.01
        small = w * h < 3 * (hud[2] - hud[0]) * (hud[3] - hud[1])
    else:
        inside = small = False
    check("motion S2: HUD found as screen-fixed overlay", ov[0]["bbox_fh"] if ov else None,
          [round(hud[0], 3), round(hud[1], 3), round(hud[2] - hud[0], 3), round(hud[3] - hud[1], 3)],
          inside and small, "must contain the HUD box, area < 3x of it")
    tail = [s for s in mo["object_segments"] if s["t0"] >= 5.9]
    check("motion: nothing moves in the black tail", len(tail), 0, not tail)

    # ---- ball: moving ball found, static decoration ignored, camera static
    st = analyze("ball.mp4", layers=1)
    mo = st.get("motion")
    segs = mo["object_segments"]
    ys = (segs[0]["bbox_fh"][1], segs[0]["bbox_fh"][1] + segs[0]["bbox_fh"][3]) if segs else (None, None)
    check("motion ball: one segment spanning the clip",
          [len(segs), round(segs[0]["t0"], 2), round(segs[0]["t1"], 2)] if segs else None,
          [1, 0.0, 2.98], len(segs) == 1 and segs[0]["t0"] < 0.1 and segs[0]["t1"] > 2.9)
    check("motion ball: no camera motion", len(mo["camera_segments"]), 0, not mo["camera_segments"])
    check("motion ball: swept y-range", None if ys[0] is None else [round(ys[0], 3), round(ys[1], 3)],
          [round(68 / H, 3), round(320 / H, 3)],
          ys[0] is not None and abs(ys[0] - 68 / H) < 0.05 and abs(ys[1] - 320 / H) < 0.03)
    st = analyze("ball_noisy.mp4", layers=1)
    mo = st.get("motion")
    n, nc = len(mo["object_segments"]), len(mo["camera_segments"])
    check("motion ball_noisy (sigma 6, crf 32): 1 segment, no camera", [n, nc], [1, 0], n == 1 and nc == 0)
    st = analyze("static_noisy.mp4", layers=1)
    mo = st.get("motion")
    got = [len(mo["object_segments"]), len(mo["camera_segments"]), len(mo["overlays"])]
    check("motion static_noisy: zero false segments", got, [0, 0, 0], got == [0, 0, 0])

    # ---- audio
    st = analyze("scenes.mp4", layers=1)
    au = st.get("audio")
    prim = [o for o in au["onsets"] if "secondary_of" not in o]
    clicks, thumps = tr["audio"]["clicks"], tr["audio"]["thumps"]
    mid = [o["t"] for o in prim if o["band"] == "mid"]
    low = [o["t"] for o in prim if o["band"] == "low"]

    def worst(found, want):
        if len(found) != len(want):
            return None
        return max(abs(found[i] - want[i]) for i in range(len(want)))
    e = worst(mid, clicks)
    check("audio: 12 clicks, band mid, timing",
          f"{len(mid)} found, max err {e * 1000:.1f} ms" if e is not None else f"{len(mid)} found",
          "12, <12 ms", e is not None and e < 0.012)
    e = worst(low, thumps)
    check("audio: 2 thumps, band low, timing",
          f"{len(low)} found, max err {e * 1000:.1f} ms" if e is not None else f"{len(low)} found",
          "2, <12 ms", e is not None and e < 0.012)
    bpm = au["tempo"]["bpm"]["v"] if au["tempo"] else None
    check("audio: tempo bpm", None if bpm is None else round(bpm, 2), tr["audio"]["tempo_bpm"],
          bpm is not None and abs(bpm - 120) < 1.5)
    check("audio: decay ripples not counted as hits", len(prim), 14, len(prim) == 14)
    for clip, tol, why in (("fmt_vp9.webm", 0.025, "opus codec delay shows up here"),
                           ("fmt_offset.mkv", 0.015, "container start offset 1.5 s must not shift onsets")):
        st = analyze(clip, layers=1)
        a2 = [o["t"] for o in st.get("audio")["onsets"] if "secondary_of" not in o and o["band"] == "mid"]
        e = worst(a2, clicks)
        check(f"audio: {clip} click timing", None if e is None else f"{e * 1000:.1f} ms", f"<{tol * 1000:.0f} ms",
              e is not None and e < tol, why)
    st = analyze("fmt_mjpeg.avi", layers=1)
    check("audio: no audio stream handled", st.get("audio")["present"], False, st.get("audio")["present"] is False)
