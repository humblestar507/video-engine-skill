"""CLI commands for layers 2-3 (track, states, curves, later export/compare)."""
import sys

from pathlib import Path

from . import probe, scenes
from .store import Store


def _box(st, s, units):
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise SystemExit("--box needs x,y,w,h")
    pv = st.get("probe")["video"]
    W, H, sar = pv["width"], pv["height"], pv.get("sar", 1.0)
    if units == "px":
        return [v[0] * sar / H, v[1] / H, v[2] * sar / H, v[3] / H]
    if units == "frac":
        return [v[0] * W * sar / H, v[1], v[2] * W * sar / H, v[3]]
    return v


def _prep(st):
    probe.run(st)
    scenes.run(st)


def _ref_fh(st, a):
    H = st.get("probe")["video"]["height"]
    if getattr(a, "ref_px", None):
        return a.ref_px / H
    return getattr(a, "ref_fh", None)


def cmd_track(st, a):
    from . import track
    _prep(st)
    out, cached = track.run(st, a.id, _box(st, a.box, a.units), a.t_init, name=a.name, ref_fh=_ref_fh(st, a),
                            t0=a.t0, t1=a.t1, update=a.update, do_measure=not a.no_measure, center=a.center, force=a.force)
    q = out["quality"]
    print(f"track {a.id} '{out['name']}' {out['t0']:.3f}-{out['t1']:.3f}s {q['frames']} frames, lost {q['lost']}, "
          f"mean score {q['mean_score']:.3f}, ref {out['ref']['src']} ({out['ref']['fh']:.4f} fh)" + (" (cached)" if cached else ""))
    if q["jitter_fh"]:
        print(f"jitter <= {q['jitter_fh']['v']:.5f} fh")
    for w in out["warnings"]:
        print("warning:", w)
    print(f"query: ve query VIDEO tracks.{a.id}.samples --t0 .. --t1 ..")


def cmd_states(st, a):
    from . import states
    _prep(st)
    out, cached = states.run(st, a.id, _box(st, a.box, a.units), track_id=a.track, t_init=a.t_init, t0=a.t0, t1=a.t1,
                             thr=a.thr, force=a.force)
    print(f"states {a.id}: {out['summary_line']}" + (" (cached)" if cached else ""))
    print(f"palette image: {out['palette_image']}")
    print(f"threshold {out['threshold']['used']:.2e} ({out['threshold']['src']}); query: ve query VIDEO states.{a.id}.timeline")
    for w in out["warnings"]:
        print("warning:", w)


def cmd_curves(st, a):
    from . import curves
    _prep(st)
    if a.ballistic:
        out, cached = curves.run_ballistic(st, a.track, axis=a.axis, force=a.force)
        print(out["summary_line"] + (" (cached)" if cached else ""))
        for im in out["impacts"]:
            print(f"  impact t={im['t']:.4f}s restitution {im['restitution']['v']:.3f}")
        for f in out["flights"]:
            if "apex" in f:
                print(f"  flight {f['t0']:.3f}-{f['t1']:.3f}s apex t={f['apex']['t']:.3f}s")
        print(f"query: ve query VIDEO curves.ballistic_{a.track}")
    else:
        out, cached = curves.run_curve(st, a.track, a.channel, force=a.force)
        print(out["summary_line"] + (" (cached)" if cached else ""))
        for s in out["segments"]:
            pr = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s["params"].items()}
            print(f"  {s['t_detect'][0]:.3f}-{s['t_detect'][1]:.3f}s {s['model']} {s['from']:.4g}->{s['to']:.4g} "
                  f"rmse {s['rmse']:.2g} conf {s['conf']} {pr}")
            if s.get("nearest_preset"):
                n = s["nearest_preset"]
                print(f"    nearest preset: {n['name']} (rmse x{n['rmse_vs_free_bezier']:.2f} of the free bezier)")
        print(f"query: ve query VIDEO curves.{a.track}_{a.channel}.segments")
    for w in out["warnings"]:
        print("warning:", w)


def reg_track(sub):
    a = sub.add_parser("track", help="layer 2: follow one object (user-confirmed box on the first frame)")
    a.add_argument("video")
    a.add_argument("--id", required=True)
    a.add_argument("--name", default="")
    a.add_argument("--box", required=True, help="x,y,w,h of the object on --t-init")
    a.add_argument("--units", default="px", choices=["px", "fh", "frac"], help="units of --box (default px)")
    a.add_argument("--t-init", type=float, default=0.0)
    a.add_argument("--t0", type=float)
    a.add_argument("--t1", type=float)
    g = a.add_mutually_exclusive_group()
    g.add_argument("--ref-px", type=float, help="size of the reference object in px (e.g. character height)")
    g.add_argument("--ref-fh", type=float)
    a.add_argument("--update", type=float, default=0.0, help="template blend 0..1 for objects that change pose")
    a.add_argument("--center", default="ncc", choices=["ncc", "centroid"], help="centroid for flat shapes that change size")
    a.add_argument("--no-measure", action="store_true")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_track)


def reg_states(sub):
    a = sub.add_parser("states", help="layer 2: pose/look timeline of an object")
    a.add_argument("video")
    a.add_argument("--id", required=True)
    a.add_argument("--box", required=True, help="x,y,w,h window around the object at --t-init")
    a.add_argument("--units", default="px", choices=["px", "fh", "frac"])
    a.add_argument("--track", help="follow this track id (otherwise the box is fixed)")
    a.add_argument("--t-init", type=float, default=0.0)
    a.add_argument("--t0", type=float)
    a.add_argument("--t1", type=float)
    a.add_argument("--thr", type=float, help="override change threshold (normalised MSE)")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_states)


def reg_curves(sub):
    a = sub.add_parser("curves", help="layer 2: fit physics/easing to a track")
    a.add_argument("video")
    a.add_argument("--track", required=True)
    a.add_argument("--channel", default="x", choices=["x", "y", "size", "scale", "opacity"])
    a.add_argument("--ballistic", action="store_true", help="free flights, shared g, impacts, restitution")
    a.add_argument("--axis", default="y", choices=["x", "y"])
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_curves)


def _thr(a):
    return {"track_p95_fh": a.thr_track_fh} if a.thr_track_fh is not None else None


def cmd_spec(st, a):
    from . import spec
    _prep(st)
    sp = spec.build(st, a.track)
    spec.save(sp, a.out)
    o = sp["objects"][0]
    print(f"spec {a.out}: object {o['id']} {o['shape']} renderable={o['renderable']} "
          + ", ".join(f"{k}={c['type']}[{c['src']}]" for k, c in o["channels"].items()))
    for w in sp["warnings"]:
        print("warning:", w)


def cmd_export(st, a):
    from . import export, spec
    sp = spec.load(a.spec)
    out = Path(a.out)
    name = a.name or Path(a.spec).stem
    made = []
    if a.target in ("js", "all"):
        made += export.export_js(sp, out, name)
    if a.target in ("css", "all"):
        out.mkdir(parents=True, exist_ok=True)
        made.append(export.export_css(sp, out / f"{name}.css"))
    if a.target in ("godot", "all"):
        out.mkdir(parents=True, exist_ok=True)
        made.append(export.export_godot(sp, out / f"{name}.gd"))
    for m in made:
        print(m)
    if a.target in ("godot", "all"):
        print("note: GDScript export is not executed in Godot by the engine (no Godot available)")


def cmd_render(st, a):
    from . import probe as pr, render, spec
    sp = spec.load(a.spec)
    times = None
    if a.times_from:
        ref = Store(a.times_from)
        pr.run(ref)
        times = pr.pts(ref)
    print(render.render(sp, a.out, times=times))


def _print_compare(rep):
    print(rep["summary_line"])
    for tid, m in rep["tracks"].items():
        print(f"  track {tid}: rms {m['rms_fh']:.5f} p95 {m['p95_fh']:.5f} max {m['max_fh']:.5f} fh | in ref units "
              f"({m['ref_units']}): rms {m['rms_ref']:.4f} | bias {m['bias_fh'][0]:+.5f},{m['bias_fh'][1]:+.5f} | lag {m['test_lag_ms']:+.0f} ms"
              + ("" if m["lag_reliable"] else " (unreliable)"))
        for w in m["worst_windows"][:2]:
            print(f"    worst window {w['t0']:.3f}-{w['t1']:.3f}s max {w['max_fh']:.5f} fh")
    bad = [p for p in rep["params_ranked"] if p["z"] > 1 and p["z"] != float("inf")]
    for p in bad[:6]:
        print(f"  param {p['name']}: ref {p['ref']:.5g} test {p['test']:.5g} (z={p['z']:.1f})")
    struct = [p["name"] for p in rep["params_ranked"] if p["z"] == float("inf")]
    if struct:
        print("  structural differences:", ", ".join(struct[:8]))
    if rep["ssim"]:
        print(f"  ssim mean {rep['ssim']['mean']:.4f} min {rep['ssim']['min']:.4f} at {rep['ssim']['min_at_s']:.2f}s (informative)")
    v = rep["verdict"]
    print("  verdict:", "PASS" if v["pass"] else "FAIL", "-", v["note"])
    for n in rep["notes"]:
        print("  note:", n)


def cmd_compare(st, a):
    from . import compare
    _prep(st)
    test = Store(a.test)
    _print_compare(compare.compare(st, test, thr=_thr(a), ssim=not a.no_ssim, tracks=a.tracks.split(",") if a.tracks else None))


def cmd_verify(st, a):
    from . import spec, verify
    _prep(st)
    sp = spec.load(a.spec)
    r = verify.verify(st, sp, a.work_dir, thr=_thr(a), tag=a.tag, apply=a.apply, ssim=not a.no_ssim)
    print("reproduction:", r["video"])
    _print_compare(r["report"])
    if a.apply:
        for c in r["changes"]:
            print(f"  update {c['param']}: {c['old']:.5g} -> {c['new']:.5g}")
        for b in r["blocked"]:
            print("  BLOCKED:", b)
        if r["spec_next"] is not None and not r["blocked"]:
            nxt = a.next_spec or a.spec
            spec.save(r["spec_next"], nxt)
            print("next spec written to", nxt)


def reg_spec(sub):
    a = sub.add_parser("spec", help="layer 3: scene spec (renderable parameters) from a track and its fits")
    a.add_argument("video")
    a.add_argument("--track", required=True)
    a.add_argument("--out", default="spec.json")
    a.set_defaults(func=cmd_spec)


def reg_export(sub):
    a = sub.add_parser("export", help="layer 3: spec -> js | css | godot")
    a.add_argument("spec")
    a.add_argument("--target", default="all", choices=["js", "css", "godot", "all"])
    a.add_argument("--out", default="export")
    a.add_argument("--name")
    a.set_defaults(func=cmd_export, no_video=True)


def reg_render(sub):
    a = sub.add_parser("render", help="layer 3: spec -> mp4")
    a.add_argument("spec")
    a.add_argument("--out", required=True)
    a.add_argument("--times-from", help="use the PTS of this video (VFR-safe pairing)")
    a.set_defaults(func=cmd_render, no_video=True)


def reg_compare(sub):
    a = sub.add_parser("compare", help="layer 3: reference video vs another video, in numbers")
    a.add_argument("video", help="reference (already analysed with track/curves/states)")
    a.add_argument("test")
    a.add_argument("--thr-track-fh", type=float, help="agreed p95 track distance threshold (default 0.01 is NOT agreed)")
    a.add_argument("--no-ssim", action="store_true")
    a.add_argument("--tracks", help="comma-separated track ids to compare (default: all)")
    a.set_defaults(func=cmd_compare)


def reg_verify(sub):
    a = sub.add_parser("verify", help="layer 3: render spec, analyse like the original, compare, optionally correct the spec")
    a.add_argument("video")
    a.add_argument("--spec", required=True)
    a.add_argument("--work-dir", default="verify_work")
    a.add_argument("--tag", default="it0")
    a.add_argument("--apply", action="store_true", help="write the corrected spec (to --next-spec or in place)")
    a.add_argument("--next-spec")
    a.add_argument("--thr-track-fh", type=float)
    a.add_argument("--no-ssim", action="store_true")
    a.set_defaults(func=cmd_verify)


REGISTER = [reg_track, reg_states, reg_curves, reg_spec, reg_export, reg_render, reg_compare, reg_verify]
