#!/usr/bin/env python
"""video-engine CLI. Run `python ve.py -h`. Every command takes the video path;
results are cached per video, repeated calls print `(cached)` and do no work."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from engine import probe, query, scenes, sheets, summary  # noqa: E402
from engine.store import Store  # noqa: E402

LAYER0 = [("probe", probe.run), ("scenes", scenes.run)]


def _layer1():
    mods = []
    try:
        from engine import motion
        mods.append(("motion", motion.run))
    except ImportError:
        pass
    try:
        from engine import audio
        mods.append(("audio", audio.run))
    except ImportError:
        pass
    return mods


def _crop(s):
    if not s:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise SystemExit("--crop needs x,y,w,h")
    return vals


def ensure(store, upto=1, force=False, quiet=False):
    mods = LAYER0 + (_layer1() if upto >= 1 else [])
    for name, fn in mods:
        res, cached = fn(store, force=force)
        if not quiet:
            el = (res.get("_meta") or {}).get("elapsed_s")
            print(f"[{name}] {'(cached)' if cached else f'done in {el}s'}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ve", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="layers 0-1 on the whole video, then the summary")
    a.add_argument("video")
    a.add_argument("--layers", type=int, default=1, help="0 = probe+scenes only, 1 = +motion+audio")
    a.add_argument("--force", action="store_true")

    a = sub.add_parser("summary", help="funnel step 1: short overview")
    a.add_argument("video")

    a = sub.add_parser("sheet", help="funnel step 2: contact sheet with timecodes")
    a.add_argument("video")
    a.add_argument("--t0", type=float)
    a.add_argument("--t1", type=float)
    a.add_argument("--scene", type=int, help="limit to one scene id")
    a.add_argument("--n", type=int, default=20)
    a.add_argument("--width", type=int, default=256, help="tile width px")
    a.add_argument("--cols", type=int, default=5)
    a.add_argument("--motion", action="store_true", help="overlay layer-1 motion (needs analyze --layers 1)")

    a = sub.add_parser("query", help="funnel step 3: table from video.json (dotted path)")
    a.add_argument("video")
    a.add_argument("what", nargs="?", default="")
    a.add_argument("--t0", type=float)
    a.add_argument("--t1", type=float)
    a.add_argument("--max-rows", type=int, default=40)

    a = sub.add_parser("frame", help="funnel step 4: one frame or crop at full res")
    a.add_argument("video")
    a.add_argument("--t", type=float, required=True)
    a.add_argument("--crop", help="x,y,w,h")
    a.add_argument("--units", default="fh", choices=["fh", "px", "frac"])
    a.add_argument("--width", type=int)

    a = sub.add_parser("strip", help="funnel step 4: N frames of a short range in one image")
    a.add_argument("video")
    a.add_argument("--t0", type=float, required=True)
    a.add_argument("--t1", type=float, required=True)
    a.add_argument("--n", type=int, default=8)
    a.add_argument("--crop", help="x,y,w,h")
    a.add_argument("--units", default="fh", choices=["fh", "px", "frac"])
    a.add_argument("--width", type=int, default=200)
    a.add_argument("--cols", type=int, default=8)

    a = sub.add_parser("path", help="print cache dir and video.json path")
    a.add_argument("video")

    for reg in _extra_commands():
        reg(sub)

    args = ap.parse_args(argv)
    st = Store(args.video) if hasattr(args, "video") else None

    if args.cmd == "analyze":
        ensure(st, upto=args.layers, force=args.force)
        print(summary.text(st))
    elif args.cmd == "summary":
        ensure(st, upto=0, quiet=True)
        print(summary.text(st))
    elif args.cmd == "sheet":
        ensure(st, upto=0, quiet=True)
        t0, t1 = args.t0, args.t1
        if args.scene is not None:
            s = st.get("scenes")["scenes"][args.scene]
            t0, t1 = s["t0"], s["t1"] - 1e-6
        if args.motion:
            ensure(st, upto=1, quiet=True)
        path, times = sheets.contact_sheet(st, t0, t1, args.n, args.width, args.cols, overlay=args.motion)
        print(path)
        print("times: " + ", ".join(f"{x:.3f}" for x in times))
    elif args.cmd == "query":
        ensure(st, upto=0, quiet=True)
        if args.what:
            print(query.run(st, args.what, args.t0, args.t1, args.max_rows))
        else:
            print("sections: " + ", ".join(k for k in st.data if k != "schema"))
    elif args.cmd == "frame":
        ensure(st, upto=0, quiet=True)
        path, ta = sheets.frame(st, args.t, _crop(args.crop), args.units, args.width)
        print(path)
        print(f"t_actual: {ta:.4f}")
    elif args.cmd == "strip":
        ensure(st, upto=0, quiet=True)
        path, got = sheets.strip(st, args.t0, args.t1, args.n, _crop(args.crop), args.units, args.width, args.cols)
        print(path)
        print("times: " + ", ".join(f"{x:.3f}" for x in got))
    elif args.cmd == "path":
        print(st.dir)
        print(st.json_path)
    else:
        args.func(st, args)


def _extra_commands():
    """Layer 2-3 commands register themselves if their modules exist."""
    regs = []
    try:
        from engine import cli_ext
        regs += cli_ext.REGISTER
    except ImportError:
        pass
    return regs


if __name__ == "__main__":
    main()
