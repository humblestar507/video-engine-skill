"""Funnel step 1: a few lines that say where to look next."""


def _fmt_t(x):
    return f"{x:.3f}"


def _v(d):
    return d["v"] if isinstance(d, dict) and "v" in d else d


def text(store):
    p = store.get("probe")
    if not p:
        return "not analyzed yet: run `ve analyze VIDEO`"
    v = p["video"]
    L = []
    fps = _v(v["fps"])
    vfr = _v(v["vfr"])
    a = p.get("audio")
    L.append(f"video: {store.video.name} | {_v(v['duration']):.3f}s | {v['width']}x{v['height']}"
             f" | {fps:.3f} fps {'VFR' if vfr else 'CFR'} (dt {v['dt_ms']['min']:.1f}-{v['dt_ms']['max']:.1f} ms)"
             f" | {_v(v['frames'])} frames | audio: {a['codec'] + ' ' + str(a['sample_rate']) + 'Hz' if a else 'none'}")
    sc = store.get("scenes")
    if sc:
        cuts = [e for e in sc["events"] if e["type"] == "cut"]
        other = {}
        for e in sc["events"]:
            if e["type"] != "cut":
                other[e["type"]] = other.get(e["type"], 0) + 1
        cut_s = ", ".join(_fmt_t(e["t"]) for e in cuts[:12]) + (" ..." if len(cuts) > 12 else "")
        L.append(f"scenes: {sc['count']}" + (f" | cuts at {cut_s}" if cuts else " | no cuts")
                 + ("".join(f" | {k} x{n}" for k, n in other.items())))
        segs = sc["activity_segments"]["items"]
        if segs:
            top = sorted(segs, key=lambda s: -s["peak"])[:5]
            L.append("activity (top): " + ", ".join(f"{s['t0']:.2f}-{s['t1']:.2f}s" for s in sorted(top, key=lambda s: s["t0"]))
                     + f" | {sc['activity_segments']['total']} segments")
        static = [s["id"] for s in sc["scenes"] if _v(s["static"])]
        if static:
            L.append(f"static scenes: {static}")
    mo = store.get("motion")
    if mo:
        L.append("motion: " + mo["summary_line"])
    au = store.get("audio")
    if au:
        L.append("audio: " + au["summary_line"])
    tr = store.get("tracks") or {}
    if tr:
        L.append("tracks: " + ", ".join(f"{k} '{t.get('name', '')}' {t['t0']:.2f}-{t['t1']:.2f}s" for k, t in tr.items()))
    st = store.get("states") or {}
    if st:
        L.append("states: " + ", ".join(f"{k}: {len(s['palette'])} poses / {len(s['timeline'])} intervals" for k, s in st.items()))
    cv = store.get("curves") or {}
    if cv:
        L.append("curves: " + "; ".join(f"{k}: {c['summary_line']}" for k, c in cv.items()))
    warns = []
    for sec in ("probe", "scenes", "motion", "audio"):
        warns += (store.get(sec) or {}).get("warnings", [])
    if warns:
        L.append("warnings: " + " | ".join(warns))
    L.append(f"cache: {store.dir}")
    return "\n".join(L)
