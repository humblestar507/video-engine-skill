"""Funnel step 3: slices of video.json as small tables.

`what` is a dotted path into video.json (e.g. scenes.events, tracks.1.samples,
audio.onsets) or `signals.<name>` for per-frame arrays in the npz files."""
import json

import numpy as np


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, dict) and "v" in x and "src" in x:
        return f"{_fmt(x['v'])} [{x['src'][0]}{x['conf']}]"
    if isinstance(x, (dict, list)):
        s = json.dumps(x, ensure_ascii=False)
        return s if len(s) < 60 else s[:57] + "..."
    return str(x)


def _rows_table(rows, cols, max_rows):
    n = len(rows)
    if n > max_rows:
        idx = np.unique(np.linspace(0, n - 1, max_rows).astype(int))
        rows = [rows[i] for i in idx]
        note = f"# {n} rows, showing {len(rows)} evenly spaced (use --t0/--t1 or --max-rows)"
    else:
        note = f"# {n} rows"
    out = [note, "\t".join(cols)]
    for r in rows:
        out.append("\t".join(_fmt(r.get(c, "")) for c in cols))
    return "\n".join(out)


def _in_window(item, t0, t1):
    a = item.get("t0", item.get("t"))
    b = item.get("t1", item.get("t_end", a))
    if a is None:
        return True
    return (t1 is None or a <= t1) and (t0 is None or b >= t0)


def columns_to_rows(d):
    keys = [k for k, v in d.items() if isinstance(v, list)]
    n = len(d[keys[0]])
    return [{k: d[k][i] for k in keys if len(d[k]) == n} for i in range(n)], [k for k in keys if len(d[k]) == n]


def run(store, what, t0=None, t1=None, max_rows=40):
    if what.startswith("signals."):
        name = what.split(".", 1)[1]
        for f in sorted(store.dir.glob("signals_*.npz")):
            z = np.load(f)
            if name in z.files:
                t = z["t"]
                m = np.ones(len(t), bool)
                if t0 is not None:
                    m &= t >= t0
                if t1 is not None:
                    m &= t <= t1
                rows = [{"t": float(a), name: float(b)} for a, b in zip(t[m], z[name][m])]
                return _rows_table(rows, ["t", name], max_rows)
        return f"no signal '{name}'"
    node = store.data
    for k in what.split("."):
        if isinstance(node, dict) and k in node:
            node = node[k]
        elif isinstance(node, list) and k.isdigit() and int(k) < len(node):
            node = node[int(k)]
        else:
            return f"no such path: {what} (stopped at '{k}'; keys: {list(node)[:30] if isinstance(node, dict) else type(node).__name__})"
    if isinstance(node, dict) and "t" in node and isinstance(node["t"], list):
        rows, cols = columns_to_rows(node)
        rows = [r for r in rows if _in_window(r, t0, t1)]
        return _rows_table(rows, cols, max_rows)
    if isinstance(node, list) and node and isinstance(node[0], dict):
        rows = [r for r in node if _in_window(r, t0, t1)]
        cols = []
        for r in rows:
            for c in r:
                if c not in cols and not isinstance(r[c], list):
                    cols.append(c)
        return _rows_table(rows, cols, max_rows)
    if isinstance(node, dict):
        lines = []
        for k, v in node.items():
            if k == "_meta":
                continue
            if isinstance(v, list) and len(v) > 8:
                lines.append(f"{k}: list[{len(v)}] -> query {what}.{k}")
            elif isinstance(v, dict) and not ("v" in v and "src" in v) and len(json.dumps(v)) > 120:
                lines.append(f"{k}: {{{', '.join(list(v)[:12])}}} -> query {what}.{k}")
            else:
                lines.append(f"{k}: {_fmt(v) if not isinstance(v, (dict, list)) or ('v' in v if isinstance(v, dict) else False) else json.dumps(v, ensure_ascii=False)}")
        return "\n".join(lines)
    return json.dumps(node, ensure_ascii=False)
