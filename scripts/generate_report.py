"""Collect experiments/results/*/metrics.json -> Markdown + LaTeX tables (+ plots if possible).

    python scripts/generate_report.py

Writes paper/tables/generated/<exp>_<table>.{md,tex} and paper/tables/generated/REPORT.md.
Missing values render as '--'; nothing is invented. Synthetic results are labeled as such.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "tables" / "generated"))
    ap.add_argument("--digits", type=int, default=3)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from hallu.evaluation.report import to_latex, to_markdown
    from hallu.utils.io import load_json

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = ["# Generated results report", ""]
    files = sorted(Path(a.results_root).glob("*/metrics.json"))
    if not files:
        report.append("No results found.")
    for f in files:
        m = load_json(f)
        exp = m.get("experiment", f.parent.name)
        report.append(f"## {exp}")
        if m.get("synthetic"):
            report.append(f"**{m.get('banner', 'SYNTHETIC — NOT A BENCHMARK RESULT')}**")
        for k in ("deviation", "note", "split_protocol"):
            if m.get(k):
                report.append(f"_{k}: {m[k]}_")
        tables = dict(m.get("tables", {}))
        if "metrics" in m and isinstance(m["metrics"], dict):
            mt = m["metrics"]
            tables["metrics"] = mt if all(isinstance(v, dict) for v in mt.values()) else {exp: mt}
        for tname, rows in tables.items():
            cols = sorted({c for r in rows.values() if isinstance(r, dict) for c in r
                           if not isinstance(r[c], (list, str)) and not (isinstance(r[c], dict) and "point" not in r[c])})
            if not cols:
                continue
            md = to_markdown(rows, cols, digits=a.digits)
            cap = f"{exp} {tname}" + (" (SYNTHETIC smoke test, not a benchmark)" if m.get("synthetic") else "")
            (out / f"{exp}_{tname}.md").write_text(md + "\n", encoding="utf-8")
            (out / f"{exp}_{tname}.tex").write_text(
                to_latex(rows, cols, digits=a.digits, caption=cap, label=f"tab:{exp}_{tname}") + "\n",
                encoding="utf-8")
            report += [f"### {tname}", "", md, ""]
    plots = _plots(files, out)
    if plots:
        report += ["## Plots", ""] + [f"- {p}" for p in plots]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out / 'REPORT.md'} ({len(files)} experiments)")


def _plots(files, out: Path) -> list[str]:
    """Bar plot of detection F1 per experiment; skipped gracefully without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; plots skipped")
        return []
    from hallu.utils.io import load_json
    paths = []
    for f in files:
        m = load_json(f)
        det = m.get("tables", {}).get("detection")
        if not det:
            continue
        names = list(det)
        vals = [float(det[n].get("f1") or 0) for n in names]
        fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(names)), 3))
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("F1 (test)")
        ax.set_title(m.get("experiment", f.parent.name) + (" [SYNTHETIC]" if m.get("synthetic") else ""))
        p = out / f"{m.get('experiment', f.parent.name)}_f1.png"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        paths.append(str(p))
    return paths


if __name__ == "__main__":
    main()
