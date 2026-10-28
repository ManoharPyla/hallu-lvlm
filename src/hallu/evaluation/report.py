"""Render results dicts to Markdown / LaTeX tables.

Never invents numbers: missing keys, None and NaN render as '--'.
"""
from __future__ import annotations

import math
from typing import Sequence

MISSING = "--"


def _ok(x) -> bool:
    try:
        return x is not None and not math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def fmt(v, digits: int = 3) -> str:
    if v is None:
        return MISSING
    if isinstance(v, dict):
        if "point" not in v or not _ok(v.get("point")):
            return MISSING
        s = fmt(v["point"], digits)
        if _ok(v.get("low")) and _ok(v.get("high")):
            s += f" [{float(v['low']):.{digits}f}, {float(v['high']):.{digits}f}]"
        return s
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v if v else MISSING
    return f"{float(v):.{digits}f}" if _ok(v) else MISSING


def _rows(results: dict, columns: Sequence[str], digits: int) -> list[list[str]]:
    return [[str(name)] + [fmt((r or {}).get(c), digits) for c in columns]
            for name, r in results.items()]


def to_markdown(results: dict, columns: Sequence[str], row_header: str = "method",
                digits: int = 3) -> str:
    """results: row_name -> {column: value | bootstrap-CI dict}."""
    lines = ["| " + " | ".join([row_header, *columns]) + " |",
             "|" + "---|" * (len(columns) + 1)]
    lines += ["| " + " | ".join(r) + " |" for r in _rows(results, columns, digits)]
    return "\n".join(lines)


def _tex(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def to_latex(results: dict, columns: Sequence[str], row_header: str = "method", digits: int = 3,
             caption: str | None = None, label: str | None = None) -> str:
    out = [r"\begin{table}[t]", r"\centering",
           r"\begin{tabular}{l" + "c" * len(columns) + "}", r"\toprule",
           " & ".join(_tex(x) for x in [row_header, *columns]) + r" \\", r"\midrule"]
    out += [" & ".join(_tex(x) for x in r) + r" \\" for r in _rows(results, columns, digits)]
    out += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        out.append(r"\caption{" + _tex(caption) + "}")
    if label:
        out.append(r"\label{" + label + "}")
    out.append(r"\end{table}")
    return "\n".join(out)
