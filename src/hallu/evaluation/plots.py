"""Plots (matplotlib imported lazily, Agg backend). Each function returns the saved path."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hallu.evaluation.metrics import f1_at, reliability_curve, risk_coverage

logger = logging.getLogger(__name__)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    _plt().close(fig)
    return str(path)


def plot_reliability(y, p, path, n_bins: int = 15, title: str = "Reliability diagram") -> str:
    plt = _plt()
    rc = reliability_curve(y, p, n_bins)
    centers = (rc["bin_edges"][:-1] + rc["bin_edges"][1:]) / 2
    m = rc["count"] > 0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.bar(centers[m], rc["accuracy"][m], width=1 / n_bins, edgecolor="k", alpha=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted hallucination probability")
    ax.set_ylabel("Observed hallucination frequency")
    ax.set_title(title)
    return _save(fig, path)


def plot_risk_coverage(curves: dict, path, title: str = "Risk-coverage") -> str:
    """curves: name -> (y, p)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4, 4))
    for name, (y, p) in curves.items():
        rc = risk_coverage(y, p)
        ax.plot(rc["coverage"], rc["risk"], label=f"{name} (AURC={rc['aurc']:.3f})")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk")
    ax.legend(fontsize=7); ax.set_title(title)
    return _save(fig, path)


def plot_threshold_sweep(y, p, path, grid=None, title: str = "Threshold sweep") -> str:
    plt = _plt()
    y, p = np.asarray(y, float), np.asarray(p, float)
    grid = np.linspace(0.01, 0.99, 99) if grid is None else np.asarray(grid, float)
    npos = max(float((y == 1).sum()), 1.0)
    prec, rec, f1 = [], [], []
    for t in grid:
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        prec.append(tp / max(float(pred.sum()), 1.0)); rec.append(tp / npos); f1.append(f1_at(y, p, t))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(grid, f1, label="F1"); ax.plot(grid, prec, label="Precision"); ax.plot(grid, rec, label="Recall")
    ax.set_xlabel("Threshold"); ax.legend(); ax.set_title(title)
    return _save(fig, path)
