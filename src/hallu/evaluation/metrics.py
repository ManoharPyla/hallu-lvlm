"""Evaluation metrics for token-level hallucination detection.

ECE definition (documented): predictions are partitioned into ``n_bins`` equal-width
(uniform) bins on [0, 1] (last bin closed on the right). For each non-empty bin b,
acc_b = mean(y in b), conf_b = mean(p in b). ECE = sum_b (|b| / N) * |acc_b - conf_b|,
i.e. the sample-weighted mean absolute calibration gap. ``y`` is the binary
"hallucinated" label and ``p`` the predicted probability of hallucination.

AUROC / AUPRC return ``nan`` (with a logged warning + RuntimeWarning) when only one
class is present.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Callable, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _arr(y, p=None):
    y = np.asarray(y).astype(float).ravel()
    if p is None:
        return y
    p = np.asarray(p).astype(float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y{y.shape} vs p{p.shape}")
    return y, p


def _single_class(y: np.ndarray, name: str) -> bool:
    if y.size == 0 or np.unique(y).size < 2:
        msg = f"{name}: only one class present (n={y.size}); returning nan"
        logger.warning(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return True
    return False


def auroc(y, p) -> float:
    y, p = _arr(y, p)
    if _single_class(y, "auroc"):
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def auprc(y, p) -> float:
    y, p = _arr(y, p)
    if _single_class(y, "auprc"):
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def _prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def detection_metrics(y, p, thr: float = 0.5) -> dict:
    y, p = _arr(y, p)
    prec, rec, f1 = _prf(y, (p >= thr).astype(float))
    return dict(precision=prec, recall=rec, f1=f1, auroc=auroc(y, p), auprc=auprc(y, p),
                pos_rate=float(y.mean()) if y.size else float("nan"), threshold=float(thr))


def _bin_ids(p: np.ndarray, n_bins: int, strategy: str):
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown strategy {strategy}")
    ids = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return ids, edges


def reliability_curve(y, p, n_bins: int = 15, strategy: str = "uniform") -> dict:
    y, p = _arr(y, p)
    ids, edges = _bin_ids(p, n_bins, strategy)
    nb = len(edges) - 1
    acc, conf, cnt = np.full(nb, np.nan), np.full(nb, np.nan), np.zeros(nb, int)
    for b in range(nb):
        m = ids == b
        cnt[b] = int(m.sum())
        if cnt[b]:
            acc[b], conf[b] = y[m].mean(), p[m].mean()
    return dict(bin_edges=edges, accuracy=acc, confidence=conf, count=cnt)


def ece(y, p, n_bins: int = 15, strategy: str = "uniform") -> float:
    y, p = _arr(y, p)
    if y.size == 0:
        return float("nan")
    rc = reliability_curve(y, p, n_bins, strategy)
    m = rc["count"] > 0
    w = rc["count"][m] / y.size
    return float(np.sum(w * np.abs(rc["accuracy"][m] - rc["confidence"][m])))


def brier(y, p) -> float:
    y, p = _arr(y, p)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def risk_coverage(y, p) -> dict:
    """Selective acceptance: accept tokens with lowest hallucination risk first.

    risk(c) = fraction hallucinated among the accepted c-fraction; AURC = mean risk.
    """
    y, p = _arr(y, p)
    if y.size == 0:
        return dict(coverage=np.array([]), risk=np.array([]), aurc=float("nan"))
    ys = y[np.argsort(p, kind="stable")]
    k = np.arange(1, y.size + 1)
    risk = np.cumsum(ys) / k
    return dict(coverage=k / y.size, risk=risk, aurc=float(risk.mean()))


def span_iou(pred_spans: Sequence, gold_spans: Sequence) -> float:
    """Character-level IoU between span unions ((start,end) tuples or objects with .start/.end)."""
    def chars(spans):
        s: set[int] = set()
        for sp in spans:
            a, b = (sp.start, sp.end) if hasattr(sp, "start") else (sp[0], sp[1])
            s.update(range(a, b))
        return s
    P, G = chars(pred_spans), chars(gold_spans)
    if not P and not G:
        return 1.0
    return len(P & G) / len(P | G)


def _norm_yn(x) -> int:
    s = str(x).strip().lower()
    return 1 if s.startswith("yes") or s in {"1", "true"} else 0


def pope_metrics(preds: Iterable, labels: Iterable) -> dict:
    """POPE metrics, 'yes' is the positive class."""
    p = np.array([_norm_yn(x) for x in preds], float)
    l = np.array([_norm_yn(x) for x in labels], float)
    if p.shape != l.shape:
        raise ValueError("preds/labels length mismatch")
    prec, rec, f1 = _prf(l, p)
    n = len(p)
    return dict(acc=float((p == l).mean()) if n else float("nan"), precision=prec, recall=rec,
                f1=f1, yes_ratio=float(p.mean()) if n else float("nan"))


def chair(caption_objects: Sequence[Iterable[str]], gt_objects: Sequence[Iterable[str]]) -> dict:
    """CHAIR (Rohrbach et al., 2018).

    caption_objects[i]: COCO classes mentioned in caption i (set semantics per caption);
    gt_objects[i]: ground-truth COCO classes of image i.
    CHAIR_i = hallucinated mentioned objects / all mentioned objects;
    CHAIR_s = captions with >=1 hallucinated object / all captions.
    """
    if len(caption_objects) != len(gt_objects):
        raise ValueError("length mismatch")
    n_obj = n_hal = n_cap_hal = 0
    for objs, gt in zip(caption_objects, gt_objects):
        objs, gt = set(objs), set(gt)
        h = objs - gt
        n_obj += len(objs)
        n_hal += len(h)
        n_cap_hal += bool(h)
    n = len(caption_objects)
    return dict(CHAIR_s=n_cap_hal / n if n else float("nan"),
                CHAIR_i=n_hal / n_obj if n_obj else float("nan"))


def bootstrap_ci(fn: Callable, *arrays, n: int = 1000, seed: int = 0, alpha: float = 0.05,
                 groups=None) -> dict:
    """Percentile bootstrap CI of ``fn(*arrays)``; with ``groups`` (e.g. image_id) resamples groups."""
    arrays = [np.asarray(a) for a in arrays]
    N = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(fn(*arrays))
    if groups is not None:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            if groups is None:
                idx = rng.integers(0, N, N)
            else:
                idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug))])
            v = fn(*[a[idx] for a in arrays])
            if v is not None and not math.isnan(float(v)):
                vals.append(float(v))
    if not vals:
        return dict(point=point, low=float("nan"), high=float("nan"), n=0)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return dict(point=point, low=float(lo), high=float(hi), n=len(vals))


def f1_at(y, p, thr) -> float:
    y, p = _arr(y, p)
    return _prf(y, (p >= thr).astype(float))[2]


def select_threshold(val_probs, val_y, objective: str = "f1", alpha: float = 0.1,
                     delta: float = 0.1, grid: Sequence[float] | None = None) -> float:
    """Choose a decision threshold using VALIDATION data only (never test data).

    objective="f1": maximize F1 over a grid.
    objective="risk_control": threshold with miss rate (FNR among positives) <= alpha w.p.
    1-delta, via ``hallu.calibration.conformal.risk_control_threshold`` if importable,
    else a Hoeffding-UCB fallback (largest grid threshold whose UCB(FNR) <= alpha).
    """
    y, p = _arr(val_y, val_probs)
    grid = np.linspace(0.01, 0.99, 99) if grid is None else np.asarray(grid, float)
    if objective == "f1":
        scores = [f1_at(y, p, t) for t in grid]
        return float(grid[int(np.argmax(scores))])
    if objective == "risk_control":
        try:
            from hallu.calibration.conformal import risk_control_threshold
            return float(risk_control_threshold(p, y, alpha, delta))
        except ImportError:
            logger.warning("hallu.calibration.conformal unavailable; using Hoeffding fallback")
        pos = p[y == 1]
        if pos.size == 0:
            logger.warning("no positives in validation; returning 0.0 (flag everything)")
            return 0.0
        slack = math.sqrt(math.log(1 / delta) / (2 * pos.size))
        ok = [t for t in grid if (pos < t).mean() + slack <= alpha]
        return float(max(ok)) if ok else 0.0
    raise ValueError(f"unknown objective {objective}")
