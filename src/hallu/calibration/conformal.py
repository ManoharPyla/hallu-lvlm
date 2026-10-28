"""Conformal risk control for the flagging threshold.

Goal: choose threshold t (flag token if p >= t) such that the miss rate among
hallucinated tokens, FNR(t) = P(p < t | y=1), is <= alpha with probability >= 1-delta.

Learn-Then-Test style: scan thresholds from high to low (fixed-sequence testing, so no
multiplicity correction is needed) and accept t while the Hoeffding upper confidence
bound  FNR_hat(t) + sqrt(log(1/delta) / (2 n_pos))  <= alpha. FNR is monotone
non-increasing as t decreases, so we return the largest threshold that passes
(most conservative flagging rate). If none passes, returns 0.0 (flag everything).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def risk_control_threshold(probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float, delta: float = 0.1,
                           grid: np.ndarray | None = None) -> float:
    p = np.nan_to_num(np.asarray(probs_cal, np.float64).reshape(-1), nan=1.0)
    y = np.asarray(y_cal).reshape(-1)
    keep = y != -100
    p, y = p[keep], y[keep].astype(int)
    pos = p[y == 1]
    n = len(pos)
    if n == 0:
        logger.warning("no positives in calibration set; returning threshold 0.0")
        return 0.0
    slack = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    if grid is None:
        grid = np.unique(np.concatenate([pos, [0.0, 1.0]]))
    grid = np.sort(np.asarray(grid, np.float64))[::-1]  # high -> low
    best = None
    for t in grid:
        fnr = float(np.mean(pos < t))
        if fnr + slack <= alpha:
            best = float(t)
            break  # fixed-sequence: first (largest) passing threshold
    if best is None:
        logger.warning("no threshold certifies FNR<=%.3f (n_pos=%d, slack=%.3f); flagging all", alpha, n, slack)
        return 0.0
    return best
