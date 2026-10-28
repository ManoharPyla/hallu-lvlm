"""Platt scaling: logistic regression on the detector logit."""
from __future__ import annotations

import numpy as np

from hallu.calibration.base import Calibrator, _clean, sigmoid


class Platt(Calibrator):
    name = "platt"

    def __init__(self) -> None:
        self.a, self.b = 1.0, 0.0

    def fit(self, logits, y) -> "Platt":
        from sklearn.linear_model import LogisticRegression
        z = _clean(logits)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            return self
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, logits) -> np.ndarray:
        return sigmoid(self.a * _clean(logits) + self.b)
