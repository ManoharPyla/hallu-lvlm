"""Isotonic regression calibration on the detector logit."""
from __future__ import annotations

import numpy as np

from hallu.calibration.base import Calibrator, _clean, sigmoid


class Isotonic(Calibrator):
    name = "isotonic"

    def __init__(self) -> None:
        self.iso = None

    def fit(self, logits, y) -> "Isotonic":
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            _clean(logits), np.asarray(y, np.float64))
        return self

    def transform(self, logits) -> np.ndarray:
        z = _clean(logits)
        if self.iso is None:
            return sigmoid(z)
        return np.clip(self.iso.predict(z), 0.0, 1.0)
