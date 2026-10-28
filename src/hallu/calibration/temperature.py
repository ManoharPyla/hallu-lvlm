"""Temperature scaling fit by LBFGS on binary NLL."""
from __future__ import annotations

import numpy as np
import torch

from hallu.calibration.base import Calibrator, _clean, sigmoid


class Temperature(Calibrator):
    name = "temperature"

    def __init__(self) -> None:
        self.T = 1.0

    def fit(self, logits, y) -> "Temperature":
        z = torch.tensor(_clean(logits), dtype=torch.float64)
        t = torch.tensor(np.asarray(y, np.float64))
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(log_t) > 0
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(z / log_t.exp(), t)
            loss.backward()
            return loss

        opt.step(closure)
        T = float(log_t.exp().item())
        self.T = T if np.isfinite(T) and T > 0 else 1.0
        return self

    def transform(self, logits) -> np.ndarray:
        return sigmoid(_clean(logits) / self.T)
