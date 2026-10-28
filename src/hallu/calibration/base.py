"""Calibrator base class and Identity calibrator."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, np.float64), -50, 50)))


def _clean(logits: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(logits, np.float64).reshape(-1), nan=0.0, posinf=50.0, neginf=-50.0)


class Calibrator:
    """Maps detector logits to calibrated probabilities."""

    name = "base"

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "Calibrator":
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, Calibrator):
            raise TypeError(f"{path} does not contain a Calibrator")
        return obj


class Identity(Calibrator):
    name = "identity"

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(_clean(logits))
