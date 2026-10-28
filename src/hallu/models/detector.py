"""Token-level hallucination detectors: RiskMLP (torch) and LogRegDetector (sklearn)."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class TokenDetector(Protocol):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None, cfg: dict | None = None) -> "TokenDetector": ...
    def predict_logits(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...


def _clean(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class RiskMLP(nn.Module):
    DEFAULTS = dict(hidden_dims=[256, 64], dropout=0.1, lr=1e-3, weight_decay=1e-4, epochs=50,
                    batch_size=512, patience=5, loss="weighted_bce", seed=0)

    def __init__(self, in_dim: int, hidden_dims: list[int] | tuple[int, ...] = (256, 64), dropout: float = 0.1) -> None:
        super().__init__()
        self.in_dim, self.hidden_dims, self.dropout = int(in_dim), list(hidden_dims), float(dropout)
        layers: list[nn.Module] = []
        d = self.in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", torch.zeros(self.in_dim))
        self.register_buffer("std", torch.ones(self.in_dim))
        self.cfg: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)

    def fit(self, X_train, y_train, X_val=None, y_val=None, cfg: dict | None = None) -> "RiskMLP":
        c = {**self.DEFAULTS, **(cfg or {})}
        self.cfg = c
        torch.manual_seed(int(c["seed"]))
        np.random.seed(int(c["seed"]))
        for mod in self.net:  # deterministic re-init under the seed
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
        X = _clean(X_train)
        y = np.asarray(y_train, np.float32)
        self.mean.copy_(torch.from_numpy(X.mean(0)))
        self.std.copy_(torch.from_numpy(X.std(0) + 1e-6))
        pos = float(y.sum())
        neg = float(len(y) - pos)
        pw = torch.tensor(neg / max(pos, 1.0)) if c["loss"] == "weighted_bce" else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.AdamW(self.parameters(), lr=float(c["lr"]), weight_decay=float(c["weight_decay"]))
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        g = torch.Generator().manual_seed(int(c["seed"]))
        bs = int(c["batch_size"])
        best, best_state, bad = -np.inf, None, 0
        for ep in range(int(c["epochs"])):
            self.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss_fn(self(Xt[b]), yt[b]).backward()
                opt.step()
            if X_val is not None and y_val is not None:
                score = _auprc(np.asarray(y_val), self.predict_logits(X_val))
                if np.isnan(score):
                    continue
                if score > best + 1e-5:
                    best, bad = score, 0
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                else:
                    bad += 1
                    if bad >= int(c["patience"]):
                        logger.info("early stop at epoch %d (best val AUPRC %.4f)", ep, best)
                        break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return self

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        return self(torch.from_numpy(_clean(X))).cpu().numpy().astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logits(X))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"in_dim": self.in_dim, "hidden_dims": self.hidden_dims, "dropout": self.dropout,
                    "cfg": self.cfg, "state": self.state_dict()}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "RiskMLP":
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        m = cls(d["in_dim"], d["hidden_dims"], d["dropout"])
        m.load_state_dict(d["state"])
        m.cfg = d.get("cfg", {})
        m.eval()
        return m


class LogRegDetector:
    def __init__(self) -> None:
        self.model: Any = None
        self.scaler: Any = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, cfg: dict | None = None) -> "LogRegDetector":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        c = {"C": 1.0, "class_weight": None, "max_iter": 1000, "seed": 0, **(cfg or {})}
        if c.get("loss") == "weighted_bce" and c["class_weight"] is None:
            c["class_weight"] = "balanced"
        X = _clean(X_train)
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(C=float(c["C"]), class_weight=c["class_weight"],
                                        max_iter=int(c["max_iter"]), random_state=int(c["seed"]))
        self.model.fit(self.scaler.transform(X), np.asarray(y_train).astype(int))
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(self.scaler.transform(_clean(X))).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logits(X))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    @classmethod
    def load(cls, path: str | Path) -> "LogRegDetector":
        with open(path, "rb") as f:
            d = pickle.load(f)
        o = cls()
        o.model, o.scaler = d["model"], d["scaler"]
        return o


def build_detector(name: str, in_dim: int, cfg: dict | None = None) -> RiskMLP | LogRegDetector:
    cfg = cfg or {}
    if name in ("mlp", "risk_mlp"):
        return RiskMLP(in_dim, cfg.get("hidden_dims", (256, 64)), cfg.get("dropout", 0.1))
    if name in ("logreg", "lr"):
        return LogRegDetector()
    raise ValueError(f"unknown detector {name!r}")
