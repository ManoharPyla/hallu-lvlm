import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from hallu.models.detector import LogRegDetector, RiskMLP


def _data(n=2000, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32) * 5 + 3
    w = rng.normal(size=d)
    y = ((X - 3) @ w / 5 + rng.normal(size=n) > 1.5).astype(int)
    return X, y


CFG = dict(hidden_dims=[32], epochs=15, batch_size=128, patience=3, seed=0)


def test_mlp_fit_predict_shapes_and_auc():
    X, y = _data()
    m = RiskMLP(X.shape[1], [32]).fit(X[:1500], y[:1500], X[1500:], y[1500:], CFG)
    z = m.predict_logits(X[1500:])
    p = m.predict_proba(X[1500:])
    assert z.shape == (500,) and p.shape == (500,)
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(y[1500:], z) > 0.85


def test_mlp_deterministic():
    X, y = _data()
    a = RiskMLP(X.shape[1], [32]).fit(X, y, cfg={**CFG, "epochs": 3}).predict_logits(X)
    b = RiskMLP(X.shape[1], [32]).fit(X, y, cfg={**CFG, "epochs": 3}).predict_logits(X)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("loss", ["bce", "weighted_bce"])
def test_mlp_losses(loss):
    X, y = _data(500)
    m = RiskMLP(X.shape[1], [8]).fit(X, y, cfg={**CFG, "epochs": 2, "loss": loss})
    assert np.isfinite(m.predict_logits(X)).all()


def test_mlp_save_load(tmp_path):
    X, y = _data(500)
    m = RiskMLP(X.shape[1], [8]).fit(X, y, cfg={**CFG, "epochs": 2})
    m.save(tmp_path / "m.pt")
    m2 = RiskMLP.load(tmp_path / "m.pt")
    np.testing.assert_allclose(m.predict_logits(X), m2.predict_logits(X), rtol=1e-6)
    assert np.allclose(m2.mean.numpy(), X.mean(0), atol=1e-3)


def test_logreg(tmp_path):
    X, y = _data()
    d = LogRegDetector().fit(X[:1500], y[:1500], cfg={"class_weight": "balanced"})
    assert roc_auc_score(y[1500:], d.predict_logits(X[1500:])) > 0.85
    d.save(tmp_path / "lr.pkl")
    np.testing.assert_allclose(d.predict_proba(X), LogRegDetector.load(tmp_path / "lr.pkl").predict_proba(X))


def test_nan_inputs():
    X, y = _data(400)
    Xn = X.copy()
    Xn[::7, 0] = np.nan
    for det in (RiskMLP(X.shape[1], [8]), LogRegDetector()):
        det.fit(Xn, y, cfg={**CFG, "epochs": 2})
        assert np.isfinite(det.predict_logits(Xn)).all()
