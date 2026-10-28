import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from hallu.calibration import get_calibrator, risk_control_threshold
from hallu.calibration.base import Calibrator


def _ece(y, p, n_bins=15):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
               for b in range(n_bins) if (bins == b).any())


def _miscalibrated(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-1.0, 2.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(int)
    return 3.0 * true_logit, y  # overconfident logits


@pytest.mark.parametrize("name", ["platt", "temperature", "isotonic"])
def test_ece_improves(name):
    z, y = _miscalibrated()
    zc, yc, zt, yt = z[:3000], y[:3000], z[3000:], y[3000:]
    before = _ece(yt, get_calibrator("identity").transform(zt))
    cal = get_calibrator(name).fit(zc, yc)
    p = cal.transform(zt)
    assert p.shape == zt.shape and np.all((p >= 0) & (p <= 1))
    assert _ece(yt, p) < before * 0.6


def test_temperature_recovers_scale():
    z, y = _miscalibrated()
    cal = get_calibrator("temperature").fit(z, y)
    assert 2.4 < cal.T < 3.6


@pytest.mark.parametrize("name", ["identity", "platt", "temperature", "isotonic"])
def test_save_load_roundtrip(name, tmp_path):
    z, y = _miscalibrated(500)
    cal = get_calibrator(name).fit(z, y)
    cal.save(tmp_path / "c.pkl")
    cal2 = Calibrator.load(tmp_path / "c.pkl")
    np.testing.assert_allclose(cal.transform(z), cal2.transform(z))


def test_nan_handling():
    z, y = _miscalibrated(500)
    z[::50] = np.nan
    for name in ["platt", "temperature", "isotonic"]:
        p = get_calibrator(name).fit(z, y).transform(np.array([np.nan, np.inf, -np.inf, 0.0]))
        assert np.all(np.isfinite(p))


def test_unknown_name():
    with pytest.raises(ValueError):
        get_calibrator("nope")


def test_conformal_guarantee_empirical():
    rng = np.random.default_rng(1)
    alpha, delta, fails, trials = 0.2, 0.1, 0, 200
    for _ in range(trials):
        n = 2000
        y = (rng.random(n) < 0.3).astype(int)
        p = np.clip(rng.beta(2, 5, n) + 0.35 * y, 0, 1)
        thr = risk_control_threshold(p, y, alpha, delta)
        y2 = (rng.random(20000) < 0.3).astype(int)
        p2 = np.clip(rng.beta(2, 5, 20000) + 0.35 * y2, 0, 1)
        if np.mean(p2[y2 == 1] < thr) > alpha:
            fails += 1
    assert fails / trials <= delta


def test_conformal_edge_cases():
    assert risk_control_threshold(np.array([0.1, 0.2]), np.array([0, 0]), 0.1) == 0.0
    y = np.array([1, 1, -100, 0])
    assert 0.0 <= risk_control_threshold(np.array([0.9, 0.8, 0.5, np.nan]), y, 0.5) <= 1.0
