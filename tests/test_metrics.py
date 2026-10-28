import math
import warnings

import numpy as np
import pytest

from hallu.evaluation import metrics as M
from hallu.evaluation.report import to_latex, to_markdown


def test_ece_perfect_and_known():
    assert M.ece([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    # all p=0.8 in one bin, acc 0.5 -> |0.5-0.8| = 0.3
    assert M.ece([0, 1, 0, 1], [0.8] * 4, n_bins=15) == pytest.approx(0.3)


def test_single_class_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = M.detection_metrics([0, 0, 0], [0.1, 0.6, 0.2])
    assert math.isnan(d["auroc"]) and math.isnan(d["auprc"])
    with pytest.warns(RuntimeWarning):
        M.auroc([1, 1], [0.2, 0.3])


def test_detection_and_pope_chair():
    d = M.detection_metrics([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.6])
    assert d["precision"] == 0.5 and d["recall"] == 0.5 and d["auroc"] == 0.75
    p = M.pope_metrics(["Yes", "no", "yes"], ["yes", "no", "no"])
    assert p["acc"] == pytest.approx(2 / 3) and p["yes_ratio"] == pytest.approx(2 / 3)
    c = M.chair([{"dog", "cat"}, {"car"}], [{"dog"}, {"car"}])
    assert c["CHAIR_i"] == pytest.approx(1 / 3) and c["CHAIR_s"] == 0.5


def test_risk_coverage_bootstrap_iou():
    rc = M.risk_coverage([0, 1], [0.1, 0.9])
    assert rc["risk"].tolist() == [0.0, 0.5]
    ci = M.bootstrap_ci(np.mean, np.array([0, 1] * 50), n=200)
    assert ci["low"] <= ci["point"] <= ci["high"]
    assert M.span_iou([(0, 4)], [(2, 6)]) == pytest.approx(2 / 6)


def test_select_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.1, 500), 0, 1)
    t = M.select_threshold(p, y, "f1")
    assert 0.2 < t < 0.8
    t2 = M.select_threshold(p, y, "risk_control", alpha=0.2)
    assert ((p[y == 1] < t2).mean()) <= 0.2


def test_report_missing():
    res = {"a": {"f1": 0.5, "ece": float("nan")}, "b": None}
    md = to_markdown(res, ["f1", "ece", "auroc"])
    assert "0.500" in md and md.count("--") >= 4
    assert "\\toprule" in to_latex(res, ["f1"])
