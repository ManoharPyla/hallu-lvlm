"""Calibrate trained detectors and select thresholds; report test metrics.

Protocol (no test-set tuning):
  * calibrator fitted on the CAL split only (identity | platt | temperature | isotonic)
  * F1 threshold selected on the VAL split
  * risk-control threshold (miss rate <= alpha w.p. >= 1-delta) computed on the CAL split
  * metrics reported on TEST, with image-level bootstrap CIs

    python scripts/calibrate_detector.py --run-dir experiments/runs/main_mlp \
        --features-dir data/features/<model_tag> --exp main_mlp

Writes <run-dir>/calibrator_<method>_seed<k>.pkl, <run-dir>/thresholds.json and
experiments/results/<exp>/metrics.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hallu.utils.config import load_config  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "evaluation.yaml"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--exp", default=None, help="results name (default: run-dir name)")
    ap.add_argument("--results-root", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.overrides)

    import numpy as np
    from hallu.calibration import get_calibrator, risk_control_threshold
    from hallu.evaluation.metrics import (auprc, bootstrap_ci, brier, detection_metrics, ece, f1_at,
                                          select_threshold)
    from hallu.features.record import build_matrix, load_records
    from hallu.utils.io import load_json, save_json
    from train_detector import load_detector

    run = Path(a.run_dir)
    meta = load_json(run / "train_meta.json")
    use, name = tuple(meta["features"]), meta["detector"]
    fd = Path(a.features_dir)
    data = {s: build_matrix(load_records(fd / s), use) for s in ("cal", "val", "test")}
    methods = a.methods or cfg["calibration"]["methods"]
    dcfg = cfg["detection"]
    g = dcfg["threshold_grid"]
    grid = np.linspace(g["start"], g["stop"], g["num"])
    alpha, delta = dcfg["risk_control"]["alpha"], dcfg["risk_control"]["delta"]
    nb = cfg["calibration"]["ece"]["n_bins"]
    nboot = a.bootstrap if a.bootstrap is not None else cfg["bootstrap"]["n"]
    yte, gte = data["test"][1], data["test"][2]

    rows, thresholds = {}, {}
    for seed in meta["seeds"]:
        det = load_detector(run, name, seed)
        logits = {s: det.predict_logits(data[s][0]) for s in data}
        for m in methods:
            cal = get_calibrator(m).fit(logits["cal"], data["cal"][1])
            cal.save(run / f"calibrator_{m}_seed{seed}.pkl")
            p = {s: cal.transform(logits[s]) for s in data}
            thr_f1 = select_threshold(p["val"], data["val"][1], "f1", grid=grid)
            thr_rc = risk_control_threshold(p["cal"], data["cal"][1], alpha, delta)
            thresholds[f"{m}_seed{seed}"] = {"f1_val": thr_f1, "risk_control_cal": thr_rc,
                                             "alpha": alpha, "delta": delta}
            for tname, thr in (("f1", thr_f1), ("risk_control", thr_rc)):
                r = detection_metrics(yte, p["test"], thr)
                pos = p["test"][yte == 1]
                r.update(ece=ece(yte, p["test"], nb), brier=brier(yte, p["test"]), threshold=thr,
                         miss_rate=float(np.mean(pos < thr)) if pos.size else None,
                         f1_ci=bootstrap_ci(lambda y, q, t=thr: f1_at(y, q, t), yte, p["test"],
                                            n=nboot, seed=seed, groups=gte) if nboot else None,
                         auprc_ci=bootstrap_ci(auprc, yte, p["test"], n=nboot, seed=seed,
                                               groups=gte) if nboot else None)
                rows[f"{name}/{m}/{tname}/seed{seed}"] = r
    save_json(run / "thresholds.json", thresholds)
    exp = a.exp or run.name
    save_json(Path(a.results_root) / exp / "metrics.json",
              {"experiment": exp, "synthetic": False, "detector": name, "features": list(use),
               "split_protocol": "calibrator fit on cal; F1 threshold on val; risk-control on cal; metrics on test",
               "tables": {"detection": rows}})
    print(f"wrote {Path(a.results_root) / exp / 'metrics.json'}")


if __name__ == "__main__":
    main()
