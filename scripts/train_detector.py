"""Train token-level hallucination detectors (mlp | logreg) for one or more seeds.

    python scripts/train_detector.py --features-dir data/features/<model_tag> --detector mlp --seeds 0 1 2 \
        --use hidden logprob entropy grounding --run-dir experiments/runs/main_mlp

Trains on the TRAIN split; early stopping (MLP) uses the VAL split. Never touches test.
Writes <run-dir>/detector_seed<k>.(pt|pkl) and <run-dir>/train_meta.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu.utils.config import load_config  # noqa: E402


def detector_path(run_dir: Path, name: str, seed: int) -> Path:
    return Path(run_dir) / f"detector_seed{seed}.{'pt' if name == 'mlp' else 'pkl'}"


def load_detector(run_dir: Path, name: str, seed: int):
    from hallu.models.detector import LogRegDetector, RiskMLP
    p = detector_path(run_dir, name, seed)
    return RiskMLP.load(p) if name == "mlp" else LogRegDetector.load(p)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "training.yaml"))
    ap.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    ap.add_argument("--features-dir", required=True, help="dir containing train/ val/ record dirs")
    ap.add_argument("--detector", choices=["mlp", "logreg"], default="mlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--use", nargs="+", default=None, help="feature groups (default: model.yaml)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.overrides)
    mcfg = load_config(a.model_config)

    import numpy as np
    from hallu.features.record import build_matrix, load_records
    from hallu.models.detector import build_detector
    from hallu.utils.io import save_json
    from hallu.utils.seed import set_seed

    use = tuple(a.use or mcfg["detector"]["features"])
    fd = Path(a.features_dir)
    Xtr, ytr, _, _ = build_matrix(load_records(fd / "train"), use)
    Xva, yva, _, _ = build_matrix(load_records(fd / "val"), use)
    logging.info("train %s (pos %.4f), val %s", Xtr.shape, float(np.mean(ytr)) if len(ytr) else 0, Xva.shape)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    det_cfg = {**cfg.get("train", {}), **(cfg.get("logreg", {}) if a.detector == "logreg" else {}),
               "hidden_dims": mcfg["detector"].get("hidden_dims", [256, 64]),
               "dropout": mcfg["detector"].get("dropout", 0.1)}
    meta = {"detector": a.detector, "features": list(use), "seeds": a.seeds, "in_dim": int(Xtr.shape[1]),
            "n_train_tokens": int(len(ytr)), "n_val_tokens": int(len(yva)),
            "features_dir": str(fd), "cfg": det_cfg, "train_time_s": {}}
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.perf_counter()
        det = build_detector(a.detector, Xtr.shape[1], det_cfg)
        det.fit(Xtr, ytr, Xva, yva, cfg={**det_cfg, "seed": seed})
        det.save(detector_path(run, a.detector, seed))
        meta["train_time_s"][str(seed)] = time.perf_counter() - t0
        logging.info("seed %d trained in %.1fs", seed, meta["train_time_s"][str(seed)])
    save_json(run / "train_meta.json", meta)
    print(f"saved detectors to {run}")


if __name__ == "__main__":
    main()
