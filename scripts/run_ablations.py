"""Run ablations A-H from configs/evaluation.yaml by invoking the pipeline scripts.

    python scripts/run_ablations.py --features-dir data/features/<model_tag> --dry-run
    python scripts/run_ablations.py --features-dir data/features/<model_tag> --only A C D

NOTE on B: the comparison against the HalLoc authors' released detector is "not available
unless the HalLoc released detector is reproduced". In this repo B trains OUR detector on
hidden states only (a HalLoc-style input), which is NOT a reproduction of HalLoc's model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu.utils.config import load_config  # noqa: E402

PY = sys.executable
S = ROOT / "scripts"
B_NOTE = "not available unless HalLoc released detector reproduced (B here = our detector on hidden only)"


def plan(key: str, spec: dict, fd: str, detector: str, seeds: list[int]) -> list[list[str]]:
    run = str(ROOT / "experiments" / "runs" / f"abl_{key}")
    exp = f"ablation_{key}_{spec['name']}"
    seeds_s = [str(s) for s in seeds]
    base_run = str(ROOT / "experiments" / "runs" / "abl_D")  # full-feature detector reused by E-H
    train = lambda feats, r: [PY, str(S / "train_detector.py"), "--features-dir", fd, "--detector", detector,
                              "--seeds", *seeds_s, "--use", *feats, "--run-dir", r]
    calib = lambda r, methods, e: [PY, str(S / "calibrate_detector.py"), "--run-dir", r, "--features-dir", fd,
                                   "--methods", *methods, "--exp", e]
    if "features" in spec:
        return [train(spec["features"], run), calib(run, ["identity", "platt", "temperature", "isotonic"], exp)]
    if "calibrator" in spec:
        m = spec["calibrator"] if isinstance(spec["calibrator"], list) else [spec["calibrator"]]
        return [calib(base_run, m, exp)]
    if "threshold_objective" in spec:  # calibrate reports both f1 and risk_control thresholds
        return [calib(base_run, ["temperature"], exp)]
    if spec.get("policy") == "selective":
        return [[PY, str(S / "run_inference.py"), "--policy", "selective", "--features-dir", fd,
                 "--run-dir", base_run, "--no-checker", "--exp", exp]]
    raise ValueError(f"cannot plan ablation {key}: {spec}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "evaluation.yaml"))
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--detector", choices=["mlp", "logreg"], default="mlp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    abl = load_config(a.config)["ablations"]
    # D first: E-H reuse its detector
    keys = sorted(a.only or abl, key=lambda k: (k != "D", k))
    for k in keys:
        spec = abl[k]
        print(f"## {k}: {spec['name']}" + (f"  [{B_NOTE}]" if k == "B" else ""))
        for cmd in plan(k, spec, a.features_dir, a.detector, a.seeds):
            print("   ", " ".join(cmd))
            if not a.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
