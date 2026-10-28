"""Aggregate paired image-level bootstrap across five matched detector seeds."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hallu.calibration import get_calibrator
from hallu.evaluation.metrics import auprc
from hallu.features.record import build_matrix, load_records
from hallu.utils.io import load_json
from train_detector import load_detector

FEATURES_DIR = ROOT / "data/features_sdpa/llava-hf__llava-onevision-qwen2-0.5b-ov-hf"
HIDDEN_RUN = ROOT / "experiments/runs/sdpa_mlp_hidden_only"
FULL_RUN = ROOT / "experiments/runs/sdpa_mlp_multiseed"
OUT = ROOT / "experiments/results/aggregate_paired_bootstrap.json"


def main(n_boot=1000, random_seed=731):
    hmeta = load_json(HIDDEN_RUN / "train_meta.json")
    fmeta = load_json(FULL_RUN / "train_meta.json")
    ht = load_json(HIDDEN_RUN / "thresholds.json")
    ft = load_json(FULL_RUN / "thresholds.json")

    seeds = hmeta["seeds"]
    if seeds != fmeta["seeds"]:
        raise ValueError("The two experiments do not have matching seed lists.")

    records = load_records(FEATURES_DIR / "test")
    Xh, y, groups, _ = build_matrix(records, tuple(hmeta["features"]))
    Xf, yf, gf, _ = build_matrix(records, tuple(fmeta["features"]))

    if not (np.array_equal(y, yf) and np.array_equal(groups, gf)):
        raise ValueError("Test tokens, labels, or image-group order do not match.")

    unique_groups = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}

    # Predictions are computed once; each bootstrap replicate resamples images.
    predictions = {}
    point_deltas = []

    for seed in seeds:
        hd = load_detector(HIDDEN_RUN, hmeta["detector"], seed)
        fd = load_detector(FULL_RUN, fmeta["detector"], seed)

        hc = get_calibrator("platt")
        hc.load(HIDDEN_RUN / f"calibrator_platt_seed{seed}.pkl")
        fc = get_calibrator("platt")
        fc.load(FULL_RUN / f"calibrator_platt_seed{seed}.pkl")

        ph = hc.transform(hd.predict_logits(Xh))
        pf = fc.transform(fd.predict_logits(Xf))

        predictions[seed] = (ph, pf)
        point_deltas.append(auprc(y, pf) - auprc(y, ph))

    point_mean = float(np.mean(point_deltas))
    rng = np.random.default_rng(random_seed)
    bootstrap_means = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([idx_by_group[g] for g in sampled])

            seed_deltas = []
            for seed in seeds:
                ph, pf = predictions[seed]
                delta = auprc(y[idx], pf[idx]) - auprc(y[idx], ph[idx])
                if np.isfinite(delta):
                    seed_deltas.append(float(delta))

            if len(seed_deltas) == len(seeds):
                bootstrap_means.append(float(np.mean(seed_deltas)))

    if not bootstrap_means:
        raise RuntimeError("No valid bootstrap replicates; inspect test labels and predictions.")

    lo, hi = np.quantile(bootstrap_means, [0.025, 0.975])
    result = {
        "comparison": "full detector minus hidden-only",
        "metric": "AUPRC",
        "calibration": "Platt",
        "threshold_policy": "not applicable to AUPRC; scores evaluated across thresholds",
        "aggregation": "mean paired AUPRC difference across matched seeds within each image bootstrap replicate",
        "bootstrap_unit": "image_id; same sampled images for both models and every seed",
        "n_images": int(len(unique_groups)),
        "n_tokens": int(len(y)),
        "seeds": [int(s) for s in seeds],
        "bootstrap_replicates_requested": n_boot,
        "bootstrap_replicates_valid": len(bootstrap_means),
        "per_seed_point_differences": {
            str(s): float(point_deltas[i]) for i, s in enumerate(seeds)
        },
        "mean_paired_auprc_difference": point_mean,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Aggregate paired image-level bootstrap")
    print(f"Images: {len(unique_groups)} | Tokens: {len(y)} | Seeds: {seeds}")
    print("Per-seed AUPRC differences:", result["per_seed_point_differences"])
    print(f"Mean paired AUPRC difference: {point_mean:+.4f}")
    print(f"95% bootstrap CI: [{lo:+.4f}, {hi:+.4f}]")
    print(f"Valid bootstrap replicates: {len(bootstrap_means)}/{n_boot}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
