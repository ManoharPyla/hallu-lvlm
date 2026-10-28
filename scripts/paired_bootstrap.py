"""Paired image-level bootstrap: full detector versus hidden-only."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hallu.evaluation.metrics import auprc, auroc, f1_at
from hallu.features.record import build_matrix, load_records
from hallu.utils.io import load_json
from train_detector import load_detector
from hallu.calibration import get_calibrator


FEATURES_DIR = ROOT / "data/features_sdpa/llava-hf__llava-onevision-qwen2-0.5b-ov-hf"
HIDDEN_RUN = ROOT / "experiments/runs/sdpa_mlp_hidden_only"
FULL_RUN = ROOT / "experiments/runs/sdpa_mlp_multiseed"
OUT = ROOT / "experiments/results/paired_bootstrap_hidden_vs_full.json"


def paired_bootstrap(y, groups, p_hidden, p_full, threshold_hidden,
                     threshold_full, n_boot=1000, seed=2026):
    """Resample identical image IDs for both models; retain all tokens per image."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    p_hidden = np.asarray(p_hidden)
    p_full = np.asarray(p_full)

    if not (len(y) == len(groups) == len(p_hidden) == len(p_full)):
        raise ValueError("Labels, groups, and predictions are not aligned.")

    unique_groups = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)

    functions = {
        "auprc": lambda yy, ph, pf: (auprc(yy, pf), auprc(yy, ph)),
        "auroc": lambda yy, ph, pf: (auroc(yy, pf), auroc(yy, ph)),
        "f1": lambda yy, ph, pf: (
            f1_at(yy, pf, threshold_full),
            f1_at(yy, ph, threshold_hidden),
        ),
    }

    point = {}
    for metric, fn in functions.items():
        full_value, hidden_value = fn(y, p_hidden, p_full)
        point[metric] = {
            "hidden_only": float(hidden_value),
            "full": float(full_value),
            "difference_full_minus_hidden": float(full_value - hidden_value),
        }

    draws = {metric: [] for metric in functions}
    valid = {metric: 0 for metric in functions}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([idx_by_group[g] for g in sampled_groups])
            yy, ph, pf = y[idx], p_hidden[idx], p_full[idx]

            for metric, fn in functions.items():
                full_value, hidden_value = fn(yy, ph, pf)
                difference = full_value - hidden_value
                if np.isfinite(difference):
                    draws[metric].append(float(difference))
                    valid[metric] += 1

    result = {}
    for metric in functions:
        vals = np.asarray(draws[metric])
        result[metric] = {
            **point[metric],
            "difference_ci95_low": float(np.quantile(vals, 0.025)) if len(vals) else None,
            "difference_ci95_high": float(np.quantile(vals, 0.975)) if len(vals) else None,
            "valid_bootstrap_replicates": valid[metric],
            "requested_bootstrap_replicates": n_boot,
        }
    return result


def main():
    hidden_meta = load_json(HIDDEN_RUN / "train_meta.json")
    full_meta = load_json(FULL_RUN / "train_meta.json")
    hidden_thresholds = load_json(HIDDEN_RUN / "thresholds.json")
    full_thresholds = load_json(FULL_RUN / "thresholds.json")

    if hidden_meta["seeds"] != full_meta["seeds"]:
        raise ValueError("Seed lists differ between experiments.")

    records = load_records(FEATURES_DIR / "test")
    X_hidden, y_hidden, g_hidden, _ = build_matrix(records, tuple(hidden_meta["features"]))
    X_full, y_full, g_full, _ = build_matrix(records, tuple(full_meta["features"]))

    if not (np.array_equal(y_hidden, y_full) and np.array_equal(g_hidden, g_full)):
        raise ValueError("Test labels or image-group order differ between feature configurations.")

    seed_results = {}
    for seed in hidden_meta["seeds"]:
        hidden_detector = load_detector(HIDDEN_RUN, hidden_meta["detector"], seed)
        full_detector = load_detector(FULL_RUN, full_meta["detector"], seed)

        hidden_cal = get_calibrator("platt")
        hidden_cal.load(HIDDEN_RUN / f"calibrator_platt_seed{seed}.pkl")
        full_cal = get_calibrator("platt")
        full_cal.load(FULL_RUN / f"calibrator_platt_seed{seed}.pkl")

        p_hidden = hidden_cal.transform(hidden_detector.predict_logits(X_hidden))
        p_full = full_cal.transform(full_detector.predict_logits(X_full))

        th = float(hidden_thresholds[f"platt_seed{seed}"]["f1_val"])
        tf = float(full_thresholds[f"platt_seed{seed}"]["f1_val"])

        seed_results[str(seed)] = {
            "threshold_hidden_only": th,
            "threshold_full": tf,
            "n_tokens": int(len(y_hidden)),
            "n_images": int(len(np.unique(g_hidden))),
            "metrics": paired_bootstrap(
                y_hidden, g_hidden, p_hidden, p_full, th, tf,
                n_boot=1000, seed=2026 + int(seed),
            ),
        }
        print(f"\nSeed {seed} (thresholds hidden={th:.4f}, full={tf:.4f})")
        for metric, values in seed_results[str(seed)]["metrics"].items():
            print(
                f"{metric:6s}: delta={values['difference_full_minus_hidden']:+.4f}, "
                f"95% CI [{values['difference_ci95_low']}, "
                f"{values['difference_ci95_high']}] "
                f"(valid {values['valid_bootstrap_replicates']}/1000)"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison": "full detector minus hidden-only",
        "calibration": "Platt",
        "threshold_selection": "validation F1 threshold, separately per model and seed",
        "bootstrap_unit": "image_id; same sampled image IDs for both models",
        "bootstrap_replicates": 1000,
        "seeds": seed_results,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved paired comparison to: {OUT}")


if __name__ == "__main__":
    main()
