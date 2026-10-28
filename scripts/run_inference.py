"""Apply a mitigation policy (none | whole | selective) to a labeled split.

Uses the teacher-forced FeatureRecords of the split (same tokens/offsets the detector saw),
a trained detector + calibrator + threshold, and a grounding checker (COCO annotations,
optionally VQA with the LVLM). Saves CorrectionResults as JSONL and verification metrics.

    python scripts/run_inference.py --policy selective --split test \
        --features-dir data/features/<model_tag> --run-dir experiments/runs/main_mlp \
        --calibrator temperature --threshold-type f1 --exp sel_main

'whole' needs the LVLM (it rewrites the entire response); 'selective' runs without an LVLM
(delete-only corrections) unless --use-lvlm is passed (enables VQA checks + span replacement).
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hallu.utils.config import load_config  # noqa: E402


def record_matrix(rec, use):
    import numpy as np
    T = len(rec.tokens)
    cols = [np.asarray(getattr(rec, n), np.float32).reshape(T, -1) for n in use]
    return np.nan_to_num(np.concatenate(cols, 1))


def mean_metrics(evals: list[dict]) -> dict:
    out = {}
    for k in ("halluc_removed", "supported_preserved", "new_halluc", "unnecessary_changes",
              "completeness", "edit_distance"):
        v = [float(e[k]) for e in evals if e.get(k) is not None and not math.isnan(float(e[k]))]
        out[k] = sum(v) / len(v) if v else None
    out["n"] = len(evals)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "evaluation.yaml"))
    ap.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    ap.add_argument("--dataset-config", default=str(ROOT / "configs" / "dataset.yaml"))
    ap.add_argument("--policy", choices=["none", "whole", "selective"], required=True)
    ap.add_argument("--split", default="test", choices=["cal", "val", "test"])
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "halloc"))
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrator", default="temperature")
    ap.add_argument("--threshold-type", choices=["f1", "risk_control"], default="f1")
    ap.add_argument("--no-checker", action="store_true", help="ablation H: no grounding check")
    ap.add_argument("--use-lvlm", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results-root", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.overrides)
    mcfg, dcfg = load_config(a.model_config), load_config(a.dataset_config)

    from hallu.data.schema import Sample
    from hallu.grounding.checker import AnnotationBackend, GroundingChecker, VQABackend
    from hallu.grounding.coco_objects import CocoGT
    from hallu.mitigation.policies import NoMitigation, SelectiveCorrection, WholeRewrite
    from hallu.mitigation.rewriter import Rewriter
    from hallu.mitigation.verification import correction_eval
    from hallu.utils.io import read_jsonl, save_json, write_jsonl

    samples = {d["sample_id"]: Sample.from_dict(d) for d in read_jsonl(Path(a.data_dir) / f"{a.split}.jsonl")}
    lvlm = None
    if a.use_lvlm or a.policy == "whole":
        from hallu.models.errors import ModelUnavailableError
        from hallu.models.lvlm import LVLM
        lc = mcfg["lvlm"]
        try:
            lvlm = LVLM(lc["model_id"], lc.get("device", "auto"), lc.get("dtype", "float16"),
                        lc.get("load_in_4bit", False), tuple(lc.get("hidden_layers", [-1])))
        except ModelUnavailableError as e:
            sys.exit(f"LVLM required for policy={a.policy}/--use-lvlm but unavailable: {e}")

    cc = dcfg["coco"]
    gt = None
    if Path(ROOT / cc["instances_json"]).exists():
        caps = ROOT / cc["captions_json"]
        gt = CocoGT(ROOT / cc["instances_json"], caps if caps.exists() else None)
    else:
        logging.warning("COCO annotations missing: annotation checker disabled, object metrics nan")

    order = list(samples)
    recs = {}
    if a.policy == "selective":
        from hallu.calibration.base import Calibrator
        from hallu.features.record import load_records
        from hallu.utils.io import load_json
        from train_detector import load_detector
        if not (a.features_dir and a.run_dir):
            sys.exit("--features-dir and --run-dir are required for the selective policy")
        meta = load_json(Path(a.run_dir) / "train_meta.json")
        det = load_detector(Path(a.run_dir), meta["detector"], a.seed)
        cal = Calibrator.load(Path(a.run_dir) / f"calibrator_{a.calibrator}_seed{a.seed}.pkl")
        thr = load_json(Path(a.run_dir) / "thresholds.json")[f"{a.calibrator}_seed{a.seed}"]
        thr = thr["f1_val" if a.threshold_type == "f1" else "risk_control_cal"]
        recs = {r.sample_id: r for r in load_records(Path(a.features_dir) / a.split)}
        order = [sid for sid in order if sid in recs]
        backends = []
        if gt is not None:
            backends.append(AnnotationBackend(gt))
        if lvlm is not None:
            backends.append(VQABackend(lvlm))
        checker = None if a.no_checker or not backends else GroundingChecker(backends)
        policy = SelectiveCorrection(det, cal, thr, checker, Rewriter(lvlm),
                                     hedge_unknown=cfg["grounding"].get("hedge_unknown", False))
    elif a.policy == "whole":
        policy = WholeRewrite(lvlm)
    else:
        policy = NoMitigation()
    order = order[: a.limit] if a.limit else order

    rows, evals = [], []
    for sid in order:
        s = samples[sid]
        ctx = dict(response=s.response, prompt=s.prompt, image_id=s.image_id, image=s.image_path)
        if sid in recs:
            r = recs[sid]
            ctx.update(offsets=r.offsets, X=record_matrix(r, tuple(meta["features"])))
            if r.meta.get("text_mismatch"):
                logging.warning("%s: tokenizer text != response; offsets may be shifted", sid)
        res = policy.apply(ctx)
        gobj = gt.gt_objects(s.image_id) if gt is not None and gt.has_image(s.image_id) else None
        ev = correction_eval(s.spans, res, gobj)
        evals.append(ev)
        rows.append({"sample_id": sid, "policy": a.policy, **res.to_dict(), "eval": ev})
    out = Path(a.results_root) / a.exp
    write_jsonl(out / "corrections.jsonl", rows)
    lat = [r["latency_s"] for r in rows]
    agg = {**mean_metrics(evals), "mean_latency_s": sum(lat) / len(lat) if lat else None}
    save_json(out / "metrics.json", {"experiment": a.exp, "synthetic": False, "split": a.split,
                                     "policy": a.policy, "calibrator": a.calibrator,
                                     "threshold_type": a.threshold_type, "checker": not a.no_checker,
                                     "tables": {"correction": {a.policy: agg}}})
    print(f"{a.policy}: {agg}\nwrote {out}")


if __name__ == "__main__":
    main()
