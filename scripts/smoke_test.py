"""Fully OFFLINE end-to-end smoke test (no model weights, no network, CPU, < 1 min).

Pipeline exercised with SYNTHETIC data:
  synthetic Samples (COCO-like GT) -> synthetic FeatureRecords (signal-bearing features)
  -> train detector (MLP + logreg) -> calibrate (cal split) -> threshold (val split)
  -> detection/calibration metrics (test split)
  -> SelectiveCorrection with the trained detector + AnnotationBackend on synthetic COCO GT
  -> correction verification metrics.

Outputs go to experiments/results/smoke/ (or --out). Every artifact carries the banner
"SYNTHETIC SMOKE TEST — NOT A BENCHMARK RESULT". These numbers say nothing about any
real model or dataset; they only show the code paths run end to end.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hallu.calibration import get_calibrator, risk_control_threshold  # noqa: E402
from hallu.data.alignment import char_labels_to_token_labels  # noqa: E402
from hallu.data.schema import HallSpan, Sample  # noqa: E402
from hallu.data.splits import check_no_leakage, group_split  # noqa: E402
from hallu.evaluation.metrics import brier, detection_metrics, ece, select_threshold  # noqa: E402
from hallu.evaluation.report import to_markdown  # noqa: E402
from hallu.features.record import FeatureRecord, build_matrix  # noqa: E402
from hallu.grounding.checker import AnnotationBackend, GroundingChecker  # noqa: E402
from hallu.grounding.coco_objects import CocoGT  # noqa: E402
from hallu.mitigation.policies import NoMitigation, SelectiveCorrection  # noqa: E402
from hallu.mitigation.verification import correction_eval  # noqa: E402
from hallu.models.detector import build_detector  # noqa: E402
from hallu.utils.io import save_json, write_jsonl  # noqa: E402
from hallu.utils.seed import set_seed  # noqa: E402

BANNER = "SYNTHETIC SMOKE TEST — NOT A BENCHMARK RESULT"
logger = logging.getLogger("smoke_test")

CLASSES = ["person", "dog", "cat", "car", "chair", "bicycle", "bus", "horse", "bird", "cup",
           "bottle", "laptop", "umbrella", "bench", "truck", "boat"]
FILLER = ["near", "the", "with", "beside", "and", "under", "a", "next", "to", "on"]
FEATURES = ("hidden", "logprob", "entropy", "grounding")
H = 16


def make_world(n_images: int, rng: random.Random):
    """Synthetic COCO-like GT (instances + captions) and synthetic captions with
    hallucinated objects (an object absent from the image's GT)."""
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]
    cid = {c["name"]: c["id"] for c in cats}
    inst, caps, samples = [], [], []
    for i in range(n_images):
        img = str(100000 + i)
        present = rng.sample(CLASSES, 3)
        for c in present:
            inst.append({"image_id": int(img), "category_id": cid[c]})
        caps.append({"image_id": int(img), "caption": f"a {present[0]} and a {present[1]}"})
        for k in range(2):  # two responses per image -> tests group (image-level) splitting
            mentioned = rng.sample(present, 2)
            words, spans = [], []
            n_hall = rng.choice([0, 1, 1, 2])
            absent = [c for c in CLASSES if c not in present]
            objs = [(o, False) for o in mentioned] + [(o, True) for o in rng.sample(absent, n_hall)]
            rng.shuffle(objs)
            text = "There is"
            for j, (o, hall) in enumerate(objs):
                text += " " + rng.choice(["a", "the"]) + " "
                s = len(text)
                text += o
                if hall:
                    spans.append(HallSpan(s, s + len(o), o, "object"))
                if j < len(objs) - 1:
                    text += " " + rng.choice(FILLER)
            text += "."
            samples.append(Sample(sample_id=f"syn_{img}_{k}", image_id=img, image_path=None,
                                  prompt="Describe this image in detail.", response=text,
                                  spans=spans, source="synthetic"))
    gt = CocoGT({"categories": cats, "annotations": inst}, {"annotations": caps})
    return samples, gt


def tokenize(text: str):
    """Whitespace/punctuation tokenizer with char offsets (stands in for the LVLM tokenizer)."""
    import re
    toks, offs = [], []
    for m in re.finditer(r"\w+|[^\w\s]", text):
        toks.append(m.group(0))
        offs.append((m.start(), m.end()))
    return toks, offs


def make_record(s: Sample, rng: np.random.Generator) -> FeatureRecord:
    """Signal-bearing synthetic features: hallucinated tokens get lower logprob, higher
    entropy and a shifted hidden direction (plus noise so the task is not trivial)."""
    toks, offs = tokenize(s.response)
    labels = np.asarray(char_labels_to_token_labels(offs, s.spans), np.int64)
    T = len(toks)
    y = (labels == 1).astype(np.float32)
    hidden = rng.normal(0, 1, (T, H)).astype(np.float32)
    hidden[:, 0] += 1.5 * y
    hidden[:, 1] -= 1.0 * y
    logprob = (-0.5 - 1.2 * y + rng.normal(0, 0.8, T)).astype(np.float32)
    entropy = (1.0 + 1.0 * y + rng.normal(0, 0.7, T)).astype(np.float32)
    return FeatureRecord(s.sample_id, s.image_id, toks, offs, logprob, entropy,
                         hidden.astype(np.float16), np.zeros((T, 0), np.float32), labels,
                         meta={"synthetic": True, "split": s.split})


def record_matrix(r: FeatureRecord, use=FEATURES) -> np.ndarray:
    T = len(r.tokens)
    cols = [np.asarray(getattr(r, n), np.float32).reshape(T, -1) for n in use]
    return np.nan_to_num(np.concatenate(cols, 1))


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "smoke"))
    ap.add_argument("--n-images", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:  # Windows consoles (cp1252) cannot print the em dash in the banner
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    t_start = time.perf_counter()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {BANNER} ===")

    samples, gt = make_world(args.n_images, random.Random(args.seed))
    splits = group_split(samples, {"train": 0.6, "cal": 0.15, "val": 0.1, "test": 0.15}, seed=args.seed)
    assert check_no_leakage(splits), "image leakage across splits"
    nrng = np.random.default_rng(args.seed)
    recs = {k: [make_record(s, nrng) for s in v] for k, v in splits.items()}
    mats = {k: build_matrix(v, FEATURES) for k, v in recs.items()}
    Xtr, ytr, _, _ = mats["train"]
    Xcal, ycal, _, _ = mats["cal"]
    Xva, yva, _, _ = mats["val"]
    Xte, yte, _, _ = mats["test"]
    logger.info("tokens: train=%d cal=%d val=%d test=%d (pos rate train %.3f)",
                len(ytr), len(ycal), len(yva), len(yte), ytr.mean())

    det_cfg = {"mlp": dict(hidden_dims=[32, 16], epochs=15, batch_size=256, patience=5,
                           lr=3e-3, loss="weighted_bce", seed=args.seed),
               "logreg": dict(C=1.0, class_weight="balanced", seed=args.seed)}
    rows, detectors = {}, {}
    for name in ("mlp", "logreg"):
        det = build_detector(name, Xtr.shape[1], det_cfg[name])
        det.fit(Xtr, ytr, Xva, yva, cfg=det_cfg[name])
        detectors[name] = det
        lcal, lva, lte = det.predict_logits(Xcal), det.predict_logits(Xva), det.predict_logits(Xte)
        for cal_name in ("identity", "platt", "temperature", "isotonic"):
            cal = get_calibrator(cal_name).fit(lcal, ycal)  # fit on CAL split only
            pva, pte = cal.transform(lva), cal.transform(lte)
            thr = select_threshold(pva, yva, objective="f1")  # chosen on VAL only
            m = detection_metrics(yte, pte, thr)
            m.update(ece=ece(yte, pte), brier=brier(yte, pte), threshold=thr)
            rows[f"{name}+{cal_name}"] = _clean(m)

    # risk-controlled threshold (conformal / LTT-style) on the cal split
    best = detectors["mlp"]
    cal = get_calibrator("temperature").fit(best.predict_logits(Xcal), ycal)
    pcal = cal.transform(best.predict_logits(Xcal))
    pte = cal.transform(best.predict_logits(Xte))
    alpha, delta = 0.2, 0.1
    rc_thr = risk_control_threshold(pcal, ycal, alpha=alpha, delta=delta)
    pos = pte[yte == 1]
    rc = dict(alpha=alpha, delta=delta, threshold=float(rc_thr),
              test_miss_rate=float(np.mean(pos < rc_thr)) if pos.size else None,
              test_flag_rate=float(np.mean(pte >= rc_thr)))
    f1_thr = select_threshold(cal.transform(best.predict_logits(Xva)), yva, objective="f1")

    # selective correction on test responses with a COCO-annotation checker
    checker = GroundingChecker([AnnotationBackend(gt)])
    policies = {"none": NoMitigation(),
                "selective": SelectiveCorrection(best, cal, f1_thr, checker)}
    corr_rows, results_dump = {}, []
    test_samples = {s.sample_id: s for s in splits["test"]}
    for pname, pol in policies.items():
        evals = []
        for r in recs["test"]:
            s = test_samples[r.sample_id]
            ctx = dict(response=s.response, prompt=s.prompt, image_id=s.image_id, image=None,
                       offsets=r.offsets, X=record_matrix(r))
            res = pol.apply(ctx)
            ev = correction_eval(s.spans, res, gt.gt_objects(s.image_id))
            evals.append(ev)
            results_dump.append({"policy": pname, "sample_id": s.sample_id, **res.to_dict(),
                                 "eval": _clean(ev)})
        agg = {}
        for k in ("halluc_removed", "supported_preserved", "new_halluc", "unnecessary_changes",
                  "completeness", "edit_distance"):
            vals = [float(e[k]) for e in evals if e[k] is not None and not math.isnan(float(e[k]))]
            agg[k] = float(np.mean(vals)) if vals else None
        agg["n"] = len(evals)
        corr_rows[pname] = agg

    elapsed = time.perf_counter() - t_start
    metrics = {"banner": BANNER, "experiment": "smoke", "synthetic": True, "seed": args.seed,
               "n_samples": {k: len(v) for k, v in splits.items()},
               "n_tokens": {"train": int(len(ytr)), "cal": int(len(ycal)), "val": int(len(yva)),
                            "test": int(len(yte))},
               "tables": {"detection": rows, "correction": corr_rows},
               "risk_control": rc, "runtime_s": elapsed}
    save_json(out / "metrics.json", metrics)
    write_jsonl(out / "corrections.jsonl", results_dump)
    det_cols = ["precision", "recall", "f1", "auroc", "auprc", "ece", "brier", "threshold"]
    cor_cols = ["halluc_removed", "supported_preserved", "new_halluc", "unnecessary_changes",
                "completeness", "edit_distance", "n"]
    md = (f"# {BANNER}\n\nSynthetic data and synthetic features. These numbers do not describe any "
          f"model or dataset.\n\n## Detection (test split)\n\n{to_markdown(rows, det_cols)}\n\n"
          f"## Risk-controlled threshold (alpha={alpha}, delta={delta})\n\n```\n{json.dumps(rc, indent=2)}\n```\n\n"
          f"## Correction (test split)\n\n{to_markdown(corr_rows, cor_cols, row_header='policy')}\n\n"
          f"Runtime: {elapsed:.1f}s\n")
    (out / "report.md").write_text(md, encoding="utf-8")
    try:
        from hallu.evaluation.plots import plot_reliability
        plot_reliability(yte, pte, out / "reliability_smoke.png", title=f"{BANNER}")
    except ImportError as e:
        logger.warning("plots skipped (matplotlib unavailable: %s)", e)
    except Exception as e:  # plotting must never fail the smoke test
        logger.warning("plot failed: %s", e)
    print(md)
    print(f"=== {BANNER} === wrote {out} in {elapsed:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
