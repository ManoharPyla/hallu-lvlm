"""Run external hallucination benchmarks with a local LVLM.

    python scripts/run_benchmarks.py --bench pope --limit 500 --exp bench_pope
    python scripts/run_benchmarks.py --bench chair --limit 500 --exp bench_chair [--policy selective --run-dir ...]
    python scripts/run_benchmarks.py --bench hallusionbench --exp bench_hb
    python scripts/run_benchmarks.py --bench mmhal --judge none --exp bench_mmhal

MMHal-Bench DEVIATION: the official MMHal score uses GPT-4 as the judge. This project is
local-only, so the judge is pluggable: `--judge none` (answers saved, no score) or
`--judge hf:<local text model id>` (a local open model prompted with an adapted rubric).
Scores from a local judge are NOT comparable to published GPT-4-judged MMHal numbers and
must be reported as such.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hallu.utils.config import load_config  # noqa: E402

JUDGE_DEVIATION = ("MMHal judge is a LOCAL substitute for GPT-4 (deviation from the official "
                   "protocol); scores are not comparable to published MMHal numbers.")
MMHAL_RUBRIC = (
    "You are grading an answer to a question about an image. You cannot see the image; use the "
    "image description and the reference answer.\nImage contents: {content}\nQuestion: {q}\n"
    "Reference answer: {gt}\nModel answer: {a}\n"
    "Rate 0-6: 6 = informative and no hallucination, 3 = adequate but some hallucination, "
    "0 = mostly hallucinated. Reply with only the integer."
)


def gen_features(out, use):
    """Detector input from a GenOutput (same feature groups as training)."""
    import numpy as np
    from hallu.features.grounding_features import grounding_features
    T = len(out.token_ids)
    parts = {"hidden": np.asarray(out.hidden, np.float32).reshape(T, -1),
             "logprob": np.asarray(out.logits["logprob"], np.float32).reshape(T, 1),
             "entropy": np.asarray(out.logits["entropy"], np.float32).reshape(T, 1),
             "grounding": grounding_features(out, T)}
    return np.nan_to_num(np.concatenate([parts[u] for u in use], 1))


class LocalTextJudge:
    def __init__(self, model_id: str):
        try:
            from transformers import pipeline
        except ImportError as e:
            sys.exit(f"local judge needs transformers: {e}")
        self.pipe = pipeline("text-generation", model=model_id, device_map="auto")

    def score(self, q, a, gt, content) -> float | None:
        msg = MMHAL_RUBRIC.format(q=q, a=a, gt=gt, content=content)
        out = self.pipe(msg, max_new_tokens=4, do_sample=False, return_full_text=False)[0]["generated_text"]
        m = re.search(r"[0-6]", out)
        return float(m.group(0)) if m else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", choices=["pope", "chair", "hallusionbench", "mmhal"], required=True)
    ap.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    ap.add_argument("--dataset-config", default=str(ROOT / "configs" / "dataset.yaml"))
    ap.add_argument("--policy", choices=["none", "selective"], default="none",
                    help="selective applies detector-gated correction to generated text (chair/mmhal)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--calibrator", default="temperature")
    ap.add_argument("--threshold-type", choices=["f1", "risk_control"], default="f1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default="none", help="none | hf:<local model id>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results-root", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mcfg = load_config(a.model_config, a.overrides)
    dcfg = load_config(a.dataset_config)

    from hallu.data.adapters import get_adapter
    from hallu.evaluation.efficiency import LatencyTimer
    from hallu.models.errors import ModelUnavailableError
    from hallu.models.lvlm import LVLM
    from hallu.utils.io import save_json, write_jsonl

    roots = [str(ROOT / r) for r in dcfg.get("image_roots", [])]
    lc = mcfg["lvlm"]
    try:
        lvlm = LVLM(lc["model_id"], lc.get("device", "auto"), lc.get("dtype", "float16"),
                    lc.get("load_in_4bit", False), tuple(lc.get("hidden_layers", [-1])),
                    lc.get("output_attentions", False))
    except ModelUnavailableError as e:
        sys.exit(f"LVLM unavailable: {e}")
    timer = LatencyTimer()
    out_dir = Path(a.results_root) / a.exp
    res: dict = {"experiment": a.exp, "synthetic": False, "bench": a.bench, "model_id": lc["model_id"],
                 "policy": a.policy}
    rows: list[dict] = []

    policy, use = None, None
    if a.policy == "selective":
        from hallu.calibration.base import Calibrator
        from hallu.grounding.checker import GroundingChecker, VQABackend
        from hallu.mitigation.policies import SelectiveCorrection
        from hallu.mitigation.rewriter import Rewriter
        from hallu.utils.io import load_json
        from train_detector import load_detector
        run = Path(a.run_dir)
        meta = load_json(run / "train_meta.json")
        use = tuple(meta["features"])
        thr = load_json(run / "thresholds.json")[f"{a.calibrator}_seed{a.seed}"]
        policy = SelectiveCorrection(load_detector(run, meta["detector"], a.seed),
                                     Calibrator.load(run / f"calibrator_{a.calibrator}_seed{a.seed}.pkl"),
                                     thr["f1_val" if a.threshold_type == "f1" else "risk_control_cal"],
                                     GroundingChecker([VQABackend(lvlm)]), Rewriter(lvlm))

    def respond(image, prompt, image_id=None):
        with timer:
            g = lvlm.generate(image, prompt, lc.get("max_new_tokens", 128))
            if policy is None:
                return g.text, None
            r = policy.apply(dict(response=g.text, prompt=prompt, image=image, image_id=image_id,
                                  offsets=g.offsets, X=gen_features(g, use)))
        return r.corrected, r.to_dict()

    if a.bench in ("pope", "hallusionbench"):
        from hallu.evaluation.pope_eval import evaluate_pope, parse_yes_no
        if a.bench == "pope":
            p = ROOT / dcfg["pope"]["root"]
            items = get_adapter("pope", image_roots=roots).load(sorted(p.glob("*.json*")) if p.is_dir() else p)
        else:
            items = get_adapter("hallusionbench", image_roots=roots,
                                include_no_visual=dcfg["hallusionbench"].get("include_no_visual", True)
                                ).load(ROOT / dcfg["hallusionbench"]["path"])
        items = [i for i in items if i.image_path or not i.meta.get("visual_input", True)]
        items = items[: a.limit] if a.limit else items
        answers = {}

        def answer_fn(it):
            with timer:
                ans = lvlm.answer(it.image_path, it.question)
            answers[it.qid] = ans
            rows.append({"qid": it.qid, "subset": it.subset, "label": it.label, "answer": ans,
                         "pred": parse_yes_no(ans)})
            return ans
        res["metrics"] = evaluate_pope(items, answer_fn)
        if a.bench == "hallusionbench":
            res["note"] = ("Per-question accuracy only; official question-pair/figure accuracy "
                           "(qAcc/fAcc) not computed.")
    elif a.bench == "chair":
        from hallu.evaluation.chair import evaluate_chair
        from hallu.grounding.coco_objects import CocoGT
        cc = dcfg["coco"]
        gt = CocoGT(ROOT / cc["instances_json"], None)
        caps_samples = get_adapter("coco", image_roots=roots).load(ROOT / cc["captions_json"])
        seen, imgs = set(), []
        for s in sorted(caps_samples, key=lambda s: s.image_id):
            if s.image_path and s.image_id not in seen:
                seen.add(s.image_id); imgs.append(s)
        imgs = imgs[: a.limit or 500]
        captions = {}
        for s in imgs:
            text, dec = respond(s.image_path, "Describe this image in detail.", s.image_id)
            captions[s.image_id] = text
            rows.append({"image_id": s.image_id, "caption": text, "correction": dec})
        r = evaluate_chair(captions, gt.gt_objects)
        res["metrics"] = {k: v for k, v in r.items() if k != "per_image"}
        res["note"] = "CHAIR uses an approximate COCO synonym list (hallu.grounding.coco_objects)."
    else:  # mmhal
        samples = get_adapter("mmhal", image_roots=roots).load(ROOT / dcfg["mmhal"]["path"])
        samples = [s for s in samples if s.image_path]
        samples = samples[: a.limit] if a.limit else samples
        judge = LocalTextJudge(a.judge.split(":", 1)[1]) if a.judge.startswith("hf:") else None
        scores = []
        for s in samples:
            text, dec = respond(s.image_path, s.prompt, s.image_id)
            sc = judge.score(s.prompt, text, s.meta.get("gt_answer"), s.meta.get("image_content")) if judge else None
            if sc is not None:
                scores.append(sc)
            rows.append({"sample_id": s.sample_id, "question": s.prompt, "answer": text, "score": sc,
                         "question_type": s.meta.get("question_type"), "correction": dec})
        res["metrics"] = {"n": len(samples), "judge": a.judge,
                          "mean_score": sum(scores) / len(scores) if scores else None,
                          "hallucination_rate": (sum(1 for x in scores if x < 3) / len(scores)) if scores else None}
        res["deviation"] = JUDGE_DEVIATION
    res["latency"] = timer.summary()
    write_jsonl(out_dir / "outputs.jsonl", rows)
    save_json(out_dir / "metrics.json", res)
    print(f"wrote {out_dir}: {res.get('metrics')}")


if __name__ == "__main__":
    main()
