# Internal interface contract (all modules MUST follow)

Package: `src/hallu` (import as `hallu.*`). Python >=3.10, type hints, `logging.getLogger(__name__)`.
Heavy deps (`transformers`, `gradio`, `PIL`) imported lazily inside functions so unit tests run with only numpy/torch/sklearn/pyyaml.
No network access in tests. Tests use synthetic fixtures only.

## Core dataclasses — `hallu/data/schema.py`
```python
@dataclass
class HallSpan:            # character-level span in response text
    start: int; end: int   # [start, end) char offsets
    text: str
    htype: str             # "object"|"attribute"|"relationship"|"scene"|"unknown"

@dataclass
class Sample:
    sample_id: str
    image_id: str          # used for leakage-free splitting
    image_path: str | None
    prompt: str
    response: str          # text whose tokens are labeled
    spans: list[HallSpan]  # hallucinated spans (empty = clean)
    source: str            # dataset name
    split: str | None = None
    meta: dict = field(default_factory=dict)

@dataclass
class YesNoItem:           # POPE / HallusionBench
    qid: str; image_id: str; image_path: str | None; question: str; label: str  # "yes"/"no"
    source: str; subset: str = ""; meta: dict = field(default_factory=dict)
```
Serialization: `to_dict()` / `from_dict()` on each; JSONL I/O in `hallu/utils/io.py`: `read_jsonl(path)->list[dict]`, `write_jsonl(path, rows)`, `save_json`, `load_json`.

## Alignment — `hallu/data/alignment.py`
`char_labels_to_token_labels(offsets: list[tuple[int,int]], spans: list[HallSpan], special_mask: list[bool]|None=None, min_overlap: float=0.0) -> list[int]`
 token label 1 if its char range overlaps any span (>min_overlap fraction), else 0; special/empty-offset tokens get -100 (ignored). Pure whitespace/punct-only tokens inside a span get label of span; outside: 0.
`word_index_to_char_spans(text, word_indices, htype) -> list[HallSpan]` (whitespace word split; HalLoc uses word_index).
`group_tokens_to_spans(offsets, flags, text, max_gap=1) -> list[tuple[int,int]]` merges flagged tokens into char spans, expanded to whole words.

## Features — `hallu/features/`
`FeatureRecord` (dataclass, `features/record.py`): `sample_id: str, image_id: str, tokens: list[str], offsets: list[tuple[int,int]], logprob: np.ndarray[T], entropy: np.ndarray[T], hidden: np.ndarray[T,H] (float16 ok), grounding: np.ndarray[T,G] (may be G=0), labels: np.ndarray[T] int (−100 ignore)`.
Save/load: `save_records(path.npz-dir, records)`, `load_records(path)`.
`build_matrix(records, use=("hidden","logprob","entropy","grounding")) -> (X: np.ndarray[N,D], y: np.ndarray[N], groups: np.ndarray[N] of image_id, index: list[(rec_i, tok_i)])` drops label −100.
Scalar features derived from logits: `token_stats(logits[T,V], token_ids[T]) -> dict(logprob, entropy, margin, rank)`.

## LVLM wrapper — `hallu/models/lvlm.py`
`class LVLM: __init__(model_id, device="auto", dtype="float16", load_in_4bit=False, hidden_layers=(-1,))`
`generate(image, prompt, max_new_tokens=128) -> GenOutput(text, token_ids, tokens, offsets, logits[T,V] or stats, hidden[T,H], image_attn_mass[T] | None, latency_s)`
`teacher_force(image, prompt, response) -> GenOutput` (same fields, for labeled responses from HalLoc).
`answer(image, question) -> str` used by verification. Raises `ModelUnavailableError` with setup hint if weights/transformers missing.

## Detector — `hallu/models/detector.py`
`class RiskMLP(nn.Module)`, `class LogRegDetector` (sklearn). Common API via `TokenDetector` protocol:
`fit(X_train, y_train, X_val=None, y_val=None, cfg: dict) -> self`; `predict_logits(X)->np.ndarray[N]`; `predict_proba(X)`; `save(path)`; `load(path)` classmethod.
Losses: "bce" or "weighted_bce" (pos_weight = neg/pos).

## Calibration — `hallu/calibration/`
`Calibrator` base: `fit(logits, y)`, `transform(logits)->probs`, `save/load`. Implement `Identity` (sigmoid), `Platt`, `Temperature`, `Isotonic`; factory `get_calibrator(name)`.
Plus `conformal.py`: `risk_control_threshold(probs_cal, y_cal, alpha, delta=0.1) -> float` (threshold so that hallucinated-token miss rate (FNR among positives) ≤ alpha with Hoeffding-UCB / Learn-Then-Test on a grid).

## Metrics — `hallu/evaluation/metrics.py`
`detection_metrics(y, p, thr=0.5) -> dict(precision, recall, f1, auroc, auprc, pos_rate)`;
`ece(y,p,n_bins=15, strategy="uniform")`, `brier(y,p)`, `reliability_curve(y,p,n_bins)->dict`, `risk_coverage(y,p)->dict(coverage, risk, aurc)`;
`span_iou(pred_spans, gold_spans)`; `pope_metrics(preds, labels)->acc,precision,recall,f1,yes_ratio`; `chair(caption_objects, gt_objects)->CHAIR_s, CHAIR_i`;
`bootstrap_ci(fn, *arrays, n=1000, seed=0)`.

## Grounding — `hallu/grounding/`
`coco_objects.py`: COCO 80-class synonym table + `extract_objects(text)->list[(word, coco_class, char_span)]`, `CocoGT(instances_json, captions_json)` -> `gt_objects(image_id)->set[str]`.
`checker.py`: `GroundingVerdict` enum {SUPPORTED, CONTRADICTED, UNKNOWN}; `GroundingChecker.check(span_text, image_id|None, image|None) -> (verdict, evidence:str)`; backends: `AnnotationBackend` (COCO GT: object mentioned but not in GT instances AND not in any GT caption → CONTRADICTED only for COCO classes; else UNKNOWN) and `VQABackend(lvlm)` ("Is there a {obj} in the image?").
`source.py`: `SourceLabel` {VISUAL_UNDERGROUNDING, LANGUAGE_PRIOR, CROSS_MODAL_CONFLICT, UNKNOWN}; heuristic `estimate_source(features)` returns UNKNOWN unless rules defined in docs fire; documented as heuristic, not causal.

## Mitigation — `hallu/mitigation/`
`policies.py`: `Policy` base `.apply(sample_ctx) -> CorrectionResult(original, corrected, edits:list[Edit], decisions:list[dict], latency_s)`; `Edit(start,end,old,new,reason)`.
Policies: `NoMitigation`, `WholeRewrite(lvlm)`, `SelectiveCorrection(detector, calibrator, threshold, checker, rewriter)`.
Selective: flagged spans (calibrated risk ≥ thr) → checker; CONTRADICTED → remove/replace span minimally (delete object phrase w/ article cleanup, or replace with VQA-supported alternative if available); UNKNOWN → keep (or hedge if cfg.hedge_unknown); SUPPORTED → keep.
`minimal_edit.py`: `remove_span(text, start, end)` grammar-light cleanup; `char_edit_distance`, `diff_spans(a,b)`.
`verification.py`: `correction_eval(original_spans_gold, result, gold_objects) -> dict(halluc_removed, supported_preserved, new_halluc, unnecessary_changes, completeness, edit_distance)`; unchanged incorrect response gets halluc_removed=0 (no reward).

## Config
YAML in `configs/`; loaded by `hallu/utils/config.py:load_config(path, overrides:list[str]) -> dict` with `a.b=c` dotted overrides; `set_seed(seed)` in `hallu/utils/seed.py`.
