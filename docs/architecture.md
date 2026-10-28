# Architecture

```
data/raw ──adapters──> Sample (char spans) ──group_split(image_id)──> train/cal/val/test.jsonl
                                   │
                    LVLM.teacher_force (features/extract.py)
                                   ▼
             FeatureRecord [T tokens: hidden, logprob, entropy, grounding, labels]
                                   │ build_matrix
        train ─> detector (RiskMLP | LogRegDetector)        (early stopping on val)
        cal   ─> calibrator (identity|platt|temperature|isotonic) + risk-control threshold
        val   ─> F1 threshold
        test  ─> detection metrics + image-level bootstrap CI
                                   │
   SelectiveCorrection: risk ≥ thr ─> flagged word spans ─> GroundingChecker
        (AnnotationBackend: COCO GT | VQABackend: LVLM yes/no)
        CONTRADICTED → delete / LVLM replacement · UNKNOWN → keep (or hedge) · SUPPORTED → keep
                                   ▼
             CorrectionResult ─> verification.correction_eval
```

| Module | Role |
|---|---|
| `hallu.data` | schema, adapters (HalLoc, POPE, COCO, HallusionBench, MMHal), alignment, splits, stats, validation |
| `hallu.models` | `LVLM` wrapper (HF transformers, lazy import), detectors |
| `hallu.features` | token stats, grounding features, FeatureRecord I/O |
| `hallu.calibration` | calibrators, conformal / Learn-Then-Test-style threshold |
| `hallu.grounding` | COCO object extraction, checkers, heuristic source labels |
| `hallu.mitigation` | policies (none / whole rewrite / selective), minimal edits, verification |
| `hallu.evaluation` | metrics, POPE, CHAIR, efficiency, report tables, plots |

The scripts in `scripts/` are thin CLIs over these modules. Split discipline: calibrators are fitted
on `cal`, F1 thresholds are chosen on `val`, and nothing is tuned on `test`.
The detector, calibrator and threshold all depend on the model that produced the features,
so each LVLM needs its own trained set.
