# Known issues / incomplete components

1. **No real-model run has been done.** No features have been extracted with any LVLM and no
   detector has been trained on HalLoc. The only results are from the synthetic smoke test.
2. **HalLoc schema nesting is unverified.** `hallu.data.adapters.halloc` handles the annotation
   structure heuristically (several key names, word_index / char_index / name search). Verify it
   against the downloaded JSON and check `data/processed/malformed.json` and the locate-method counts.
3. **Source classifier is heuristic only.** `hallu.grounding.source.estimate_source` uses rules
   and returns UNKNOWN unless a rule fires. It is not validated and not causal.
4. **No released HalLoc detector.** Ablation B is "not available unless the HalLoc released
   detector is reproduced". The B row is our own detector trained on hidden states only.
5. **MMHal judge substitute.** The official protocol uses GPT-4. This repo uses `--judge none`
   or a local `hf:<model>` judge, so the scores are not comparable to published numbers.
6. **CHAIR synonym list is approximate.** The COCO synonym table in `hallu.grounding.coco_objects`
   is not the exact list from the original CHAIR paper.
7. Teacher-forced decoding can differ from the dataset text (`text_mismatch`). Those records are
   counted and their offsets may be shifted.
8. HallusionBench reports per-question accuracy only. qAcc and fAcc are not implemented.
9. Ablations A–H are a working definition (`configs/evaluation.yaml`) that has not been confirmed.
