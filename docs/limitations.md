# Limitations

- **Novelty is narrow.** Token detectors with calibration (MetaToken), hidden-state localizers
  (HalLoc) and post-hoc correction (Woodpecker, LURE) already exist. Any contribution would have
  to come from risk-controlled thresholds plus minimal-edit correction with abstention. That
  claim is untested.
- **Model dependence.** Features, detectors, calibrators and thresholds are specific to one LVLM.
  Results from the local 0.5B model do not carry over to LLaVA-1.5-7B.
- **Risk-control guarantee.** The miss-rate bound assumes the cal and test tokens are
  exchangeable. Tokens within one response are correlated, so the guarantee is approximate. It
  does not hold under dataset shift (for example, HalLoc → POPE/MMHal).
- **Grounding checks.** The COCO annotation backend covers only the 80 COCO classes. A missing
  annotation does not prove the object is absent, which is why caption-only mentions return
  UNKNOWN. The VQA backend uses the same LVLM and can repeat its errors.
- **Correction quality.** Without an LVLM the system can only delete text. With a small LVLM,
  replacements may be poor. Attribute, relation and scene errors are flagged but rarely fixed.
- **Source attribution** (under-grounding / language prior / conflict) comes from heuristic
  rules. It is not causal and has no ground-truth labels.
- **Benchmarks.** The MMHal judge is a local substitute for GPT-4. HallusionBench is reported as
  per-question accuracy only. The CHAIR synonym list is approximate.
- **Compute.** The 4 GB GPU cannot run the 7B target model in fp16.
