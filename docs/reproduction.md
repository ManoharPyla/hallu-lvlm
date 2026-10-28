# Reproduction

## Hardware
- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` (default): fits on a **4 GB GPU** in fp16 and is slow on CPU.
- `llava-hf/llava-1.5-7b-hf`: needs **≥16 GB GPU** in fp16, or 4-bit (`lvlm.load_in_4bit=true`, bitsandbytes, CUDA).
  4-bit on 4 GB is marginal.
- Switch models with a dotted override, e.g. `lvlm.model_id=llava-hf/llava-1.5-7b-hf`.
  Use a separate features directory for each model.

`F=data/features/llava-hf__llava-onevision-qwen2-0.5b-ov-hf` below.

| # | Experiment | Command |
|---|---|---|
| 0 | Offline smoke (synthetic) | `python scripts/smoke_test.py` |
| 0 | Unit tests | `python -m pytest -q` |
| 1 | Data | `python scripts/prepare_data.py --download halloc pope coco_annotations && python scripts/prepare_data.py` |
| 2 | Features | `for s in train cal val test; do python scripts/extract_features.py --split $s; done` |
| 3 | Detector (MLP, 3 seeds) | `python scripts/train_detector.py --features-dir $F --detector mlp --seeds 0 1 2 --run-dir experiments/runs/main_mlp` |
| 3b | Detector (logreg) | `python scripts/train_detector.py --features-dir $F --detector logreg --seeds 0 1 2 --run-dir experiments/runs/main_logreg` |
| 4 | Calibration + thresholds | `python scripts/calibrate_detector.py --run-dir experiments/runs/main_mlp --features-dir $F --exp main_mlp` (same for main_logreg) |
| 5a | No mitigation | `python scripts/run_inference.py --policy none --split test --exp corr_none` |
| 5b | Whole rewrite | `python scripts/run_inference.py --policy whole --split test --exp corr_whole` |
| 5c | Selective | `python scripts/run_inference.py --policy selective --split test --features-dir $F --run-dir experiments/runs/main_mlp --exp corr_selective` |
| 5d | Selective, risk-control threshold | add `--threshold-type risk_control --exp corr_selective_rc` |
| 6 | POPE | `python scripts/run_benchmarks.py --bench pope --exp bench_pope` |
| 7 | CHAIR (500 COCO val images) | `python scripts/run_benchmarks.py --bench chair --limit 500 --exp bench_chair` and add `--policy selective --run-dir experiments/runs/main_mlp --exp bench_chair_sel` |
| 8 | HallusionBench | `python scripts/run_benchmarks.py --bench hallusionbench --exp bench_hb` |
| 9 | MMHal (local judge, deviation) | `python scripts/run_benchmarks.py --bench mmhal --judge hf:<local model> --exp bench_mmhal` |
| 10 | Ablations A–H | `python scripts/run_ablations.py --features-dir $F` (use `--dry-run` to list the commands) |
| 11 | Tables / plots | `python scripts/generate_report.py` |

All seeds come from configs (`seed: 0`) and `--seeds`. Report mean ± std over seeds, with
image-level bootstrap CIs.
