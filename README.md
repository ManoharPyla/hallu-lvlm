# hallu — calibrated, risk-guided selective hallucination correction for LVLMs

Research code. A token-level detector reads LVLM features (hidden states, log-prob,
entropy, optional image-attention mass). Its scores are calibrated, and a threshold is chosen
on held-out data, either by F1 or by distribution-free risk control. Spans above the
threshold go to a grounding check (COCO annotations or VQA). Only spans the check
contradicts are deleted or replaced, and every other part of the response is left as it was.

## Honest status

- **No real-model experiment has been run yet.** The only numbers in this repo come from the
  offline synthetic smoke test, which is marked *SYNTHETIC SMOKE TEST — NOT A BENCHMARK RESULT*.
- Every component (adapters, features, detector, calibration, policies, metrics, benchmarks,
  demo) is implemented and unit-tested on synthetic fixtures. See `docs/known_issues.md`.
- Novelty is limited. See `docs/phase1_plan.md` §4 and `docs/limitations.md`.

## Install

```bash
uv venv --python 3.11 && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"                 # core: numpy, torch, sklearn, pyyaml, matplotlib
uv pip install -e ".[model,data,demo]"     # transformers, huggingface_hub, gradio
```
pip also works: `pip install -e ".[dev,model,data,demo]"`.
**CUDA torch:** the default wheel may be CPU-only. Install a CUDA build that matches your
driver, e.g. `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`.

## Quick checks (offline, CPU, no weights)

```bash
python scripts/smoke_test.py      # < 1 min; writes experiments/results/smoke/
python -m pytest -q
```

## Full pipeline

```bash
python scripts/prepare_data.py --dry-run                       # instructions only
python scripts/prepare_data.py --download halloc pope coco_annotations
python scripts/prepare_data.py                                 # splits -> data/processed/
for s in train cal val test; do python scripts/extract_features.py --split $s; done
python scripts/train_detector.py --features-dir data/features/<model_tag> --detector mlp --seeds 0 1 2 --run-dir experiments/runs/main_mlp
python scripts/calibrate_detector.py --run-dir experiments/runs/main_mlp --features-dir data/features/<model_tag>
python scripts/run_inference.py --policy selective --split test --features-dir data/features/<model_tag> --run-dir experiments/runs/main_mlp --exp sel_main
python scripts/run_benchmarks.py --bench pope --exp bench_pope
python scripts/run_ablations.py --features-dir data/features/<model_tag>
python scripts/generate_report.py                              # paper/tables/generated/
python app/demo.py --run-dir experiments/runs/main_mlp         # local Gradio demo
```
`docs/reproduction.md` lists the exact command for each experiment.

## Layout
`src/hallu/` package · `configs/` YAML · `scripts/` CLI entry points · `app/` demo ·
`tests/` · `docs/` (architecture, dataset card, limitations, known issues, reproduction) ·
`experiments/` runs and results.
