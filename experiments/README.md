# experiments/

- `runs/<name>/`: trained artifacts: `detector_seed<k>.(pt|pkl)`, `train_meta.json`,
  `calibrator_<method>_seed<k>.pkl`, `thresholds.json`.
- `results/<exp>/metrics.json`: metrics for one experiment, plus `corrections.jsonl` /
  `outputs.jsonl` with per-sample outputs.
- `results/smoke/`: output of `python scripts/smoke_test.py`. **SYNTHETIC SMOKE TEST — NOT A
  BENCHMARK RESULT.** Do not cite these numbers.

`python scripts/generate_report.py` collects every `results/*/metrics.json` into
`paper/tables/generated/` as Markdown and LaTeX. Missing values render as `--`.
