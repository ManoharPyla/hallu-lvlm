"""Teacher-force an LVLM over prepared Samples and save token FeatureRecords.

    python scripts/extract_features.py --split train --limit 2000
    python scripts/extract_features.py --split test lvlm.model_id=llava-hf/llava-1.5-7b-hf lvlm.load_in_4bit=true

Output: data/features/<model_tag>/<split>/ (index.json + npz per record) and counts.json.
Features from different LVLMs are NOT interchangeable; keep one directory per model.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hallu.utils.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu.data.schema import Sample
from hallu.utils.config import load_config  # noqa: E402


def model_tag(model_id: str) -> str:
    return model_id.replace("/", "__")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "model.yaml"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "halloc"))
    ap.add_argument("--input-file", default=None,
                help="Optional explicit JSONL input file; overrides --data-dir and --split.")
    ap.add_argument("--split", default="train", choices=["train", "cal", "val", "test"])
    ap.add_argument("--out-root", default=str(ROOT / "data" / "features"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-missing-image", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.overrides)

    from hallu.data.schema import Sample
    from hallu.features.extract import extract_features
    from hallu.features.record import save_records
    from hallu.models.errors import ModelUnavailableError
    from hallu.models.lvlm import LVLM
    from hallu.utils.io import read_jsonl, save_json

    input_path = Path(a.input_file) if a.input_file else Path(a.data_dir) / f"{a.split}.jsonl"
    samples = [Sample.from_dict(d) for d in read_jsonl(input_path)]
    logging.info("reading samples from %s", input_path)
    if not a.allow_missing_image:
        n0 = len(samples)
        samples = [s for s in samples if s.image_path]
        logging.info("kept %d/%d samples with a resolved image", len(samples), n0)
    samples = samples[: a.limit] if a.limit else samples
    lc = cfg["lvlm"]
    try:
        lvlm = LVLM(lc["model_id"], lc.get("device", "auto"), lc.get("dtype", "float16"),
                    lc.get("load_in_4bit", False), tuple(lc.get("hidden_layers", [-1])),
                    lc.get("output_attentions", False))
    except ModelUnavailableError as e:
        sys.exit(f"LVLM unavailable: {e}\nInstall: pip install -e .[model] (and CUDA torch).")
    records, counts = extract_features(lvlm, samples)
    out = Path(a.out_root) / model_tag(lc["model_id"]) / a.split
    save_records(out, records)
    save_json(out / "counts.json", {**counts, "model_id": lc["model_id"], "split": a.split})
    print(f"saved {len(records)} records to {out}; counts={counts}")


if __name__ == "__main__":
    main()
