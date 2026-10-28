"""Download (optional) and prepare datasets into leakage-free splits.

Nothing is downloaded unless --download is passed. Manual instructions:

  HalLoc (license UNSTATED, research use only):
      huggingface: https://huggingface.co/datasets/uunicee/HalLoc  -> data/raw/halloc/
  POPE (MIT): https://github.com/RUCAIBox/POPE  (output/coco/*.json, JSONL)  -> data/raw/pope/
  COCO 2014 annotations (CC-BY 4.0):
      http://images.cocodataset.org/annotations/annotations_trainval2014.zip -> data/raw/coco/
  COCO val2014 images (6.6 GB): http://images.cocodataset.org/zips/val2014.zip -> data/images/coco2014/
  HallusionBench (BSD-3): https://github.com/tianyi-lab/HallusionBench (json + images zip)
  MMHal-Bench (Apache-2.0): https://huggingface.co/datasets/Shengcao1006/MMHal-Bench

Keep large data OUTSIDE the OneDrive-synced folder (symlink data/ if needed).

    python scripts/prepare_data.py --dry-run
    python scripts/prepare_data.py --download halloc pope coco_annotations
    python scripts/prepare_data.py            # build splits from data/raw
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu.utils.config import load_config  # noqa: E402

logger = logging.getLogger("prepare_data")

POPE_GIT = "https://github.com/RUCAIBox/POPE.git"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
HALLOC_REPO = "uunicee/HalLoc"


def download(what: list[str], cfg: dict, dry: bool) -> None:
    raw = ROOT / "data" / "raw"
    if "halloc" in what:
        dst = ROOT / cfg["halloc"]["root"]
        print(f"[halloc] huggingface_hub.snapshot_download({HALLOC_REPO!r}, repo_type='dataset') -> {dst}")
        print("         NOTE: HalLoc license is unstated; research use only, do not redistribute.")
        if not dry:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                sys.exit("huggingface_hub missing: pip install -e .[data]")
            snapshot_download(HALLOC_REPO, repo_type="dataset", local_dir=str(dst),
                              allow_patterns=["*.json", "*.jsonl", "*.md"])
    if "pope" in what:
        tmp = raw / "_pope_git"
        dst = ROOT / cfg["pope"]["root"]
        print(f"[pope] git clone --depth 1 {POPE_GIT} {tmp}; copy output/coco/*.json -> {dst}")
        if not dry:
            if not tmp.exists():
                subprocess.run(["git", "clone", "--depth", "1", POPE_GIT, str(tmp)], check=True)
            dst.mkdir(parents=True, exist_ok=True)
            for f in (tmp / "output" / "coco").glob("*.json"):
                (dst / f.name).write_bytes(f.read_bytes())
    if "coco_annotations" in what:
        dst = raw / "coco"
        z = dst / "annotations_trainval2014.zip"
        print(f"[coco] {COCO_ANN_URL} -> {z} (~253 MB) and unzip")
        if not dry:
            dst.mkdir(parents=True, exist_ok=True)
            if not z.exists():
                urllib.request.urlretrieve(COCO_ANN_URL, z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dst)
    for other in set(what) - {"halloc", "pope", "coco_annotations"}:
        print(f"[{other}] no automatic download; see instructions at top of this script.")


def build(cfg: dict, sources: list[str], dry: bool) -> None:
    from hallu.data.adapters import get_adapter
    from hallu.data.splits import check_no_leakage, group_split, subsample
    from hallu.data.stats import dataset_stats, stats_to_markdown
    from hallu.data.validate import validate_samples
    from hallu.utils.io import save_json, write_jsonl

    roots = [str(ROOT / r) for r in cfg.get("image_roots", [])]
    out = ROOT / cfg["output"]["processed_dir"]
    ratios, seed = cfg["split"]["ratios"], int(cfg.get("seed", 0))
    malformed: dict[str, list] = {}
    md = []
    for src in sources:
        if src == "halloc":
            ad = get_adapter("halloc", image_roots=roots, keep_partial=cfg["halloc"].get("keep_partial", False))
            path = ROOT / cfg["halloc"]["root"]
            if not path.exists():
                logger.warning("HalLoc not found at %s; skipping", path); continue
            samples = ad.load(path, cfg["halloc"].get("tasks", ["vqa", "instruct", "caption"]))
            samples = subsample(samples, cfg["halloc"].get("max_samples"), seed)
            samples, bad = validate_samples(samples)
            malformed["halloc"] = ad.malformed + bad
            logger.info("halloc locate methods: %s", ad.locate_counts)
            splits = group_split(samples, ratios, seed)
            assert check_no_leakage(splits)
            stats = dataset_stats([s for v in splits.values() for s in v])
            md.append(stats_to_markdown(stats, "HalLoc"))
            if not dry:
                save_json(out / "halloc" / "stats.json", stats)
                for k, v in splits.items():
                    write_jsonl(out / "halloc" / f"{k}.jsonl", [s.to_dict() for s in v])
        elif src in ("pope", "hallusionbench", "mmhal"):
            path = ROOT / cfg[src]["root" if src == "pope" else "path"]
            if not path.exists():
                logger.warning("%s not found at %s; skipping", src, path); continue
            kw = {"include_no_visual": cfg[src].get("include_no_visual", True)} if src == "hallusionbench" else {}
            ad = get_adapter(src, image_roots=roots, **kw)
            items = ad.load(sorted(path.glob("*.json*")) if src == "pope" and path.is_dir() else path)
            malformed[src] = ad.malformed
            md.append(f"## {src}\n\nitems: {len(items)}, images resolved: "
                      f"{sum(1 for i in items if i.image_path)}\n")
            if not dry:
                write_jsonl(out / src / "all.jsonl", [i.to_dict() for i in items])
        else:
            logger.warning("unknown source %s", src)
    if dry:
        print("\n".join(md) or "no datasets found under data/raw (see instructions above)")
        return
    save_json(ROOT / cfg["output"]["malformed_report"],
              {k: {"n": len(v), "examples": v[:200]} for k, v in malformed.items()})
    stats_md = ROOT / cfg["output"]["stats_markdown"]
    stats_md.parent.mkdir(parents=True, exist_ok=True)
    stats_md.write_text("# Dataset statistics\n\n" + "\n".join(md), encoding="utf-8")
    (out / "dataset_stats.md").write_text("# Dataset statistics\n\n" + "\n".join(md), encoding="utf-8")
    print("\n".join(md) or "no datasets found under data/raw (see --dry-run instructions)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Prepare datasets")
    ap.add_argument("--config", default=str(ROOT / "configs" / "dataset.yaml"))
    ap.add_argument("--download", nargs="*", default=[],
                    help="halloc pope coco_annotations (others: manual)")
    ap.add_argument("--sources", nargs="*", default=["halloc", "pope", "hallusionbench", "mmhal"])
    ap.add_argument("--dry-run", action="store_true", help="print actions; download/write no data files")
    ap.add_argument("overrides", nargs="*", help="dotted overrides a.b=c")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.overrides)
    if a.dry_run:
        print(__doc__)
    if a.download:
        download(a.download, cfg, a.dry_run)
    build(cfg, a.sources, a.dry_run)


if __name__ == "__main__":
    main()
