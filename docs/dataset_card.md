# Dataset card

No data is bundled. `scripts/prepare_data.py --dry-run` prints the download instructions.

| Dataset | Source | License | Use here | Notes |
|---|---|---|---|---|
| HalLoc | HF `uunicee/HalLoc` (CVPR'25) | **Unstated**: research use only; contact the authors before redistributing | detector train/cal/val/test | ~156k samples with token-level Object/Attribute/Relationship/Scene labels (word_index). Images come from VG + COCO2014 and are not included. The annotation nesting is handled heuristically (see known_issues). |
| COCO 2014 | cocodataset.org | Annotations CC-BY 4.0; images under Flickr terms | images, CHAIR GT, annotation checker | val2014 is 6.6 GB; annotations ~253 MB |
| POPE | github.com/RUCAIBox/POPE | MIT | yes/no object probing | uses COCO val2014 images |
| HallusionBench | github.com/tianyi-lab/HallusionBench | BSD-3-Clause | yes/no visual/knowledge questions | images zip from Google Drive; this repo reports per-question accuracy only |
| MMHal-Bench | HF `Shengcao1006/MMHal-Bench` | Apache-2.0 | open-ended answers | the official judge is GPT-4. This repo uses a local substitute judge, which is a deviation from the protocol |

Splits are grouped by `image_id` (70/10/10/10) with a fixed seed, so no image appears in
more than one split. Malformed records are dropped and logged to `data/processed/malformed.json`.
Large data should live outside the OneDrive-synced project folder.
