# Phase 1 — Environment Inspection & Research Plan (2026-10-20)

## 1. Environment (measured)
| Item | Value | Consequence |
|---|---|---|
| OS | Windows 11 | Use `uv` venv, avoid bitsandbytes pain where possible |
| GPU | RTX 2050, **4 GB VRAM** | LLaVA-1.5-7B fp16 (~15 GB) impossible; 4-bit (~4–5 GB) marginal/unreliable |
| torch | 2.14.0+**cpu** on Python 3.14 | Need CUDA torch in a separate env (Python 3.11 via uv available) |
| RAM | 16 GB | 7B CPU fp16 not feasible; 7B CPU bf16 ~14 GB is borderline |
| Disk | 45.8 GB free (OneDrive-synced folder!) | Keep data/weights OUTSIDE OneDrive (e.g. `C:\hallu_data`) |
| Missing libs | transformers, accelerate, gradio, datasets, bitsandbytes | install in project venv |
| Project dir | empty | greenfield, nothing to reuse (no HSA-DPO code present) |

## 2. Backend decision
- **Primary (local dev + smoke):** `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` (0.89B, 1.8 GB fp16, Apache-2.0) on CUDA fp16 in 4 GB.
  Same HF `generate(output_scores, output_hidden_states)` path as LLaVA-1.5.
- **Target (paper results):** `llava-hf/llava-1.5-7b-hf` (llama2 license, 14.1 GB) — needs ≥16 GB GPU (cloud/lab machine) or 4-bit trial locally.
- Trade-off: HalLoc annotations label *source texts already in the dataset*, so the detector can be trained on teacher-forced features of ANY model; but features from a 0.5B model differ from 7B — results do not transfer and must be reported per-model.

## 3. Data (verified sources)
| Dataset | Source | License | Size | Notes |
|---|---|---|---|---|
| HalLoc | HF `uunicee/HalLoc`, CVPR'25 | **unstated** | 155,953 samples, token-level (Object/Attribute/Relationship/Scene, word_index) | images from Visual Genome + COCO2014 (not bundled) |
| COCO 2014 | cocodataset.org | CC-BY 4.0 (annots) | val2014 6.6 GB, annots 253 MB | for POPE, CHAIR, HalLoc subset |
| POPE | RUCAIBox/POPE | MIT | small | needs COCO val2014 images |
| HallusionBench | tianyi-lab | BSD-3 | images zip on Google Drive | non-COCO |
| MMHal-Bench | HF Shengcao1006 | Apache-2.0 | ~350 MB, 96 Q | official scoring uses GPT-4 → need local judge substitute (deviation, must be disclosed) |
| Visual Genome | VG site | CC-BY 4.0 | ~15 GB images | only if HalLoc VG subset used — defer; start with COCO-image HalLoc subset |

Storage estimate for MVP: model 1.8 GB + COCO val2014 6.6 GB + annots 0.3 GB + HalLoc json <1 GB + features (~2–5 GB) ≈ **12–15 GB**. Fits.

## 4. Novelty assessment (honest)
Already published: token detectors on prob/entropy w/ calibration (**MetaToken**, VISAPP'25, arXiv 2405.19186) + LURE correction; hidden-state token localizers (**HalLoc**, CVPR'25, 2506.10286); post-hoc correction (Woodpecker, LURE, LogicCheckGPT); detector-gated decoding (ECD 2504.12137, HALC).
→ "MLP + calibration + correction" alone ≈ MetaToken+LURE. **Not novel.**

Plausible narrower contribution (to be tested, not claimed):
1. Threshold with **distribution-free risk control** (conformal / Learn-Then-Test) on residual hallucination rate, instead of plain Platt.
2. **Minimal-change span correction with abstention**, measured by preservation / new-hallucination / over-edit rates vs Woodpecker-style and whole-rewrite baselines.
3. Beyond objects (attributes/relations via HalLoc) + calibration transfer across datasets.
Several citations above are from agent recall and must be re-verified before the paper.

## 5. Key risks
1. 4 GB VRAM → cannot run the 7B target locally; main results need external GPU.
2. HalLoc license unstated → research use only, contact authors before redistribution.
3. MMHal-Bench official judge is GPT-4 (external API) — conflicts with local-only goal; substitute or skip.
4. Corrector needs a generator: with frozen small LVLM, correction quality will be weak; grounding check on COCO = annotation-based (absence of annotation ≠ absence).
5. Source-attribution labels (under-grounding / language prior / conflict) do not exist → "unknown" unless a manual subset is annotated.
6. Project folder is OneDrive-synced → large files must live elsewhere.

## 6. Next steps (awaiting approval)
- **Phase 2 (cheap, no big downloads):** uv venv (Py 3.11, CUDA torch cu12x), repo skeleton, configs, dataset adapters (HalLoc/POPE/COCO), word→subword label alignment, tests on synthetic fixtures. Parallel agents per module.
- **Needs approval:** (a) ~2.5 GB pip install incl. CUDA torch, (b) 1.8 GB 0.5B model, (c) 6.9 GB COCO val2014 + annotations, (d) HalLoc JSON, (e) decision on where to run 7B.
