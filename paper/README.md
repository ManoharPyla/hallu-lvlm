# Paper draft

- `main.tex`: draft paper. It uses the generic `article` class. **Swap in the target venue's official template before submission.**
- `tables/`: `main_results.tex`, `ablation.tex`, `related_work.tex`. These are pulled into `main.tex` with `\input`.
- `references.bib`: the bibliography (natbib, `plainnat`).

## No fabricated numbers
Every result cell is `\TBD{}`, which renders as a red **[TBD]**. Fill a cell only from logged experiment outputs. `related_work.tex` is a qualitative design comparison and holds no numbers. Re-check it against each paper before submission.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Run it from `paper/`.

## Reference verification log (2026-10-21)
Titles, authors and years were checked with the arXiv API for every arXiv entry. Venues were checked in one of these ways:
- From the arXiv comment or journal_ref: HalLoc (CVPR'25), LURE (ICLR'24), HALC (ICML'24), OPERA (CVPR'24), LogicCheckGPT (ACL'24), M-HalDetect (AAAI'24), POPE (EMNLP'23), CHAIR (EMNLP'18), HallusionBench (CVPR'24), LLaVA-1.5 (CVPR'24), Guo et al. (ICML'17), Whitehead et al. (ECCV'22), Woodpecker (Sci. China Inf. Sci. 2024).
- By web search of the official proceedings pages: VCD (CVPR'24, pp. 13872-13882, CVF open access), HSA-DPO (AAAI'25, 39(24):25543-25551), VISTA (ICML'25, PMLR 267), LLaVA-RLHF/MMHal-Bench (Findings of ACL 2024), Learn-Then-Test (Ann. Appl. Stat. 19(2), 2025), Platt (Advances in Large Margin Classifiers, MIT Press, pp. 61-74; some sources date the volume to 2000).
- Cited as arXiv only: ECD (2504.12137, no venue found), LLaVA-OneVision (2408.03326), Conformal Risk Control (2208.02814, added as a related risk-control reference).
- Partly verified: **MetaToken** VISAPP 2025. arXiv metadata was confirmed, but no VISAPP/SciTePress page was found. It is cited as arXiv with a note. Confirm the venue before submission.
- COCO is cited as ECCV 2014. The arXiv entry was confirmed, but its comment does not state the venue. ECCV 2014 is the standard citation.

## Omissions
None of the requested candidates were dropped. Only MetaToken's venue is unconfirmed (see above). Learn-Then-Test page numbers beyond the start page (1641) were left out because they could not be confirmed.

## Open items
- MMHal-Bench local judge model: `\TBD{}` in the setup section. This is a declared deviation from the GPT-4 judge.
- Seeds and the hardware for the 7B runs are still to be decided.
