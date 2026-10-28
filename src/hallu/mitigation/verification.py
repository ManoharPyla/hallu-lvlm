"""Correction quality evaluation.

An UNCHANGED response that contains gold hallucinations is never a success:
halluc_removed = 0 and completeness = 0 (no reward for doing nothing).
Hedging keeps the hallucinated words and therefore does not count as removal.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

from hallu.grounding.coco_objects import extract_objects
from hallu.mitigation.minimal_edit import char_edit_distance, diff_spans

logger = logging.getLogger(__name__)


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _span_removed(span, changes, corrected: str) -> bool:
    s, e, stext = _get(span, "start"), _get(span, "end"), _get(span, "text")
    words = re.findall(r"\w+", stext.lower())
    for _tag, a1, a2, b1, b2 in changes:
        if a1 < e and a2 > s:  # change overlaps the gold span's characters
            new_words = set(re.findall(r"\w+", corrected[b1:b2].lower()))
            if not words or not all(w in new_words for w in words):
                return True
    return False


def correction_eval(original_spans_gold: list, result, gold_objects: Iterable[str] | None) -> dict:
    """original_spans_gold: HallSpan-like (start, end, text) in the original response.
    result: CorrectionResult-like (original, corrected, edits).
    gold_objects: COCO classes present in the image (None -> object metrics are nan).
    """
    original, corrected = _get(result, "original"), _get(result, "corrected")
    edits = list(_get(result, "edits") or [])
    changes = diff_spans(original, corrected) if original != corrected else []
    n_gold = len(original_spans_gold)
    nan = float("nan")

    if n_gold == 0:
        halluc_removed = nan
    elif not changes:
        halluc_removed = 0.0
    else:
        halluc_removed = sum(_span_removed(sp, changes, corrected) for sp in original_spans_gold) / n_gold

    def overlaps_gold(a1, a2):
        return any(a1 <= _get(sp, "end") and a2 >= _get(sp, "start") for sp in original_spans_gold)

    unnecessary = sum(1 for (_t, a1, a2, _b1, _b2) in changes if not overlaps_gold(a1, a2))

    if gold_objects is not None:
        gold = set(gold_objects)
        orig_objs = {c for _, c, _ in extract_objects(original)}
        corr_objs = {c for _, c, _ in extract_objects(corrected)}
        supported = orig_objs & gold
        supported_preserved = len(supported & corr_objs) / len(supported) if supported else nan
        new_halluc = len(corr_objs - orig_objs - gold)
    else:
        supported_preserved, new_halluc = nan, 0

    if n_gold == 0:
        completeness = 1.0 if (not changes and new_halluc == 0) else 0.0
    else:
        completeness = 1.0 if (halluc_removed == 1.0 and new_halluc == 0) else 0.0

    return dict(halluc_removed=halluc_removed, supported_preserved=supported_preserved,
                new_halluc=new_halluc, unnecessary_changes=unnecessary,
                completeness=completeness, edit_distance=char_edit_distance(original, corrected),
                n_edits=len(edits), changed=original != corrected)
