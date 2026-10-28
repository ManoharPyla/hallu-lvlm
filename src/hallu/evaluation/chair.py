"""CHAIR evaluation over generated captions (Rohrbach et al., 2018).

Object extraction uses the *approximate* COCO synonym table in
``hallu.grounding.coco_objects`` (not the official list); report numbers as such.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

from hallu.evaluation.metrics import chair as chair_metric
from hallu.grounding.coco_objects import extract_objects

logger = logging.getLogger(__name__)


def caption_classes(text: str) -> set[str]:
    return {c for _, c, _ in extract_objects(text)}


def evaluate_chair(captions: dict[str, str], gt_fn: Callable[[str], Iterable[str]]) -> dict:
    """captions: image_id -> caption; gt_fn(image_id) -> ground-truth COCO classes."""
    ids = list(captions)
    cap_objs = [caption_classes(captions[i]) for i in ids]
    gts = [set(gt_fn(i)) for i in ids]
    res = chair_metric(cap_objs, gts)
    res["per_image"] = [dict(image_id=i, mentioned=sorted(c), hallucinated=sorted(c - g))
                        for i, c, g in zip(ids, cap_objs, gts)]
    res["n_captions"] = len(ids)
    return res
