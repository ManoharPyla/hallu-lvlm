"""POPE / yes-no benchmark evaluation."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Callable, Iterable

from hallu.evaluation.metrics import pope_metrics

logger = logging.getLogger(__name__)


def parse_yes_no(answer: str) -> str:
    """Map a free-form answer to 'yes'/'no' (leading yes/no wins; else negation -> 'no')."""
    words = re.findall(r"[a-z]+", answer.strip().lower())
    if not words:
        return "no"
    if words[0] in {"yes", "no"}:
        return words[0]
    if "no" in words or "not" in words:
        return "no"
    return "yes" if "yes" in words else "no"


def evaluate_pope(items: Iterable, answer_fn: Callable[[object], str]) -> dict:
    """items: YesNoItem-like (label, subset); answer_fn(item) -> raw model answer."""
    by: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    allp, alll = [], []
    for it in items:
        pred = parse_yes_no(answer_fn(it))
        key = getattr(it, "subset", "") or "all"
        by[key][0].append(pred)
        by[key][1].append(it.label)
        allp.append(pred)
        alll.append(it.label)
    out = {"overall": pope_metrics(allp, alll)}
    for k, (p, l) in by.items():
        out[k] = pope_metrics(p, l)
    return out
