"""Grounding checker: is a flagged span supported by the image?

Key principle: absence of an annotation is NOT absence of the object. The
annotation backend only returns CONTRADICTED for a mention of one of the 80 COCO
classes that is absent from BOTH the instance annotations AND every reference
caption of a known image. Everything else it cannot decide is UNKNOWN.
"""
from __future__ import annotations

import enum
import logging
from typing import Protocol

from hallu.grounding.coco_objects import CocoGT, extract_objects

logger = logging.getLogger(__name__)


class GroundingVerdict(str, enum.Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class Backend(Protocol):
    def check(self, span_text: str, image_id=None, image=None) -> tuple[GroundingVerdict, str]: ...


def _combine(verdicts: list[GroundingVerdict]) -> GroundingVerdict:
    if not verdicts:
        return GroundingVerdict.UNKNOWN
    if any(v == GroundingVerdict.CONTRADICTED for v in verdicts):
        return GroundingVerdict.CONTRADICTED
    if all(v == GroundingVerdict.SUPPORTED for v in verdicts):
        return GroundingVerdict.SUPPORTED
    return GroundingVerdict.UNKNOWN


class AnnotationBackend:
    """COCO-annotation-based checker (objects only)."""

    def __init__(self, gt: CocoGT):
        self.gt = gt

    def check(self, span_text: str, image_id=None, image=None) -> tuple[GroundingVerdict, str]:
        objs = extract_objects(span_text)
        if image_id is None or not objs:
            return GroundingVerdict.UNKNOWN, "no image id or no COCO object in span"
        if not self.gt.has_image(image_id):
            return GroundingVerdict.UNKNOWN, f"image {image_id} not in COCO annotations"
        inst, caps = self.gt.gt_objects(image_id), self.gt.caption_objects(image_id)
        verdicts, ev = [], []
        for word, cls, _ in objs:
            if cls in inst:
                verdicts.append(GroundingVerdict.SUPPORTED); ev.append(f"'{word}'->{cls}: in instances")
            elif cls in caps:
                verdicts.append(GroundingVerdict.UNKNOWN); ev.append(f"'{word}'->{cls}: caption-only")
            else:
                verdicts.append(GroundingVerdict.CONTRADICTED)
                ev.append(f"'{word}'->{cls}: absent from instances and captions")
        return _combine(verdicts), "; ".join(ev)


class VQABackend:
    """Asks the LVLM 'Is there a {obj} in the image?'. Requires an image."""

    def __init__(self, lvlm):
        self.lvlm = lvlm

    def check(self, span_text: str, image_id=None, image=None) -> tuple[GroundingVerdict, str]:
        objs = extract_objects(span_text)
        if image is None or not objs:
            return GroundingVerdict.UNKNOWN, "no image or no COCO object in span"
        from hallu.evaluation.pope_eval import parse_yes_no
        verdicts, ev = [], []
        for word, cls, _ in objs:
            q = f"Is there a {cls} in the image? Answer yes or no."
            try:
                a = self.lvlm.answer(image, q)
            except Exception as e:  # model unavailable etc. -> no evidence
                logger.warning("VQA backend failed: %s", e)
                return GroundingVerdict.UNKNOWN, f"VQA error: {e}"
            yn = parse_yes_no(a)
            verdicts.append(GroundingVerdict.SUPPORTED if yn == "yes" else GroundingVerdict.CONTRADICTED)
            ev.append(f"VQA '{cls}': {a!r}")
        return _combine(verdicts), "; ".join(ev)


class GroundingChecker:
    """Runs backends in order. Agreeing decisive verdicts win; disagreement -> UNKNOWN."""

    def __init__(self, backends: list):
        self.backends = list(backends)

    def check(self, span_text: str, image_id=None, image=None) -> tuple[GroundingVerdict, str]:
        decisive, ev = [], []
        for b in self.backends:
            v, e = b.check(span_text, image_id=image_id, image=image)
            ev.append(f"{type(b).__name__}: {v.value} ({e})")
            if v != GroundingVerdict.UNKNOWN:
                decisive.append(v)
        if not decisive or len(set(decisive)) > 1:
            return GroundingVerdict.UNKNOWN, " | ".join(ev)
        return decisive[0], " | ".join(ev)
