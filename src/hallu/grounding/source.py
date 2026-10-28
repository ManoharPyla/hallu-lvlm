"""Heuristic hallucination *source* labels (see docs/source_taxonomy.md).

These are HEURISTICS over observable signals, not causal attributions. Default is
UNKNOWN; a label is only emitted when a documented rule fires on the provided
features. Thresholds are placeholders to be set on a validation split.
"""
from __future__ import annotations

import enum
import logging
from typing import Mapping

logger = logging.getLogger(__name__)


class SourceLabel(str, enum.Enum):
    VISUAL_UNDERGROUNDING = "visual_undergrounding"
    LANGUAGE_PRIOR = "language_prior"
    CROSS_MODAL_CONFLICT = "cross_modal_conflict"
    UNKNOWN = "unknown"


DEFAULT_RULES = dict(
    low_image_attn=0.10,     # image_attn_mass below this -> under-grounded
    prior_gap=1.0,           # logprob(with image) - logprob(text only) below this -> prior-driven
    high_conf_logprob=-0.5,  # confident token (logprob above this)
)


def estimate_source(features: Mapping, rules: Mapping | None = None) -> SourceLabel:
    """Optional keys in ``features``: image_attn_mass, logprob, logprob_text_only,
    grounding_verdict ('contradicted'|...). Missing signals -> rule does not fire.

    R3 (cross-modal conflict): grounding CONTRADICTED and token was confident with
        image attention not low (model looked, yet asserted something contradicting).
    R2 (language prior): text-only logprob nearly as high as with image (gap < prior_gap).
    R1 (visual under-grounding): image attention mass < low_image_attn.
    Precedence R3 > R2 > R1. Exactly one label; otherwise UNKNOWN.
    """
    r = {**DEFAULT_RULES, **(rules or {})}
    attn = features.get("image_attn_mass")
    lp = features.get("logprob")
    lp_txt = features.get("logprob_text_only")
    verdict = str(getattr(features.get("grounding_verdict"), "value", features.get("grounding_verdict")))

    if (verdict == "contradicted" and lp is not None and lp > r["high_conf_logprob"]
            and attn is not None and attn >= r["low_image_attn"]):
        return SourceLabel.CROSS_MODAL_CONFLICT
    if lp is not None and lp_txt is not None and (lp - lp_txt) < r["prior_gap"]:
        return SourceLabel.LANGUAGE_PRIOR
    if attn is not None and attn < r["low_image_attn"]:
        return SourceLabel.VISUAL_UNDERGROUNDING
    return SourceLabel.UNKNOWN
