"""Mitigation policies.

``sample_ctx`` (dict) keys used:
  response (str, required), prompt (str), image_id, image,
  offsets: list[(start, end)] token char offsets,
  probs: np.ndarray[T] calibrated token risks  (OR)  X: np.ndarray[T, D] token features.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from hallu.grounding.checker import GroundingVerdict
from hallu.mitigation.minimal_edit import diff_spans, expand_to_words
from hallu.mitigation.rewriter import Rewriter

logger = logging.getLogger(__name__)


@dataclass
class Edit:
    start: int  # offsets in the ORIGINAL text
    end: int
    old: str
    new: str
    reason: str


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    edits: list[Edit] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _edits_from_diff(original: str, corrected: str, decisions: list[dict]) -> list[Edit]:
    edits = []
    for _tag, a1, a2, b1, b2 in diff_spans(original, corrected):
        reason = "rewrite"
        for d in decisions:
            if d.get("action") in ("delete", "replace", "hedge") and d["start"] <= a2 and a1 <= d["end"]:
                reason = f"{d['action']}:{d['verdict']}"
                break
        edits.append(Edit(a1, a2, original[a1:a2], corrected[b1:b2], reason))
    return edits


def group_flagged(offsets, flags, text: str, max_gap: int = 1) -> list[tuple[int, int]]:
    """Merge flagged tokens into whole-word char spans (uses hallu.data.alignment if present)."""
    try:
        from hallu.data.alignment import group_tokens_to_spans
        return [tuple(s) for s in group_tokens_to_spans(offsets, list(map(bool, flags)), text, max_gap)]
    except ImportError:
        pass
    spans: list[list[int]] = []
    last_i = None
    for i, ((s, e), f) in enumerate(zip(offsets, flags)):
        if not f or e <= s:
            continue
        if spans and last_i is not None and i - last_i <= max_gap:
            spans[-1][1] = e
        else:
            spans.append([s, e])
        last_i = i
    return [expand_to_words(text, s, e) for s, e in spans]


class Policy:
    name = "policy"

    def apply(self, sample_ctx: dict) -> CorrectionResult:  # pragma: no cover
        raise NotImplementedError


class NoMitigation(Policy):
    name = "none"

    def apply(self, sample_ctx: dict) -> CorrectionResult:
        r = sample_ctx["response"]
        return CorrectionResult(r, r, [], [], 0.0)


class WholeRewrite(Policy):
    name = "whole_rewrite"

    def __init__(self, lvlm=None, rewriter: Rewriter | None = None):
        self.rewriter = rewriter or Rewriter(lvlm)

    def apply(self, sample_ctx: dict) -> CorrectionResult:
        t0 = time.perf_counter()
        r = sample_ctx["response"]
        new = self.rewriter.whole_rewrite(sample_ctx.get("prompt", ""), r, sample_ctx.get("image"))
        dec = [dict(start=0, end=len(r), action="rewrite", verdict="n/a")]
        return CorrectionResult(r, new, _edits_from_diff(r, new, dec), dec, time.perf_counter() - t0)


class SelectiveCorrection(Policy):
    """Detector-gated, grounding-checked minimal correction.

    risk >= threshold -> grounding check:
      CONTRADICTED -> delete phrase (or LVLM replacement if rewriter has a model)
      UNKNOWN      -> keep (or hedge if hedge_unknown)
      SUPPORTED    -> keep
    Never invents content without a model; edits are whole words only.
    """
    name = "selective"

    def __init__(self, detector=None, calibrator=None, threshold: float = 0.5, checker=None,
                 rewriter: Rewriter | None = None, hedge_unknown: bool = False, max_gap: int = 1):
        self.detector, self.calibrator, self.threshold = detector, calibrator, float(threshold)
        self.checker = checker
        self.rewriter = rewriter or Rewriter(None)
        self.hedge_unknown = hedge_unknown
        self.max_gap = max_gap

    def _probs(self, ctx: dict) -> np.ndarray:
        if ctx.get("probs") is not None:
            return np.asarray(ctx["probs"], float)
        logits = self.detector.predict_logits(ctx["X"])
        if self.calibrator is not None:
            return np.asarray(self.calibrator.transform(logits), float)
        return 1.0 / (1.0 + np.exp(-np.asarray(logits, float)))

    def apply(self, sample_ctx: dict) -> CorrectionResult:
        t0 = time.perf_counter()
        text = sample_ctx["response"]
        offsets = sample_ctx["offsets"]
        probs = self._probs(sample_ctx)
        flags = probs >= self.threshold
        spans = group_flagged(offsets, flags, text, self.max_gap)
        decisions = []
        for s, e in spans:
            if e <= s or not re.search(r"\w", text[s:e]):
                continue
            risk = float(max((p for (a, b), p in zip(offsets, probs) if a < e and b > s), default=0.0))
            if self.checker is not None:
                verdict, evidence = self.checker.check(text[s:e], sample_ctx.get("image_id"),
                                                       sample_ctx.get("image"))
            else:
                verdict, evidence = GroundingVerdict.UNKNOWN, "no checker"
            if verdict == GroundingVerdict.CONTRADICTED:
                action = "delete_or_replace"
            elif verdict == GroundingVerdict.UNKNOWN and self.hedge_unknown:
                action = "hedge"
            else:
                action = "keep"
            decisions.append(dict(start=s, end=e, text=text[s:e], risk=risk,
                                  verdict=GroundingVerdict(verdict).value, evidence=evidence,
                                  action=action))
        # apply right-to-left on the evolving text so earlier offsets stay valid
        new = text
        for d in sorted(decisions, key=lambda d: d["start"], reverse=True):
            if d["action"] == "delete_or_replace":
                new, d["action"] = self.rewriter.rewrite_span(new, d["start"], d["end"],
                                                              sample_ctx.get("image"))
            elif d["action"] == "hedge":
                new = self.rewriter.hedge(new, d["start"], d["end"])
        edits = _edits_from_diff(text, new, decisions)
        return CorrectionResult(text, new, edits, decisions, time.perf_counter() - t0)


def get_policy(name: str, **kw) -> Policy:
    return {"none": NoMitigation, "whole_rewrite": WholeRewrite,
            "selective": SelectiveCorrection}[name](**kw)
