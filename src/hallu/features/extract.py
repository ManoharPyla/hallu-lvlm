"""Run LVLM.teacher_force over labeled Samples -> FeatureRecords with token labels."""
from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

from hallu.data.alignment import char_labels_to_token_labels
from hallu.features.grounding_features import grounding_features
from hallu.features.record import FeatureRecord

logger = logging.getLogger(__name__)


def extract_record(lvlm: Any, sample: Any, min_overlap: float = 0.0) -> FeatureRecord:
    out = lvlm.teacher_force(sample.image_path, sample.prompt, sample.response)
    T = len(out.token_ids)
    special = [e <= s for s, e in out.offsets]
    labels = char_labels_to_token_labels(list(out.offsets), sample.spans, special_mask=special, min_overlap=min_overlap)
    stats = out.logits
    meta = {"source": sample.source, "split": sample.split, "latency_s": float(out.latency_s)}
    if out.text != sample.response:
        meta["text_mismatch"] = True
    return FeatureRecord(
        sample_id=sample.sample_id, image_id=sample.image_id, tokens=list(out.tokens),
        offsets=[tuple(o) for o in out.offsets],
        logprob=np.asarray(stats["logprob"], np.float32), entropy=np.asarray(stats["entropy"], np.float32),
        hidden=np.asarray(out.hidden, np.float16), grounding=grounding_features(out, T),
        labels=np.asarray(labels, np.int64), meta=meta,
    )


def extract_features(lvlm: Any, samples: Iterable[Any], min_overlap: float = 0.0,
                     max_failures: int | None = None) -> tuple[list[FeatureRecord], dict]:
    """Returns (records, counts). Failures are logged and counted, not raised
    (unless more than ``max_failures`` occur)."""
    records: list[FeatureRecord] = []
    counts = {"total": 0, "ok": 0, "failed": 0, "text_mismatch": 0}
    for s in samples:
        counts["total"] += 1
        try:
            r = extract_record(lvlm, s, min_overlap)
        except Exception as e:  # keep going; record failure
            counts["failed"] += 1
            logger.exception(
                "feature extraction failed for %s (%s: %r)",
                getattr(s, "sample_id", "?"),
                type(e).__name__,
                e,
        )
            if max_failures is not None and counts["failed"] > max_failures:
                raise
            continue
        counts["ok"] += 1
        counts["text_mismatch"] += int(r.meta.get("text_mismatch", False))
        records.append(r)
    logger.info("feature extraction: %s", counts)
    return records, counts
