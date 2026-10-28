"""Scalar per-token statistics derived from logits."""
from __future__ import annotations

import numpy as np


def token_stats(logits: np.ndarray, token_ids: np.ndarray) -> dict[str, np.ndarray]:
    """logits [T,V], token_ids [T] -> dict(logprob, entropy, margin, rank), each float32 [T].

    margin = logprob(top1) - logprob(chosen) (0 when chosen is top-1, else gap to top-1);
    rank = 0-based rank of the chosen token.
    """
    logits = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(token_ids, dtype=np.int64)
    T = len(ids)
    if T == 0:
        z = np.zeros(0, np.float32)
        return {"logprob": z, "entropy": z.copy(), "margin": z.copy(), "rank": z.copy()}
    logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    m = logits.max(-1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(-1, keepdims=True))
    logp = logits - lse
    p = np.exp(logp)
    chosen = logp[np.arange(T), ids]
    entropy = -(p * logp).sum(-1)
    margin = logp.max(-1) - chosen
    rank = (logp > chosen[:, None]).sum(-1)
    return {"logprob": chosen.astype(np.float32), "entropy": entropy.astype(np.float32),
            "margin": margin.astype(np.float32), "rank": rank.astype(np.float32)}
