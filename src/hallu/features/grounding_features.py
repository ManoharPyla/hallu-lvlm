"""Grounding-related token features.

Currently: image-attention mass per token (head-mean last-layer attention onto image
tokens), when the LVLM was run with output_attentions. This is a descriptive signal
only; high attention mass is NOT proof that a token is grounded in the image.
If unavailable, returns a zero-width [T, 0] array.
"""
from __future__ import annotations

import numpy as np


def grounding_features(gen_output, T: int | None = None) -> np.ndarray:
    T = len(gen_output.token_ids) if T is None else T
    mass = getattr(gen_output, "image_attn_mass", None)
    if mass is None:
        return np.zeros((T, 0), np.float32)
    m = np.nan_to_num(np.asarray(mass, np.float32).reshape(-1)[:T])
    if len(m) < T:
        m = np.pad(m, (0, T - len(m)))
    return m[:, None]
