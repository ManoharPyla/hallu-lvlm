"""FeatureRecord dataclass, save/load and design-matrix construction."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ARRAY_FIELDS = ("logprob", "entropy", "hidden", "grounding", "labels")


@dataclass
class FeatureRecord:
    sample_id: str
    image_id: str
    tokens: list[str]
    offsets: list[tuple[int, int]]
    logprob: np.ndarray
    entropy: np.ndarray
    hidden: np.ndarray
    grounding: np.ndarray
    labels: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)


def save_records(path: str | Path, records: list[FeatureRecord]) -> None:
    """Save to a directory: index.json + one compressed npz per record."""
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    index = []
    for i, r in enumerate(records):
        fn = f"rec_{i:07d}.npz"
        np.savez_compressed(d / fn, **{k: np.asarray(getattr(r, k)) for k in ARRAY_FIELDS})
        index.append({"file": fn, "sample_id": r.sample_id, "image_id": r.image_id, "tokens": r.tokens,
                      "offsets": [list(map(int, o)) for o in r.offsets], "meta": r.meta})
    (d / "index.json").write_text(json.dumps(index), encoding="utf-8")


def load_records(path: str | Path) -> list[FeatureRecord]:
    d = Path(path)
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    out = []
    for e in index:
        with np.load(d / e["file"]) as z:
            arrs = {k: z[k] for k in ARRAY_FIELDS}
        out.append(FeatureRecord(e["sample_id"], e["image_id"], list(e["tokens"]),
                                 [tuple(o) for o in e["offsets"]], meta=e.get("meta", {}), **arrs))
    return out


def build_matrix(records: list[FeatureRecord], use: tuple[str, ...] = ("hidden", "logprob", "entropy", "grounding")):
    """Stack token features; drops label -100. Returns (X[N,D], y[N], groups[N], index)."""
    Xs, ys, gs, idx = [], [], [], []
    for ri, r in enumerate(records):
        T = len(r.labels)
        cols = []
        for name in use:
            a = np.asarray(getattr(r, name), dtype=np.float32)
            a = a.reshape(T, -1) if a.size else np.zeros((T, 0), np.float32)
            cols.append(a)
        feats = np.concatenate(cols, 1) if cols else np.zeros((T, 0), np.float32)
        keep = np.asarray(r.labels) != -100
        Xs.append(feats[keep])
        ys.append(np.asarray(r.labels)[keep])
        gs.append(np.full(int(keep.sum()), r.image_id, dtype=object))
        idx.extend((ri, int(t)) for t in np.nonzero(keep)[0])
    if not Xs:
        return np.zeros((0, 0), np.float32), np.zeros(0, np.int64), np.zeros(0, object), []
    X = np.nan_to_num(np.concatenate(Xs, 0))
    return X, np.concatenate(ys).astype(np.int64), np.concatenate(gs), idx
