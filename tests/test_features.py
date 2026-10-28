import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from hallu.features.grounding_features import grounding_features
from hallu.features.record import FeatureRecord, build_matrix, load_records, save_records
from hallu.features.token_stats import token_stats
from hallu.models.errors import ModelUnavailableError
from hallu.models.lvlm import GenOutput, incremental_offsets


def test_token_stats():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    s = token_stats(logits, np.array([0, 2]))
    assert set(s) == {"logprob", "entropy", "margin", "rank"}
    assert s["rank"][0] == 0 and s["margin"][0] == 0
    np.testing.assert_allclose(s["entropy"][1], np.log(3), rtol=1e-5)
    np.testing.assert_allclose(np.exp(s["logprob"][1]), 1 / 3, rtol=1e-5)
    assert token_stats(np.zeros((0, 3)), np.zeros(0, int))["logprob"].shape == (0,)


def _rec(i, T=5, H=4, G=0):
    rng = np.random.default_rng(i)
    return FeatureRecord(f"s{i}", f"img{i % 2}", [f"t{j}" for j in range(T)], [(j, j + 1) for j in range(T)],
                         rng.normal(size=T).astype(np.float32), rng.random(T).astype(np.float32),
                         rng.normal(size=(T, H)).astype(np.float16), np.zeros((T, G), np.float32),
                         np.array([0, 1, -100, 0, 1][:T]))


def test_save_load_and_matrix(tmp_path):
    recs = [_rec(0), _rec(1)]
    save_records(tmp_path / "f", recs)
    back = load_records(tmp_path / "f")
    assert back[1].sample_id == "s1" and back[1].offsets[2] == (2, 3)
    np.testing.assert_array_equal(back[0].hidden, recs[0].hidden)
    X, y, g, idx = build_matrix(back)
    assert X.shape == (8, 4 + 1 + 1) and y.shape == (8,) and len(idx) == 8
    assert -100 not in y and set(g) == {"img0", "img1"}
    assert idx[2] == (0, 3)


def test_grounding_features():
    out = SimpleNamespace(token_ids=[1, 2, 3], image_attn_mass=None)
    assert grounding_features(out).shape == (3, 0)
    out.image_attn_mass = np.array([0.1, np.nan])
    g = grounding_features(out)
    assert g.shape == (3, 1) and np.isfinite(g).all()


class _Tok:
    vocab = {1: "A", 2: " cat", 3: " sits", 0: ""}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)


def test_incremental_offsets():
    text, toks, offs = incremental_offsets(_Tok(), [1, 2, 0, 3])
    assert text == "A cat sits"
    assert offs == [(0, 1), (1, 5), (5, 5), (5, 10)]
    assert toks[1] == " cat"


# Add the new code below this function
class _ChangingDecodeTok:
    """Simulate a tokenizer whose decoded prefix can change."""
    vocab = {1: "a", 2: "X"}

    def decode(self, ids, skip_special_tokens=True):
        if ids == [1]:
            return "a"
        if ids == [1, 2]:
            return "ba"
        return ""


def test_incremental_offsets_detects_prefix_rewrite():
    with pytest.raises(ValueError, match="Unstable decoded prefix"):
        incremental_offsets(_ChangingDecodeTok(), [1, 2])


def test_lvlm_unavailable(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "transformers":
            raise ImportError("no transformers")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    from hallu.models.lvlm import LVLM
    with pytest.raises(ModelUnavailableError, match="Hint"):
        LVLM()


def _load_data_modules():
    try:
        from hallu.data.alignment import char_labels_to_token_labels  # noqa: F401
        from hallu.data.schema import HallSpan, Sample
    except ImportError:
        pytest.skip("hallu.data not available yet")
    return HallSpan, Sample


class _FakeLVLM:
    def teacher_force(self, image, prompt, response):
        if response == "boom":
            raise RuntimeError("fail")
        text, toks, offs = incremental_offsets(_Tok(), [1, 2, 3])
        T = 3
        return GenOutput(text, [1, 2, 3], toks, offs,
                         {"logprob": -np.ones(T), "entropy": np.ones(T), "margin": np.zeros(T), "rank": np.zeros(T)},
                         np.zeros((T, 4), np.float16), np.array([0.2, 0.3, 0.1]), 0.01)


def test_extract_with_mock_lvlm():
    HallSpan, Sample = _load_data_modules()
    from hallu.features.extract import extract_features
    good = Sample("a", "img", None, "Describe.", "A cat sits", [HallSpan(2, 5, "cat", "object")], "test")
    bad = Sample("b", "img", None, "Describe.", "boom", [], "test")
    recs, counts = extract_features(_FakeLVLM(), [good, bad])
    assert counts == {"total": 2, "ok": 1, "failed": 1, "text_mismatch": 0}
    r = recs[0]
    assert list(r.labels) == [0, 1, 0]
    assert r.grounding.shape == (3, 1) and r.hidden.shape == (3, 4)
