"""Round-trip serialization of result / data objects."""
import json

import numpy as np

from hallu.data.schema import HallSpan, Sample, YesNoItem
from hallu.features.record import FeatureRecord, load_records, save_records
from hallu.mitigation.policies import CorrectionResult, Edit
from hallu.utils.io import load_json, read_jsonl, save_json, write_jsonl


def _sample():
    return Sample("s1", "42", None, "Describe.", "A dog and a cat.",
                  [HallSpan(12, 15, "cat", "object")], "synthetic", "test", {"k": 1})


def test_sample_roundtrip(tmp_path):
    s = _sample()
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [s.to_dict()])
    back = Sample.from_dict(read_jsonl(p)[0])
    assert back == s


def test_yesno_roundtrip():
    it = YesNoItem("q1", "7", None, "Is there a dog?", "yes", "pope", "random", {"a": 2})
    assert YesNoItem.from_dict(json.loads(json.dumps(it.to_dict()))) == it


def test_correction_result_roundtrip(tmp_path):
    r = CorrectionResult("A dog and a cat.", "A dog.", [Edit(5, 15, " and a cat", "", "delete:contradicted")],
                         [dict(start=12, end=15, text="cat", risk=0.9, verdict="contradicted",
                               evidence="x", action="delete")], 0.01)
    p = tmp_path / "r.jsonl"
    write_jsonl(p, [r.to_dict()])
    d = read_jsonl(p)[0]
    back = CorrectionResult(d["original"], d["corrected"], [Edit(**e) for e in d["edits"]],
                            d["decisions"], d["latency_s"])
    assert back == r


def test_metrics_json_roundtrip(tmp_path):
    m = {"rows": {"mlp": {"f1": 0.5, "auroc": None}}, "banner": "x — y"}
    save_json(tmp_path / "m.json", m)
    assert load_json(tmp_path / "m.json") == m


def test_feature_record_roundtrip(tmp_path):
    T = 4
    r = FeatureRecord("s1", "42", ["a", "b", "c", "d"], [(0, 1), (2, 3), (4, 5), (6, 7)],
                      np.arange(T, dtype=np.float32), np.ones(T, np.float32),
                      np.zeros((T, 3), np.float16), np.zeros((T, 0), np.float32),
                      np.array([0, 1, -100, 0]), meta={"m": 1})
    save_records(tmp_path / "f", [r])
    b = load_records(tmp_path / "f")[0]
    assert b.sample_id == r.sample_id and b.offsets == r.offsets and b.meta == r.meta
    for k in ("logprob", "entropy", "hidden", "grounding", "labels"):
        np.testing.assert_array_equal(getattr(b, k), getattr(r, k))
