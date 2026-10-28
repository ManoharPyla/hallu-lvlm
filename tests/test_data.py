import json

import pytest

from hallu.data.adapters import get_adapter, resolve_image_path
from hallu.data.schema import HallSpan, Sample, YesNoItem
from hallu.data.splits import check_no_leakage, group_split, subsample
from hallu.data.stats import dataset_stats, stats_to_markdown
from hallu.data.validate import validate_samples
from hallu.utils.config import load_config
from hallu.utils.io import load_json, read_jsonl, save_json, write_jsonl
from hallu.utils.seed import set_seed


def mk(i, img, spans=()):
    return Sample(f"s{i}", img, None, "p", "A red dog sits.", list(spans), "t")


# ---------------- schema / io / config ----------------

def test_schema_roundtrip(tmp_path):
    s = mk(1, "img1", [HallSpan(2, 5, "red", "attribute")])
    y = YesNoItem("q", "i", None, "Is there?", "yes", "pope", "random")
    write_jsonl(tmp_path / "a.jsonl", [s.to_dict()])
    assert Sample.from_dict(read_jsonl(tmp_path / "a.jsonl")[0]) == s
    save_json(tmp_path / "b.json", y.to_dict())
    assert YesNoItem.from_dict(load_json(tmp_path / "b.json")) == y


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b: 1\nx: foo\n", encoding="utf-8")
    cfg = load_config(p, ["a.b=3", "a.c.d=true", "x=bar"])
    assert cfg == {"a": {"b": 3, "c": {"d": True}}, "x": "bar"}


def test_dataset_yaml_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "dataset.yaml")
    assert sum(cfg["split"]["ratios"].values()) == pytest.approx(1.0)


def test_set_seed_runs():
    import numpy as np

    set_seed(3)
    a = np.random.rand()
    set_seed(3)
    assert np.random.rand() == a


# ---------------- splits ----------------

def test_group_split_no_leakage_and_deterministic():
    samples = [mk(i, f"img{i % 37}") for i in range(300)]
    sp1 = group_split(samples, {"train": 0.7, "cal": 0.1, "val": 0.1, "test": 0.1}, seed=1)
    sp2 = group_split(list(reversed(samples)), {"train": 0.7, "cal": 0.1, "val": 0.1, "test": 0.1}, seed=1)
    assert check_no_leakage(sp1)
    assert sum(len(v) for v in sp1.values()) == 300
    assert {k: sorted(s.sample_id for s in v) for k, v in sp1.items()} == {
        k: sorted(s.sample_id for s in v) for k, v in sp2.items()
    }
    imgs = {k: {s.image_id for s in v} for k, v in sp1.items()}
    assert len(imgs["train"]) == round(0.7 * 37)
    assert all(s.split == "test" for s in sp1["test"])
    assert samples[0].split is None  # originals untouched


def test_check_no_leakage_detects():
    assert not check_no_leakage({"train": [mk(1, "a")], "test": [mk(2, "a")]})


def test_subsample_deterministic():
    samples = [mk(i, "x") for i in range(50)]
    a = subsample(samples, 10, seed=4)
    b = subsample(list(reversed(samples)), 10, seed=4)
    assert [s.sample_id for s in a] == [s.sample_id for s in b] and len(a) == 10
    assert len(subsample(samples, None)) == 50


# ---------------- validate / stats ----------------

def test_validate_records_reasons():
    good = mk(1, "a", [HallSpan(2, 5, "red", "attribute")])
    bad_range = mk(2, "a", [HallSpan(2, 50, "red", "attribute")])
    bad_text = mk(3, "a", [HallSpan(2, 5, "blu", "attribute")])
    valid, bad = validate_samples([good, bad_range, bad_text])
    assert [s.sample_id for s in valid] == ["s1"]
    assert {b["sample_id"] for b in bad} == {"s2", "s3"}
    assert any("out of range" in r for r in bad[0]["reasons"])
    assert any("mismatch" in r for r in bad[1]["reasons"])


def test_stats_and_markdown():
    st = dataset_stats([mk(1, "a", [HallSpan(2, 5, "red", "attribute")]), mk(2, "b")])
    assert st["n_samples"] == 2 and st["n_hallucinated_samples"] == 1
    assert st["htype_counts"] == {"attribute": 1}
    md = stats_to_markdown(st)
    assert "| n_spans | 1 |" in md


# ---------------- HalLoc adapter ----------------

HALLOC_RECS = [
    {  # dict-of-lists with char_index
        "id": 1, "split": "train", "image_id": "397133", "prompt": "Describe.",
        "source_text": "A brown dog sits.", "hallucinated_text": "A red dog sits on a sofa.",
        "annotations": {
            "Attribute": [{"name": "red", "word_index": 1, "char_index": 2}],
            "Object": [{"name": "sofa", "word_index": 6, "char_index": 20}],
        },
    },
    {  # wrong char_index -> fallback to word_index
        "id": 2, "split": "train", "image_id": "42", "prompt": "Q",
        "source_text": "x", "hallucinated_text": "Two cats play in snow.",
        "annotations": {"Object": {"name": "cats", "word_index": 1, "char_index": 99}},
    },
    {  # list entries with type, word_index only; scene
        "id": 3, "split": "val", "image_id": "7", "prompt": "Q",
        "source_text": "x", "hallucinated_text": "It is a beach scene.",
        "annotations": [{"type": "Scene", "token": "beach", "word_index": 3}],
    },
    {  # flat [name, word_index, char_index] sequence
        "id": 4, "split": "test", "image_id": "8", "prompt": "Q",
        "source_text": "x", "hallucinated_text": "The man rides a horse.",
        "annotations": {"Relationship": ["rides", 2, 8]},
    },
    {  # unresolvable -> malformed
        "id": 5, "split": "test", "image_id": "9", "prompt": "Q",
        "source_text": "x", "hallucinated_text": "A cat.",
        "annotations": {"Object": [{"name": "zebra", "word_index": 7, "char_index": 40}]},
    },
    {  # clean (no annotations)
        "id": 6, "split": "train", "image_id": "10", "prompt": "Q",
        "source_text": "A dog.", "hallucinated_text": "A dog.", "annotations": {},
    },
    {"id": 7, "image_id": "11", "annotations": {}},  # missing text -> malformed
]


def test_halloc_adapter(tmp_path):
    (tmp_path / "halloc_vqa.json").write_text(json.dumps(HALLOC_RECS), encoding="utf-8")
    ad = get_adapter("halloc", image_roots=[str(tmp_path / "imgs")])
    samples = ad.load(tmp_path, tasks=["vqa"])
    by = {s.meta["orig_id"]: s for s in samples}
    assert set(by) == {"1", "2", "3", "4", "6"}
    assert [(sp.text, sp.htype) for sp in by["1"].spans] == [("red", "attribute"), ("sofa", "object")]
    assert by["2"].spans[0].text == "cats"
    assert by["3"].spans[0].text == "beach" and by["3"].spans[0].htype == "scene"
    assert by["4"].spans[0].text == "rides" and by["4"].spans[0].htype == "relationship"
    assert by["6"].spans == []
    assert {m["sample_id"] for m in ad.malformed} == {"halloc_vqa_5", "halloc_vqa_7"}
    assert all(s.image_path is None for s in samples)
    valid, bad = validate_samples(samples)
    assert len(valid) == len(samples) and not bad


def test_halloc_keep_partial():
    ad = get_adapter("halloc", keep_partial=True)
    out = ad.parse_records([HALLOC_RECS[4]], "vqa")
    assert len(out) == 1 and out[0].meta["partial"] and len(ad.malformed) == 1


def test_resolve_image_path(tmp_path):
    d = tmp_path / "val2014"
    d.mkdir()
    f = d / "COCO_val2014_000000000042.jpg"
    f.write_bytes(b"")
    assert resolve_image_path("42", [tmp_path]) == str(f)
    assert resolve_image_path("43", [tmp_path]) is None
    assert resolve_image_path(None, [tmp_path]) is None


# ---------------- other adapters ----------------

def test_pope_adapter(tmp_path):
    p = tmp_path / "coco_pope_adversarial.json"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"question_id": 1, "image": "COCO_val2014_000000310196.jpg", "text": "Is there a dog?", "label": "yes"},
                {"question_id": 2, "image": "COCO_val2014_000000310196.jpg", "text": "Is there a cat?", "label": "no"},
                {"question_id": 3, "image": "x.jpg", "text": "?", "label": "maybe"},
            ]
        ),
        encoding="utf-8",
    )
    ad = get_adapter("pope")
    items = ad.load(p)
    assert len(items) == 2 and items[0].subset == "adversarial"
    assert items[0].image_id == "310196" and items[1].label == "no"
    assert len(ad.malformed) == 1


def test_hallusionbench_adapter(tmp_path):
    recs = [
        {"category": "VD", "subcategory": "illusion", "set_id": "0", "figure_id": "1", "question_id": "0",
         "question": "Is the line longer?", "gt_answer": "1", "filename": "./VD/illusion/0_1.png"},
        {"category": "VS", "subcategory": "chart", "set_id": "2", "figure_id": "0", "question_id": "1",
         "question": "Was X larger?", "gt_answer": "0", "filename": None},
        {"category": "VS", "question": "bad", "gt_answer": "2"},
    ]
    (tmp_path / "HallusionBench.json").write_text(json.dumps(recs), encoding="utf-8")
    ad = get_adapter("hallusionbench")
    items = ad.load(tmp_path)
    assert [i.label for i in items] == ["yes", "no"]
    assert items[0].image_id == "VD/illusion/0_1.png"
    assert items[1].image_path is None and not items[1].meta["visual_input"]
    assert len(ad.malformed) == 1


def test_mmhal_adapter(tmp_path):
    recs = [{"image_src": "https://x/y/abc123.jpg", "image_id": "abc123", "question": "What color?",
             "gt_answer": "Blue", "image_content": ["car"], "question_type": "attribute"}]
    (tmp_path / "response_template.json").write_text(json.dumps(recs), encoding="utf-8")
    items = get_adapter("mmhal-bench").load(tmp_path)
    assert items[0].image_id == "abc123" and items[0].meta["question_type"] == "attribute"


def test_coco_adapter(tmp_path):
    caps = {"images": [{"id": 5, "file_name": "COCO_val2014_000000000005.jpg"}],
            "annotations": [{"id": 1, "image_id": 5, "caption": "A dog on grass."}]}
    inst = {"images": [{"id": 5}, {"id": 6}], "categories": [{"id": 18, "name": "dog"}],
            "annotations": [{"image_id": 5, "category_id": 18}]}
    (tmp_path / "c.json").write_text(json.dumps(caps), encoding="utf-8")
    (tmp_path / "i.json").write_text(json.dumps(inst), encoding="utf-8")
    ad = get_adapter("coco")
    s = ad.load_captions(tmp_path / "c.json")
    assert s[0].image_id == "5" and s[0].spans == []
    assert ad.load_instances(tmp_path / "i.json") == {"5": {"dog"}, "6": set()}


def test_unknown_adapter():
    with pytest.raises(KeyError):
        get_adapter("nope")
