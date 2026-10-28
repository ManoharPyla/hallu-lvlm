from hallu.grounding.checker import AnnotationBackend, GroundingChecker, GroundingVerdict, VQABackend
from hallu.grounding.coco_objects import COCO_CLASSES, CocoGT, extract_objects
from hallu.grounding.source import SourceLabel, estimate_source

INST = {"categories": [{"id": 1, "name": "dog"}, {"id": 2, "name": "person"}],
        "annotations": [{"image_id": 7, "category_id": 1}]}
CAPS = {"annotations": [{"image_id": 7, "caption": "A man walks a dog."}]}


def test_extract():
    assert len(COCO_CLASSES) == 80
    objs = extract_objects("A hot dog and two dogs near a teddy bear.")
    assert [c for _, c, _ in objs] == ["hot dog", "dog", "teddy bear"]


def test_annotation_backend():
    b = AnnotationBackend(CocoGT(INST, CAPS))
    assert b.check("dog", 7)[0] == GroundingVerdict.SUPPORTED
    assert b.check("man", 7)[0] == GroundingVerdict.UNKNOWN       # caption-only
    assert b.check("cat", 7)[0] == GroundingVerdict.CONTRADICTED
    assert b.check("red", 7)[0] == GroundingVerdict.UNKNOWN       # not a COCO object
    assert b.check("cat", 999)[0] == GroundingVerdict.UNKNOWN     # unknown image


class FakeLVLM:
    def answer(self, image, q):
        return "Yes" if "dog" in q else "No, there is not."


def test_vqa_and_checker_disagreement():
    v = VQABackend(FakeLVLM())
    assert v.check("cat", image=object())[0] == GroundingVerdict.CONTRADICTED
    assert v.check("cat")[0] == GroundingVerdict.UNKNOWN
    ch = GroundingChecker([AnnotationBackend(CocoGT(INST, CAPS)), VQABackend(lambda: None)])
    assert ch.check("cat", 7)[0] == GroundingVerdict.CONTRADICTED


def test_source_default_unknown():
    assert estimate_source({}) == SourceLabel.UNKNOWN
    assert estimate_source({"image_attn_mass": 0.01}) == SourceLabel.VISUAL_UNDERGROUNDING
