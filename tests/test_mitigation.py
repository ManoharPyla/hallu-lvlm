import re

import numpy as np

from hallu.grounding.checker import AnnotationBackend, GroundingChecker
from hallu.grounding.coco_objects import CocoGT
from hallu.mitigation.minimal_edit import char_edit_distance, expand_to_words, remove_span
from hallu.mitigation.policies import CorrectionResult, NoMitigation, SelectiveCorrection
from hallu.mitigation.verification import correction_eval

INST = {"categories": [{"id": 1, "name": "dog"}], "annotations": [{"image_id": 7, "category_id": 1}]}


def toks(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]


def test_remove_span_grammar():
    t = "A dog and a cat on the grass."
    s = t.index("cat")
    assert remove_span(t, s, s + 3) == "A dog on the grass."
    assert expand_to_words("doghouse", 2, 4) == (0, 8)
    assert char_edit_distance("kitten", "sitting") == 3


def test_selective_whole_words_and_unknown_keep():
    t = "A dog and a cat sit with a zebra-ish blanket."
    offs = toks(t)
    probs = np.zeros(len(offs))
    ci = [i for i, (a, b) in enumerate(offs) if t[a:b] == "cat"][0]
    probs[ci] = 0.9
    checker = GroundingChecker([AnnotationBackend(CocoGT(INST, {"annotations": []}))])
    pol = SelectiveCorrection(threshold=0.5, checker=checker)
    r = pol.apply(dict(response=t, offsets=offs, probs=probs, image_id=7))
    assert "cat" not in r.corrected and "dog" in r.corrected
    assert r.decisions[0]["verdict"] == "contradicted"
    # flag a subword-ish non-COCO token -> UNKNOWN -> kept unchanged, never invented
    probs2 = np.zeros(len(offs)); probs2[-2] = 0.9
    r2 = pol.apply(dict(response=t, offsets=offs, probs=probs2, image_id=7))
    assert r2.corrected == t
    pol_h = SelectiveCorrection(threshold=0.5, checker=checker, hedge_unknown=True)
    r3 = pol_h.apply(dict(response=t, offsets=offs, probs=probs2, image_id=7))
    assert "possibly" in r3.corrected and "blanket" in r3.corrected


def test_unchanged_incorrect_not_success():
    t = "A dog and a cat."
    gold = [dict(start=t.index("cat"), end=t.index("cat") + 3, text="cat")]
    r = NoMitigation().apply(dict(response=t))
    ev = correction_eval(gold, r, {"dog"})
    assert ev["halluc_removed"] == 0.0 and ev["completeness"] == 0.0
    fixed = CorrectionResult(t, "A dog.", [], [])
    ev2 = correction_eval(gold, fixed, {"dog"})
    assert ev2["halluc_removed"] == 1.0 and ev2["completeness"] == 1.0
    assert ev2["supported_preserved"] == 1.0 and ev2["new_halluc"] == 0
    bad = CorrectionResult(t, "A dog and a horse.", [], [])
    assert correction_eval(gold, bad, {"dog"})["new_halluc"] == 1
