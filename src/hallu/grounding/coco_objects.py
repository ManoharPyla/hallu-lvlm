"""COCO 80-class synonym table, object extraction, and COCO ground truth access.

IMPORTANT: ``COCO_SYNONYMS`` is our own hand-written table. It APPROXIMATES the
synonym list used by the official CHAIR implementation (Rohrbach et al., 2018;
``synonyms.txt``) but is not identical. CHAIR numbers computed with it are
comparable within this repo, not exactly with published numbers. Swap in the
official list via ``set_synonyms`` for paper results.

Matching: lowercase, whole-word, longest phrase first (so "hot dog" is not "dog",
"teddy bear" is not "bear"); simple plural forms (+s, +es, y->ies) and a few
irregular plurals are accepted.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

COCO_SYNONYMS: dict[str, list[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child",
               "children", "kid", "baby", "player", "guy", "lady", "adult", "teenager",
               "skier", "surfer", "skateboarder", "rider", "pedestrian", "human", "toddler"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "car": ["car", "automobile", "sedan", "taxi", "suv", "van"],
    "motorcycle": ["motorcycle", "motorbike", "motor bike", "scooter", "moped"],
    "airplane": ["airplane", "plane", "jet", "aircraft", "airliner", "aeroplane"],
    "bus": ["bus", "trolley"],
    "train": ["train", "locomotive", "tram", "subway"],
    "truck": ["truck", "pickup", "lorry"],
    "boat": ["boat", "ship", "yacht", "canoe", "kayak", "sailboat", "ferry"],
    "traffic light": ["traffic light", "traffic signal", "stop light", "stoplight"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter", "meter"],
    "bench": ["bench"],
    "bird": ["bird", "pigeon", "seagull", "gull", "duck", "goose", "parrot", "crow", "sparrow"],
    "cat": ["cat", "kitten", "kitty"],
    "dog": ["dog", "puppy", "pup", "doggy"],
    "horse": ["horse", "pony", "foal"],
    "sheep": ["sheep", "lamb", "ram"],
    "cow": ["cow", "cattle", "bull", "calf", "oxen", "ox"],
    "elephant": ["elephant"],
    "bear": ["bear", "grizzly", "polar bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "back pack", "rucksack"],
    "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse", "hand bag"],
    "tie": ["tie", "necktie", "bow tie"],
    "suitcase": ["suitcase", "luggage", "suit case"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["ball", "soccer ball", "football", "baseball", "basketball", "tennis ball",
                    "volleyball", "sports ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "mitt", "glove"],
    "skateboard": ["skateboard", "skate board"],
    "surfboard": ["surfboard", "surf board"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass"],
    "cup": ["cup", "mug"],
    "fork": ["fork"],
    "knife": ["knife", "knives"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "burger", "sub", "hamburger"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake", "cupcake", "cheesecake"],
    "chair": ["chair", "stool"],
    "couch": ["couch", "sofa"],
    "potted plant": ["potted plant", "houseplant", "house plant", "plant"],
    "bed": ["bed"],
    "dining table": ["dining table", "table", "desk"],
    "toilet": ["toilet"],
    "tv": ["tv", "television", "monitor", "tv set"],
    "laptop": ["laptop", "computer", "notebook computer"],
    "mouse": ["mouse"],
    "remote": ["remote", "remote control", "controller"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "cellphone", "phone", "smartphone", "mobile phone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book", "novel"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy", "stuffed animal"],
    "hair drier": ["hair drier", "hair dryer", "hairdryer", "blow dryer"],
    "toothbrush": ["toothbrush", "tooth brush"],
}
COCO_CLASSES: list[str] = list(COCO_SYNONYMS)
assert len(COCO_CLASSES) == 80

_IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child",
              "knives": "knife", "mice": "mouse", "geese": "goose", "oxen": "ox"}

_lookup: dict[str, str] = {}
_pattern: re.Pattern | None = None


def _plurals(w: str) -> set[str]:
    head, _, last = w.rpartition(" ")
    forms = {last, last + "s", last + "es"}
    if last.endswith("y") and last[-2:-1] not in "aeiou":
        forms.add(last[:-1] + "ies")
    if last.endswith("f"):
        forms.add(last[:-1] + "ves")
    if last.endswith("fe"):
        forms.add(last[:-2] + "ves")
    return {(head + " " + f).strip() for f in forms}


def set_synonyms(table: dict[str, list[str]]) -> None:
    """Install a synonym table (e.g. the official CHAIR list) and rebuild the matcher."""
    global _pattern
    _lookup.clear()
    for cls, syns in table.items():
        for s in syns:
            for f in _plurals(s.lower()):
                _lookup.setdefault(f, cls)
    for irr, sing in _IRREGULAR.items():
        if sing in _lookup and irr not in _lookup:
            _lookup[irr] = _lookup[sing]
    alts = sorted(_lookup, key=len, reverse=True)
    _pattern = re.compile(r"\b(" + "|".join(re.escape(a) for a in alts) + r")\b")


set_synonyms(COCO_SYNONYMS)


def normalize(word: str) -> str | None:
    """Map a word/phrase to its COCO class or None."""
    return _lookup.get(word.strip().lower())


def extract_objects(text: str) -> list[tuple[str, str, tuple[int, int]]]:
    """Return (surface_word, coco_class, (start, end)) for each COCO object mention."""
    assert _pattern is not None
    return [(text[m.start():m.end()], _lookup[m.group(1)], (m.start(), m.end()))
            for m in _pattern.finditer(text.lower())]


class CocoGT:
    """Ground-truth objects per image from COCO instances (+ optionally captions) JSON.

    ``gt_objects`` = classes in instance annotations; ``caption_objects`` = classes
    mentioned in any reference caption. Image ids are compared as strings.
    """

    def __init__(self, instances_json: str | Path | dict | None = None,
                 captions_json: str | Path | dict | None = None):
        self._inst: dict[str, set[str]] = defaultdict(set)
        self._caps: dict[str, set[str]] = defaultdict(set)
        if instances_json is not None:
            d = self._load(instances_json)
            cats = {c["id"]: c["name"] for c in d.get("categories", [])}
            for a in d.get("annotations", []):
                name = cats.get(a["category_id"])
                if name:
                    self._inst[str(a["image_id"])].add(name)
        if captions_json is not None:
            d = self._load(captions_json)
            for a in d.get("annotations", []):
                self._caps[str(a["image_id"])].update(c for _, c, _ in extract_objects(a["caption"]))

    @staticmethod
    def _load(src):
        if isinstance(src, dict):
            return src
        with open(src, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _key(image_id) -> str:
        s = str(image_id)
        m = re.search(r"(\d+)$", Path(s).stem) if not s.isdigit() else None
        return str(int(m.group(1))) if m else (str(int(s)) if s.isdigit() else s)

    def gt_objects(self, image_id) -> set[str]:
        return set(self._inst.get(self._key(image_id), set()))

    def caption_objects(self, image_id) -> set[str]:
        return set(self._caps.get(self._key(image_id), set()))

    def has_image(self, image_id) -> bool:
        k = self._key(image_id)
        return k in self._inst or k in self._caps
