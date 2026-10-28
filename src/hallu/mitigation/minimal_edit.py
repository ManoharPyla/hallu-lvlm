"""Minimal, word-level text edits with light grammar cleanup.

All edits operate on WHOLE words: any char range is first expanded to word
boundaries, so a subword is never cut.
"""
from __future__ import annotations

import difflib
import re

_ARTICLES = ("a", "an", "the", "some", "two", "three", "several", "many", "one")
_WORD = re.compile(r"[\w'-]")


def expand_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand [start, end) so it begins and ends on word boundaries; strips outer whitespace."""
    start, end = max(0, start), min(len(text), end)
    while start > 0 and _WORD.match(text[start - 1]) and start < len(text) and _WORD.match(text[start]):
        start -= 1
    while end < len(text) and end > 0 and _WORD.match(text[end - 1]) and _WORD.match(text[end]):
        end += 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _phrase_range(text: str, start: int, end: int) -> tuple[int, int]:
    """Word-expanded span plus a directly preceding article/determiner."""
    s, e = expand_to_words(text, start, end)
    m = re.search(r"\b(" + "|".join(_ARTICLES) + r")\s+$", text[:s], flags=re.I)
    if m:
        s = m.start()
    return s, e


def removal_range(text: str, start: int, end: int) -> tuple[int, int]:
    """Char range to delete when removing the object phrase at [start, end)."""
    s, e = _phrase_range(text, start, end)
    before, after = text[:s], text[e:]
    m = re.search(r"(,\s*(and\s+)?|\s+and\s+|\s+or\s+)$", before, flags=re.I)
    if m and not re.match(r"\s*,", after):
        return m.start(), e  # "X and <span>" / "X, <span>"
    m2 = re.match(r"(\s*,\s*(and\s+)?|\s+and\s+|\s+or\s+)", after, flags=re.I)
    if m2:
        return s, e + m2.end()  # "<span> and Y"
    m3 = re.match(r"\s+(with|on|in|near|next to|beside)\s*$", before[-14:], flags=re.I)
    if m3 and re.match(r"\s*[.!?]", after):
        # "... sitting next to <span>." -> drop dangling preposition too
        return s - len(m3.group(0)), e
    return s, e


def _cleanup(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*([.;:!?])", r"\1", text)
    text = re.sub(r"\b(a|an|the)\s+([.,;:!?]|$)", r"\2", text, flags=re.I)
    text = re.sub(r"\ba ([aeiouAEIOU])", r"an \1", text)
    text = re.sub(r"\ban ([^aeiouAEIOU\W])", r"a \1", text)
    return text.strip()


def remove_span(text: str, start: int, end: int) -> str:
    """Delete the object phrase covering [start, end) (whole words + article + list glue)."""
    s, e = removal_range(text, start, end)
    return _cleanup(text[:s] + " " + text[e:])


def replace_span(text: str, start: int, end: int, new: str) -> str:
    s, e = expand_to_words(text, start, end)
    return _cleanup(text[:s] + new + text[e:])


def hedge_span(text: str, start: int, end: int, hedge: str = "possibly") -> str:
    """Insert a hedge word before the phrase; content is kept, nothing is invented."""
    s, _ = _phrase_range(text, start, end)
    return text[:s] + hedge + " " + text[s:]


def char_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (characters)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def diff_spans(a: str, b: str) -> list[tuple[str, int, int, int, int]]:
    """Non-equal opcodes (tag, a_start, a_end, b_start, b_end) between a and b, word-granular."""
    ta = [(m.start(), m.end()) for m in re.finditer(r"\S+", a)]
    tb = [(m.start(), m.end()) for m in re.finditer(r"\S+", b)]
    wa = [a[s:e] for s, e in ta]
    wb = [b[s:e] for s, e in tb]
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        a_s = ta[i1][0] if i1 < len(ta) else len(a)
        a_e = ta[i2 - 1][1] if i2 > i1 else a_s
        b_s = tb[j1][0] if j1 < len(tb) else len(b)
        b_e = tb[j2 - 1][1] if j2 > j1 else b_s
        out.append((tag, a_s, a_e, b_s, b_e))
    return out
