import re

import pytest

from hallu.data.alignment import (
    IGNORE_INDEX,
    char_labels_to_token_labels,
    group_tokens_to_spans,
    word_index_to_char_spans,
)
from hallu.data.schema import HallSpan

TEXT = "A red dog sits."


def span(text, phrase, htype="attribute"):
    s = text.index(phrase)
    return HallSpan(s, s + len(phrase), phrase, htype)


def test_fast_tokenizer_offsets_phrase_span():
    # <s> A red dog sits .  </s>
    offsets = [(0, 0), (0, 1), (2, 5), (6, 9), (10, 14), (14, 15), (0, 0)]
    labels = char_labels_to_token_labels(offsets, [span(TEXT, "red dog")])
    assert labels == [IGNORE_INDEX, 0, 1, 1, 0, 0, IGNORE_INDEX]


def test_sentencepiece_style_leading_space_offsets():
    # Llama-style tokens include the leading space: " red" -> (1,5)
    offsets = [(0, 1), (1, 5), (5, 9), (9, 14), (14, 15)]
    labels = char_labels_to_token_labels(offsets, [span(TEXT, "red dog")])
    assert labels == [0, 1, 1, 0, 0]


def test_subword_pieces_all_labeled():
    text = "A giraffe stands."
    # gir ##aff ##e
    offsets = [(0, 1), (2, 5), (5, 8), (8, 9), (10, 16), (16, 17)]
    labels = char_labels_to_token_labels(offsets, [span(text, "giraffe", "object")])
    assert labels == [0, 1, 1, 1, 0, 0]


def test_special_mask_overrides():
    offsets = [(0, 1), (2, 5), (6, 9)]
    labels = char_labels_to_token_labels(offsets, [span(TEXT, "red")], special_mask=[True, False, False])
    assert labels == [IGNORE_INDEX, 1, 0]


def test_special_mask_length_mismatch():
    with pytest.raises(ValueError):
        char_labels_to_token_labels([(0, 1)], [], special_mask=[True, False])


def test_punctuation_inside_and_outside_span():
    text = "a big, red dog."
    sp = span(text, "big, red", "attribute")
    offsets = [(0, 1), (2, 5), (5, 6), (7, 10), (11, 14), (14, 15)]
    labels = char_labels_to_token_labels(offsets, [sp])
    assert labels == [0, 1, 1, 1, 0, 0]  # comma inside span -> 1, final period -> 0


def test_min_overlap_fraction():
    # token "reddish" partially overlapped by span "red"
    text = "reddish"
    offsets = [(0, 7)]
    sp = HallSpan(0, 3, "red", "attribute")
    assert char_labels_to_token_labels(offsets, [sp], min_overlap=0.0) == [1]
    assert char_labels_to_token_labels(offsets, [sp], min_overlap=0.5) == [0]


def test_no_spans_all_zero():
    assert char_labels_to_token_labels([(0, 1), (2, 5)], []) == [0, 0]


def test_none_offset_ignored():
    assert char_labels_to_token_labels([None, (0, 1)], [HallSpan(0, 1, "A")]) == [IGNORE_INDEX, 1]


def test_multiple_spans():
    text = "A red dog and a blue cat."
    spans = [span(text, "red"), span(text, "cat", "object")]
    offsets = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
    labels = char_labels_to_token_labels(offsets, spans)
    toks = [text[s:e] for s, e in offsets]
    assert [t for t, l in zip(toks, labels) if l == 1] == ["red", "cat"]


def test_word_index_to_char_spans_merges_consecutive_and_strips_punct():
    out = word_index_to_char_spans(TEXT, [1, 2], "attribute")
    assert len(out) == 1 and out[0].text == "red dog"
    assert TEXT[out[0].start:out[0].end] == "red dog"
    out = word_index_to_char_spans(TEXT, 3, "relationship")
    assert out[0].text == "sits"  # trailing period stripped
    out = word_index_to_char_spans(TEXT, [1, 3])
    assert [s.text for s in out] == ["red", "sits"]


def test_word_index_out_of_range():
    with pytest.raises(IndexError):
        word_index_to_char_spans(TEXT, [10])


def test_group_tokens_to_spans_expands_words():
    text = "A giraffe stands near a red dog."
    offsets = [(0, 1), (2, 5), (5, 8), (8, 9), (10, 16), (17, 21), (22, 23), (24, 27), (28, 31), (31, 32)]
    flags = [0, 0, 1, 0, 0, 0, 0, 1, 1, 0]  # only "aff" flagged, then "red dog"
    spans = group_tokens_to_spans(offsets, flags, text)
    assert [text[s:e] for s, e in spans] == ["giraffe", "red dog"]


def test_group_tokens_gap():
    text = "red big dog"
    offsets = [(0, 3), (4, 7), (8, 11)]
    assert [text[s:e] for s, e in group_tokens_to_spans(offsets, [1, 0, 1], text, max_gap=1)] == ["red", "dog"]
    assert [text[s:e] for s, e in group_tokens_to_spans(offsets, [1, 0, 1], text, max_gap=2)] == ["red big dog"]


def test_roundtrip_labels_to_spans():
    offsets = [(0, 1), (1, 5), (5, 9), (9, 14), (14, 15)]
    labels = char_labels_to_token_labels(offsets, [span(TEXT, "red dog")])
    spans = group_tokens_to_spans(offsets, [l == 1 for l in labels], TEXT)
    assert [TEXT[s:e] for s, e in spans] == ["red dog"]
