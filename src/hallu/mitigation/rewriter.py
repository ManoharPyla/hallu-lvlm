"""LVLM-based rewriting with a no-model fallback.

Without a model the rewriter NEVER generates content: it only deletes
(CONTRADICTED) or hedges (UNKNOWN, if enabled) and otherwise keeps the text.
"""
from __future__ import annotations

import logging
import re

from hallu.mitigation.minimal_edit import expand_to_words, hedge_span, remove_span, replace_span

logger = logging.getLogger(__name__)

SPAN_REWRITE_PROMPT = (
    "Look at the image. In the description below, the phrase \"{span}\" may be wrong.\n"
    "Description: {response}\n"
    "Reply with ONLY a short replacement phrase that is visible in the image, "
    "or the single word NONE if the phrase should simply be removed."
)
WHOLE_REWRITE_PROMPT = (
    "Look at the image. The following answer to \"{prompt}\" may contain details that are "
    "not in the image.\nAnswer: {response}\n"
    "Rewrite the answer, removing anything not visible in the image and changing as little "
    "as possible. Do not add new details. Output only the rewritten answer."
)
MAX_REPLACEMENT_WORDS = 6


class Rewriter:
    def __init__(self, lvlm=None, hedge_word: str = "possibly"):
        self.lvlm = lvlm
        self.hedge_word = hedge_word

    def delete(self, text: str, start: int, end: int) -> str:
        return remove_span(text, start, end)

    def hedge(self, text: str, start: int, end: int) -> str:
        return hedge_span(text, start, end, self.hedge_word)

    def rewrite_span(self, text: str, start: int, end: int, image=None) -> tuple[str, str]:
        """Rewrite a CONTRADICTED span. Returns (new_text, action in {'replace','delete'})."""
        s, e = expand_to_words(text, start, end)
        if self.lvlm is None or image is None:
            return self.delete(text, s, e), "delete"
        try:
            out = self.lvlm.answer(image, SPAN_REWRITE_PROMPT.format(span=text[s:e], response=text))
        except Exception as ex:
            logger.warning("span rewrite failed (%s); deleting instead", ex)
            return self.delete(text, s, e), "delete"
        out = out.strip().strip('."\'')
        if (not out or out.upper() == "NONE" or len(out.split()) > MAX_REPLACEMENT_WORDS
                or out.lower() == text[s:e].lower()):
            return self.delete(text, s, e), "delete"
        return replace_span(text, s, e, out), "replace"

    def whole_rewrite(self, prompt: str, response: str, image=None) -> str:
        if self.lvlm is None or image is None:
            logger.warning("whole rewrite needs an LVLM and image; returning response unchanged")
            return response
        q = WHOLE_REWRITE_PROMPT.format(prompt=prompt, response=response)
        if hasattr(self.lvlm, "generate"):
            out = self.lvlm.generate(image, q)
            out = getattr(out, "text", out)
        else:
            out = self.lvlm.answer(image, q)
        return re.sub(r"\s+", " ", str(out)).strip() or response
