"""Hugging Face LVLM wrapper (LLaVA-OneVision / LLaVA-1.5).

``transformers`` and ``PIL`` are imported lazily so the rest of the package (and the
unit tests) work without them.

Notes
-----
* Per-token logits are reduced to scalar statistics (logprob, entropy, margin, rank) to
  avoid storing T x V arrays. ``GenOutput.logits`` holds a dict of these stats.
* ``image_attn_mass`` (fraction of last-layer attention from each response token onto
  image-token positions) is only computed when ``output_attentions=True`` (forces eager
  attention). It is a descriptive signal, NOT proof that a token is grounded in the image.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sympy import content

from hallu.features.token_stats import token_stats
from hallu.models.errors import ModelUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"


@dataclass
class GenOutput:
    text: str
    token_ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]
    logits: dict[str, np.ndarray]  # stats: logprob, entropy, margin, rank (each [T])
    hidden: np.ndarray  # [T, H] float16
    image_attn_mass: np.ndarray | None
    latency_s: float


def incremental_offsets(tokenizer: Any, token_ids: list[int]):
    """Decode tokens incrementally and map them to character offsets.

    Raises ValueError when decoded prefixes cannot be aligned reliably.
    """
    prefixes = [
        tokenizer.decode(token_ids[:i + 1], skip_special_tokens=True)
        for i in range(len(token_ids))
    ]

    text = prefixes[-1] if prefixes else ""
    offsets = []
    tokens = []
    prev = ""

    for i, cur in enumerate(prefixes):
        if not cur.startswith(prev):
            raise ValueError(
                f"Unstable decoded prefix at token {i}: "
                f"previous={prev!r}, current={cur!r}"
            )

        start = len(prev)
        end = len(cur)
        tokens.append(cur[start:end])
        offsets.append((start, end))
        prev = cur

    return text, tokens, offsets

class LVLM:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "float16",
        load_in_4bit: bool = False,
        hidden_layers: tuple[int, ...] = (-1,),
        output_attentions: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.hidden_layers = tuple(hidden_layers)
        self.output_attentions = output_attentions
        self.model: Any = None
        self.processor: Any = None
        self.image_token_id: int | None = None
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        try:
            import torch
            import transformers
        except ImportError as e:
            raise ModelUnavailableError(f"transformers/torch not importable: {e}") from e
        mid = self.model_id.lower()
        if "onevision" in mid:
            cls = getattr(transformers, "LlavaOnevisionForConditionalGeneration", None)
        elif "llava" in mid:
            cls = getattr(transformers, "LlavaForConditionalGeneration", None)
        else:
            cls = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(
                transformers, "AutoModelForVision2Seq", None)
        if cls is None:
            raise ModelUnavailableError(f"transformers {transformers.__version__} lacks a class for {self.model_id}",
                                        "Upgrade: `pip install -U transformers`.")
        torch_dtype = getattr(torch, self.dtype, torch.float16)
        if self.device == "cpu" and torch_dtype == torch.float16:
            torch_dtype = torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        if self.load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ModelUnavailableError("bitsandbytes missing for 4-bit", "`pip install bitsandbytes` (CUDA only).") from e
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
        if self.output_attentions:
            kwargs["attn_implementation"] = "eager"
        else:
            kwargs["attn_implementation"] = "sdpa"
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            self.model = cls.from_pretrained(self.model_id, **kwargs)
        except Exception as e:
            raise ModelUnavailableError(f"could not load {self.model_id}: {e}") from e
        if self.device != "auto" and not self.load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None)

    @property
    def tokenizer(self) -> Any:
        return getattr(self.processor, "tokenizer", self.processor)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_image(image: Any) -> Any:
        if image is None or not isinstance(image, str):
            return image
        try:
            from PIL import Image
        except ImportError as e:
            raise ModelUnavailableError("Pillow missing", "`pip install pillow`") from e
        return Image.open(image).convert("RGB")

    def _inputs(self, image: Any, prompt: str, response: str | None = None) -> Any:
        img = self._load_image(image)

        # Dataset prompts may already contain an image placeholder.
        # The chat template inserts its own marker when an image is supplied.
        if img is not None:
            prompt = prompt.replace("<image>", "").strip()

        content: list[dict] = [{"type": "image"}] if img is not None else []
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        if response is not None:
            text = text + response
        kw: dict[str, Any] = {"text": text, "return_tensors": "pt"}
        if img is not None:
            kw["images"] = img
        inputs = self.processor(**kw)
        return inputs.to(next(self.model.parameters()).device)

    def _attn_mass(self, attn_last: Any, input_ids: Any, q_positions: Any) -> np.ndarray | None:
        """attn_last: [B, heads, Q, K]. Head-mean attention mass onto image-token keys."""
        if attn_last is None or self.image_token_id is None:
            return None
        img_mask = input_ids[0] == self.image_token_id
        if not bool(img_mask.any()):
            return None
        a = attn_last[0].float().mean(0)[q_positions]  # [q, K]
        K = min(a.shape[-1], img_mask.shape[0])
        return a[:, :K][:, img_mask[:K]].sum(-1).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ API
    def generate(self, image: Any, prompt: str, max_new_tokens: int = 128) -> GenOutput:
        import torch

        t0 = time.perf_counter()
        inputs = self._inputs(image, prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                output_hidden_states=True, output_attentions=self.output_attentions,
            )
        plen = inputs["input_ids"].shape[1]
        T = len(out.scores)
        ids = out.sequences[0, plen:plen + T].tolist()
        stats = token_stats(torch.stack([s[0] for s in out.scores]).float().cpu().numpy(), np.asarray(ids))
        hid = [torch.cat([step[l][0, -1] for l in self.hidden_layers], -1).float().cpu().numpy()
               for step in out.hidden_states[:T]]
        hidden = np.stack(hid).astype(np.float16) if hid else np.zeros((0, 0), np.float16)
        mass = None
        if self.output_attentions and getattr(out, "attentions", None):
            vals = []
            for step in out.attentions[:T]:
                m = self._attn_mass(step[-1], inputs["input_ids"], torch.tensor([-1]))
                vals.append(float(m[0]) if m is not None else np.nan)
            mass = np.asarray(vals, np.float32)
        text, tokens, offsets = incremental_offsets(self.tokenizer, ids)
        return GenOutput(text, ids, tokens, offsets, stats, hidden, mass, time.perf_counter() - t0)

    def teacher_force(self, image: Any, prompt: str, response: str) -> GenOutput:
        """Single forward pass over prompt+response; returns stats for response positions."""
        import torch

        t0 = time.perf_counter()
        plen = self._inputs(image, prompt)["input_ids"].shape[1]
        inputs = self._inputs(image, prompt, response)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True,
                             output_attentions=self.output_attentions, return_dict=True)
        input_ids = inputs["input_ids"]
        ids = input_ids[0, plen:].tolist()
        T = len(ids)
        pred = out.logits[0, plen - 1: plen - 1 + T].float().cpu().numpy()  # pos p predicts p+1
        stats = token_stats(pred, np.asarray(ids))
        hidden = torch.cat([out.hidden_states[l][0, plen:plen + T] for l in self.hidden_layers], -1)
        hidden = hidden.float().cpu().numpy().astype(np.float16)
        mass = None
        if self.output_attentions and getattr(out, "attentions", None):
            mass = self._attn_mass(out.attentions[-1], input_ids, torch.arange(plen, plen + T))
        text, tokens, offsets = incremental_offsets(self.tokenizer, ids)
        if text != response:
            logger.debug("teacher_force: decoded text differs from response (%d vs %d chars)", len(text), len(response))
        return GenOutput(text, ids, tokens, offsets, stats, hidden, mass, time.perf_counter() - t0)

    def answer(self, image: Any, question: str, max_new_tokens: int = 16) -> str:
        return self.generate(image, question, max_new_tokens=max_new_tokens).text.strip()
