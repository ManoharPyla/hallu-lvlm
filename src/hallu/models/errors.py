"""Errors raised by model wrappers."""


class ModelUnavailableError(RuntimeError):
    """Raised when an LVLM cannot be loaded (missing deps or weights)."""

    INSTALL_HINT = (
        "Install the optional model deps: `pip install transformers accelerate pillow` "
        "(and `bitsandbytes` for 4-bit). Weights are downloaded from the Hugging Face Hub on "
        "first use; for offline use pre-download with `huggingface-cli download <model_id>`."
    )

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(f"{message}\nHint: {hint or self.INSTALL_HINT}")
