"""LLM backend for translation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from qverify.utils.logging import get_logger
from qverify.utils.models import TRANSLATOR_MODEL_ID


@runtime_checkable
class TranslationBackend(Protocol):
    """Anything callable from the Translator with ``generate(prompt)``."""

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str: ...


class GemmaE2BBackend:
    """Gemma 4 E2B via transformers — production backend.

    Both the tokenizer and the model are loaded lazily on the first call to
    :meth:`generate` so that importing this module (or the parent package)
    does not pull torch into memory or hit HuggingFace.
    """

    def __init__(
        self,
        model_id: str = TRANSLATOR_MODEL_ID,
        device: str = "auto",
    ) -> None:
        self._model_id: str = model_id
        self._device: str = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._log = get_logger("qverify.translator.gemma")

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._log.info("loading translator model %s", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Greedily decode ``max_new_tokens`` from the prompt and strip the prompt prefix."""
        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        decoded: str = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            return decoded[len(prompt) :]
        return decoded


class StubBackend:
    """Deterministic backend for tests — returns canned responses by exact prompt match."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses: dict[str, str] = dict(responses)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if prompt not in self._responses:
            preview = prompt[:80].replace("\n", " ")
            raise KeyError(f"No stub response for prompt starting: {preview!r}...")
        return self._responses[prompt]
