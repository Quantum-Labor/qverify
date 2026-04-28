"""LLM backend for translation — constrained-generation via outlines."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from qverify.utils.logging import get_logger
from qverify.utils.models import TRANSLATOR_MODEL_ID


@runtime_checkable
class TranslationBackend(Protocol):
    """Anything callable from the Translator with ``generate(prompt)``."""

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str: ...


class Gemma4StructuredBackend:
    """Gemma 4 E2B via transformers + outlines — schema-constrained backend.

    The model is wrapped in an :mod:`outlines` ``Transformers`` adapter and a
    :class:`outlines.Generator` is constructed once with
    :class:`qverify.translator.schema.TranslationSchema` as the output type.
    Outlines compiles the schema into a token-level finite state machine and
    forces every emitted token to keep the running output schema-conformant.
    Result: 2B-parameter models that previously fell into garbage on retry
    (``\"3\\n\\u0060\\u0060\\u0060\\n\"``) now emit exactly one
    schema-valid JSON document — every time, no retries needed for syntax.

    The schema compilation happens once on the first :meth:`generate` call
    and is cached on the instance; subsequent translations pay only the
    normal generate-token cost.

    Both the tokenizer/model and the outlines model are loaded lazily on the
    first :meth:`generate` call so that importing this module (or the
    parent package) does not pull torch, transformers, or outlines into
    ``sys.modules``.
    """

    def __init__(
        self,
        model_id: str = TRANSLATOR_MODEL_ID,
        device: str = "auto",
    ) -> None:
        self._model_id: str = model_id
        self._device: str = device
        self._generator: Any = None
        self._log = get_logger("qverify.translator.gemma4_structured")

    def _load(self) -> None:
        if self._generator is not None:
            return

        # Heavy imports happen here, not at module level — preserves the
        # lazy-load contract enforced by Phase 5/6 unit tests.
        import outlines
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from qverify.translator.schema import TranslationSchema

        self._log.info("loading translator model %s", self._model_id)
        tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        # outlines's stub has narrower tokenizer typing than the runtime
        # accepts; transformers PreTrainedTokenizer instances work fine.
        model = outlines.from_transformers(hf_model, tokenizer)  # type: ignore[arg-type]
        self._generator = outlines.Generator(model, output_type=TranslationSchema)
        self._log.info("compiled schema into outlines generator")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Run constrained generation; return the JSON serialization.

        Outlines normally returns the parsed Pydantic instance directly when
        a schema is supplied. We re-serialize via :meth:`model_dump_json`
        so the public ``generate(prompt) -> str`` contract — which the
        :func:`qverify.translator.parser.parse_llm_output` consumer depends
        on — is preserved.
        """
        self._load()
        result = self._generator(prompt, max_new_tokens=max_new_tokens)
        # Outlines may return either the parsed Pydantic instance or its
        # JSON-string form depending on the schema/transport combo. Handle
        # both so subsequent transformer/outlines upgrades don't break us.
        if hasattr(result, "model_dump_json"):
            return str(result.model_dump_json())
        return str(result)


# Backward-compatibility alias — pre-Phase-6 callers and tests that import
# ``GemmaE2BBackend`` continue to work unchanged. The name is now slightly
# misleading: the default model is Gemma 4 E4B (see qverify.utils.models)
# because E2B could not reliably encode universal quantification. Pass an
# explicit ``model_id="google/gemma-4-E2B-it"`` to opt back into E2B.
GemmaE2BBackend = Gemma4StructuredBackend


class StubBackend:
    """Deterministic backend for tests — returns canned responses by exact prompt match."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses: dict[str, str] = dict(responses)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        if prompt not in self._responses:
            preview = prompt[:80].replace("\n", " ")
            raise KeyError(f"No stub response for prompt starting: {preview!r}...")
        return self._responses[prompt]
