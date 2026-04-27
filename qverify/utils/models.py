"""Canonical model identifiers for the QVerify project.

All Gemma models are gated on Hugging Face. Users must accept the license at
https://huggingface.co/google before downloading any of them.
"""

from typing import Final

# Translator: small model that converts NL reasoning steps into CNF formulas.
# Called frequently inside the reasoning loop, so we want the smallest viable size.
TRANSLATOR_MODEL_ID: Final[str] = "google/gemma-4-E2B-it"

# Reasoner baseline: thinking-mode model used for development and most benchmarks.
# Fits comfortably on a single 24 GB GPU in bf16.
REASONER_E4B_MODEL_ID: Final[str] = "google/gemma-4-E4B-it"

# Reasoner show-off: top-tier MoE model used for headline benchmark numbers.
# Requires distributed inference across multiple GPUs.
REASONER_26B_MOE_MODEL_ID: Final[str] = "google/gemma-4-26B-A4B-it"
