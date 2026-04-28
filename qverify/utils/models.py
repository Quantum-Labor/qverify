"""Canonical model identifiers for the QVerify project.

All Gemma models are gated on Hugging Face. Users must accept the license at
https://huggingface.co/google before downloading any of them.
"""

from typing import Final

# Translator: model that converts NL reasoning steps into CNF formulas.
# Phase 6 used Gemma 4 E2B but it failed to grasp universal quantification
# (emitted constants where free variables were required). Bumped to E4B for
# semantic reliability; the additional 2-3x latency per call is acceptable
# given that Translator runs O(steps) times per reasoning loop. E2B remains
# available for callers that want lower latency and can tolerate
# first-order quality loss — pass model_id="google/gemma-4-E2B-it"
# explicitly when constructing the backend.
TRANSLATOR_MODEL_ID: Final[str] = "google/gemma-4-E4B-it"

# Reasoner baseline: thinking-mode model used for development and most benchmarks.
# Fits comfortably on a single 24 GB GPU in bf16.
REASONER_E4B_MODEL_ID: Final[str] = "google/gemma-4-E4B-it"

# Reasoner show-off: top-tier MoE model used for headline benchmark numbers.
# Requires distributed inference across multiple GPUs.
REASONER_26B_MOE_MODEL_ID: Final[str] = "google/gemma-4-26B-A4B-it"
