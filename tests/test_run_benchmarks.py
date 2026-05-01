"""Tests for the benchmark CLI's translation-cache plumbing.

The actual Gemma 4 E2B path is GPU-only and out of scope for CI; these
tests exercise the cache, force flag, failure counting, and metric
plumbing using a fake translator stubbed in via monkeypatch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from qverify.eval.datasets import DatasetExample
from qverify.translator.cnf import CNF, Clause, Literal


def _load_run_benchmarks() -> ModuleType:
    """Import scripts/run_benchmarks.py as a module by file path."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("_run_benchmarks", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_run_benchmarks"] = module
    spec.loader.exec_module(module)
    return module


_RB = _load_run_benchmarks()


def _toy_cnf(predicate: str = "P") -> CNF:
    return CNF(clauses=(Clause(literals=(Literal(predicate=predicate),)),))


def _toy_example(example_id: str = "ex1") -> DatasetExample:
    return DatasetExample(
        id=example_id,
        premises=("Bird is bird.",),
        hypothesis="Bird is bird.",
        label="consistent",
        source="ruletaker",
        rendered_cnf=None,
    )


def test_cnf_round_trips_through_dict() -> None:
    cnf = CNF(
        clauses=(
            Clause(
                literals=(
                    Literal(predicate="P"),
                    Literal(predicate="Q", negated=True),
                )
            ),
            Clause(
                literals=(Literal(predicate="Cat", args=("Tom",)),),
            ),
        )
    )
    raw = _RB._cnf_to_dict(cnf)
    decoded = _RB._cnf_from_dict(raw)
    assert decoded == cnf


def test_translation_cache_path_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_RB, "_TRANSLATION_CACHE_ROOT", tmp_path)
    p = _RB._translation_cache_path("ruletaker", 1, "dev")
    assert p == tmp_path / "ruletaker" / "depth-1" / "dev.json"


def test_load_translation_cache_returns_empty_when_missing(tmp_path: Path) -> None:
    cache = _RB._load_translation_cache(tmp_path / "missing.json")
    assert cache == {}


def test_load_translation_cache_recovers_from_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert _RB._load_translation_cache(bad) == {}


def test_save_translation_cache_writes_json(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "cache.json"
    _RB._save_translation_cache(p, {"ex1": _RB._cnf_to_dict(_toy_cnf())})
    assert p.exists()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "ex1" in raw


def _build_translate_with_fake_translator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fake_translate_factory: Any,
    force: bool = False,
) -> tuple[Any, dict[str, int]]:
    """Build the gemma-e2b translate callback, but stub the LLM with a fake."""
    counter = {"calls": 0}
    monkeypatch.setattr(_RB, "_TRANSLATION_CACHE_ROOT", tmp_path)

    # Stub torch so the device-detection branch never tries CUDA.
    fake_torch = ModuleType("torch")
    fake_torch.cuda = ModuleType("cuda")  # type: ignore[attr-defined]
    fake_torch.cuda.is_available = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Build a fake Translator class drop-in.
    class _FakeTranslator:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def translate(self, statement: str) -> Any:
            counter["calls"] += 1
            return fake_translate_factory(statement)

    class _FakeBackend:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    fake_pkg = ModuleType("qverify.translator")
    fake_pkg.Translator = _FakeTranslator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qverify.translator", fake_pkg)

    fake_llm = ModuleType("qverify.translator.llm")
    fake_llm.GemmaE2BBackend = _FakeBackend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qverify.translator.llm", fake_llm)

    fn = _RB._build_gemma_e2b_translate(
        dataset="ruletaker", depth=1, split="dev", force_translate=force
    )
    return fn, counter


def test_translate_callback_caches_results_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qverify.translator.types import TranslationResult
    from qverify.verifier._universe import Universe

    def factory(_statement: str) -> TranslationResult:
        return TranslationResult(
            cnf=CNF(clauses=(Clause(literals=(Literal(predicate="A"),)),)),
            universe=Universe(constants=()),
        )

    fn, counter = _build_translate_with_fake_translator(
        monkeypatch, tmp_path, fake_translate_factory=factory
    )

    example = _toy_example("rt_1")
    result1 = fn(example)
    assert isinstance(result1, CNF)
    assert counter["calls"] >= 1

    # Flush the in-memory cache to disk so a fresh callback can read it.
    fn.flush()

    cache_path = _RB._translation_cache_path("ruletaker", 1, "dev")
    assert cache_path.exists()
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "rt_1" in raw


def test_translate_callback_cache_hit_skips_translator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qverify.translator.types import TranslationResult
    from qverify.verifier._universe import Universe

    # Pre-populate the cache so the translator should never be called.
    monkeypatch.setattr(_RB, "_TRANSLATION_CACHE_ROOT", tmp_path)
    cache_path = _RB._translation_cache_path("ruletaker", 1, "dev")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"rt_cached": _RB._cnf_to_dict(_toy_cnf("Cached"))}),
        encoding="utf-8",
    )

    def factory(_statement: str) -> TranslationResult:
        return TranslationResult(
            cnf=CNF(clauses=(Clause(literals=(Literal(predicate="ShouldNotRun"),)),)),
            universe=Universe(constants=()),
        )

    fn, counter = _build_translate_with_fake_translator(
        monkeypatch, tmp_path, fake_translate_factory=factory
    )

    example = _toy_example("rt_cached")
    result = fn(example)
    assert counter["calls"] == 0
    assert result.clauses[0].literals[0].predicate == "Cached"


def test_translate_callback_force_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qverify.translator.types import TranslationResult
    from qverify.verifier._universe import Universe

    monkeypatch.setattr(_RB, "_TRANSLATION_CACHE_ROOT", tmp_path)
    cache_path = _RB._translation_cache_path("ruletaker", 1, "dev")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"rt_cached": _RB._cnf_to_dict(_toy_cnf("Cached"))}),
        encoding="utf-8",
    )

    def factory(_statement: str) -> TranslationResult:
        return TranslationResult(
            cnf=CNF(clauses=(Clause(literals=(Literal(predicate="Fresh"),)),)),
            universe=Universe(constants=()),
        )

    fn, counter = _build_translate_with_fake_translator(
        monkeypatch, tmp_path, fake_translate_factory=factory, force=True
    )

    example = _toy_example("rt_cached")
    result = fn(example)
    # Translator was called for both premise and hypothesis (>=2 calls)
    assert counter["calls"] >= 2
    assert result.clauses[0].literals[0].predicate == "Fresh"


def test_translate_callback_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(_statement: str) -> Any:
        raise RuntimeError("model down")

    fn, _counter = _build_translate_with_fake_translator(
        monkeypatch, tmp_path, fake_translate_factory=factory
    )

    with pytest.raises(RuntimeError, match="model down"):
        fn(_toy_example())
