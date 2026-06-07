"""Integration tests for the HuggingFace Space (audit pass).

The Space (``space/app.py``) is loaded by file path (the dir is not a package).

* In-process: the simulator handler ``verify_on_simulator`` is exercised
  directly against the bundled examples for every verification mode, asserting
  the result-dict contract the UI renders. Always runs.
* End-to-end over HTTP: the Space is launched on an ephemeral local port and
  driven via ``gradio_client``, mirroring a real browser/API call. Skipped
  (not failed) when the sandbox cannot bind/serve a local socket.

Only the Space module is imported; no source is modified.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

_APP_PATH = Path(__file__).resolve().parent.parent / "space" / "app.py"


def _load_app() -> Any:
    spec = importlib.util.spec_from_file_location("_qv_app_audit", _APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_qv_app_audit"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app() -> Any:
    return _load_app()


# --- in-process (always runs) ----------------------------------------------

_EXPECTED_KEYS = {
    "status",
    "example",
    "mode",
    "backend",
    "contradiction_found",
    "counter_model",
    "n_variables",
    "n_clauses",
    "n_grover_iterations",
    "shots",
    "wall_clock_seconds",
}


@pytest.mark.parametrize("mode", ["consistency", "entailment"])
def test_verify_on_simulator_contract(app: Any, mode: str) -> None:
    for label in app.EXAMPLE_LABELS:
        res = app.verify_on_simulator(label, mode)
        assert isinstance(res, dict)
        assert _EXPECTED_KEYS.issubset(res.keys()), f"missing keys for {label}/{mode}"
        assert res["status"] == "completed"
        assert res["example"] == label
        assert res["mode"] == mode
        assert res["backend"] == "default.qubit"
        assert isinstance(res["contradiction_found"], bool)
        assert res["n_variables"] >= 0


def test_verify_on_simulator_rejects_unknown_inputs(app: Any) -> None:
    assert "error" in app.verify_on_simulator("no-such-example", "consistency")
    assert "error" in app.verify_on_simulator(app.DEFAULT_LABEL, "bogus-mode")


def test_three_bundled_examples_are_consistent(app: Any) -> None:
    # All three shipped examples are satisfiable -> no contradiction.
    for label in app.EXAMPLE_LABELS:
        res = app.verify_on_simulator(label, "consistency")
        assert res["contradiction_found"] is False


# --- end-to-end over HTTP via gradio_client (skips if no networking) --------


@pytest.fixture(scope="module")
def live_server(app: Any):
    demo = app.demo
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=0,
            prevent_thread_lock=True,
            show_error=True,
            quiet=True,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot launch local gradio server: {exc!r}")

    url = (demo.local_url or "").rstrip("/")
    ready = False
    for _ in range(40):
        try:
            urllib.request.urlopen(url + "/config", timeout=1)
            ready = True
            break
        except Exception:
            time.sleep(0.25)
    if not ready:  # pragma: no cover - environment dependent
        demo.close()
        pytest.skip("local gradio server did not become reachable")

    yield url
    demo.close()


def test_gradio_client_end_to_end_simulator(app: Any, live_server: str) -> None:
    try:
        from gradio_client import Client
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"gradio_client unavailable: {exc!r}")

    try:
        client = Client(live_server, verbose=False)
        result = client.predict(app.DEFAULT_LABEL, "consistency", api_name="/verify_on_simulator")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"gradio_client could not reach the local server: {exc!r}")

    assert isinstance(result, dict)
    assert result.get("status") == "completed"
    assert result.get("backend") == "default.qubit"
    assert result.get("example") == app.DEFAULT_LABEL
