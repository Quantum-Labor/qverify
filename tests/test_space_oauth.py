"""OAuth owner-gate tests for the Space.

Pure logic: no HF/IBM credentials, no network, no GPU. app.py is loaded by file
path (space/ is not a package). Covers the OAuth helpers and (after the UI
commit) the defense-in-depth check in verify_on_ibm.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

_APP = Path(__file__).resolve().parent.parent / "space" / "app.py"


def _load_app() -> Any:
    spec = importlib.util.spec_from_file_location("_qv_app_oauth", _APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_qv_app_oauth"] = module
    spec.loader.exec_module(module)
    return module


app = _load_app()


class _Profile:
    """Stand-in for gr.OAuthProfile (only .username is read)."""

    def __init__(self, username: Any) -> None:
        self.username = username


# -- helper: get_authenticated_user / is_owner ------------------------------


def test_no_oauth_is_not_owner() -> None:
    assert app.get_authenticated_user(None) is None
    assert app.is_owner(None) is False


def test_owner_username_is_owner() -> None:
    assert app.get_authenticated_user(_Profile("Laborator")) == "Laborator"
    assert app.is_owner(_Profile("Laborator")) is True


def test_other_user_is_not_owner() -> None:
    assert app.get_authenticated_user(_Profile("someone-else")) == "someone-else"
    assert app.is_owner(_Profile("someone-else")) is False


def test_malformed_profile_is_not_owner() -> None:
    # No .username attribute at all, plus wrong-type / empty usernames.
    assert app.is_owner(object()) is False
    assert app.is_owner(_Profile(None)) is False
    assert app.is_owner(_Profile(123)) is False
    assert app.is_owner(_Profile("")) is False


def test_owner_constant_value() -> None:
    assert app.OWNER_USERNAME == "Laborator"


# -- defense in depth: verify_on_ibm refuses non-owners ---------------------


def test_verify_on_ibm_blocks_non_owner() -> None:
    import gradio as gr

    with pytest.raises(gr.Error):
        app.verify_on_ibm(app.DEFAULT_LABEL, "consistency", profile=None)
    with pytest.raises(gr.Error):
        app.verify_on_ibm(app.DEFAULT_LABEL, "consistency", profile=_Profile("nope"))
