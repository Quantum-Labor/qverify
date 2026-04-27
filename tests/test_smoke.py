"""Smoke tests — verify the package imports and basic invariants hold."""

import qverify


def test_version_is_set() -> None:
    assert isinstance(qverify.__version__, str)
    assert qverify.__version__.count(".") == 2


def test_subpackages_importable() -> None:
    from qverify import controller, eval, translator, utils, verifier  # noqa: F401
