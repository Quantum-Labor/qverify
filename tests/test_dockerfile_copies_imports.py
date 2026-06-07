"""Guard against a missing Dockerfile COPY for Space files app.py needs.

The Space is ``sdk: docker``: only files the Dockerfile ``COPY``s reach the
running image. A missing COPY leaves a file in the git repo but absent from the
container, which silently breaks features (the hardware gallery fell back to
"Gallery data unavailable" this way). This test parses ``space/Dockerfile`` and
the Space sources and fails if a referenced local module or data dir is not
copied.

Note: app.py loads ``safety.py`` and ``gallery.py`` by file path (string
literals), not via ``import``. So detection covers both ast imports and
``"<name>.py"`` string-literal references; an imports-only scan would miss them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_SPACE = Path(__file__).resolve().parent.parent / "space"
_DOCKERFILE = _SPACE / "Dockerfile"
_APP = _SPACE / "app.py"


def _copy_srcs() -> set[str]:
    """Return every ``src`` token from ``COPY <src> [<src>...] <dst>`` lines."""
    srcs: set[str] = set()
    for line in _DOCKERFILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("COPY "):
            parts = stripped.split()
            # COPY <src> <dst>  ->  every token between COPY and the final dst
            for token in parts[1:-1]:
                srcs.add(token.rstrip("/"))
    return srcs


def _local_modules() -> set[str]:
    """Local Space Python module names (excludes app.py and test files)."""
    return {
        p.stem for p in _SPACE.glob("*.py") if p.name != "app.py" and not p.name.startswith("test_")
    }


def _referenced_modules() -> set[str]:
    """Local modules app.py depends on, via import or ``"<name>.py"`` literal."""
    src = _APP.read_text(encoding="utf-8")
    locals_ = _local_modules()
    referenced: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            referenced |= {alias.name for alias in node.names if alias.name in locals_}
        elif isinstance(node, ast.ImportFrom) and node.module in locals_:
            referenced.add(node.module)
    referenced |= {m for m in locals_ if f"{m}.py" in src}
    return referenced


def test_dockerfile_copies_every_referenced_module() -> None:
    copied = _copy_srcs()
    referenced = _referenced_modules()
    assert referenced, "sanity: app.py should reference at least one local module"
    missing = sorted(f"{m}.py" for m in referenced if f"{m}.py" not in copied)
    assert not missing, f"space/Dockerfile is missing COPY lines for: {missing}"


def test_dockerfile_copies_data_dir_when_referenced() -> None:
    sources = _APP.read_text(encoding="utf-8") + "".join(
        (_SPACE / f"{m}.py").read_text(encoding="utf-8") for m in _local_modules()
    )
    needs_data = (
        bool(re.search(r"data/[a-z_]+\.json", sources))
        or ('"data"' in sources and ".json" in sources)
        or (_SPACE / "data").is_dir()
    )
    if needs_data:
        assert "data" in _copy_srcs(), (
            "Space references a data/ file but space/Dockerfile has no `COPY data/`"
        )


def test_gallery_and_safety_modules_are_copied() -> None:
    # Explicit regression anchor for the two modules that motivated this guard.
    copied = _copy_srcs()
    assert "safety.py" in copied
    assert "gallery.py" in copied
    assert "data" in copied
