"""Set Space/OAuth env vars before any test imports ``space/app.py``.

``space/app.py`` builds the Gradio Blocks at import time, which runs
``attach_oauth``. Gradio's ``get_space()`` is truthy only when **both**
``SYSTEM=spaces`` and ``SPACE_ID`` are set; otherwise it takes the mocked-OAuth
path that requires an HF login (which CI runners do not have). Once truthy, the
real path reads ``OAUTH_*`` at import time and raises if any is missing. So set
all of them to harmless dummies here -- this runs before test collection, so the
import succeeds in CI, fresh clones, and offline. The real values are injected by
HF at runtime; ``setdefault`` never overrides them.
"""

import os

os.environ.setdefault("SYSTEM", "spaces")
os.environ.setdefault("SPACE_ID", "Laborator/qverify")
os.environ.setdefault("OAUTH_CLIENT_ID", "ci-dummy")
os.environ.setdefault("OAUTH_CLIENT_SECRET", "ci-dummy")
os.environ.setdefault("OAUTH_SCOPES", "openid profile")
os.environ.setdefault("OPENID_PROVIDER_URL", "https://huggingface.co")
