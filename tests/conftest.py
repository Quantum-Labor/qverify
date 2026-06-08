"""Set SYSTEM=spaces before any test imports space/app.py so
Gradio's get_space() returns truthy and skips the mocked-OAuth
code path that requires an HF login. Mirrors the Dockerfile."""
import os
os.environ.setdefault("SYSTEM", "spaces")
