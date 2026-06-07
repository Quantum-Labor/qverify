FROM python:3.11-slim

WORKDIR /app

# git is needed for `pip install` from github URLs (the qverify install
# below uses pip's git fetcher). python:3.11-slim does not ship git, so
# install it explicitly and clean up apt lists in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install verifier-only Python deps. requirements.txt deliberately omits
# torch / transformers / outlines because the Space does not load Gemma.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install qverify itself without its declared dependencies (those include
# torch / transformers, which the verifier-only Space does not need).
RUN pip install --no-cache-dir --no-deps \
    "qverify @ git+https://github.com/Quantum-Labor/qverify.git@main"

COPY app.py .
COPY safety.py .
COPY gallery.py .
COPY assets/ ./assets/
COPY benchmarks/ ./benchmarks/
COPY data/ ./data/

# HuggingFace Spaces expect the app to listen on 0.0.0.0:7860.
# PYTHONUNBUFFERED + `python -u` so cold-import progress (pennylane,
# qiskit, qiskit-ibm-runtime — ~60–120 s on CPU Basic) is visible in
# the Space's container logs in real time, instead of appearing only
# after Gradio finishes booting.
#
# SYSTEM=spaces: HF sets this only on Gradio/Streamlit-SDK Spaces, not on
# docker Spaces. Gradio's get_space() keys real OAuth off it (oauth.py); without
# it, gr.LoginButton takes the *mocked* OAuth path and crashes at startup
# (ValueError: must be logged in to HF). Forcing it makes Gradio use the real HF
# OAuth flow with the OAUTH_* env that `hf_oauth: true` injects.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1 \
    SYSTEM=spaces
EXPOSE 7860

CMD ["python", "-u", "app.py"]
