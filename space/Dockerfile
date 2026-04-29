FROM python:3.11-slim

WORKDIR /app

# Install verifier-only Python deps. requirements.txt deliberately omits
# torch / transformers / outlines because the Space does not load Gemma.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install qverify itself without its declared dependencies (those include
# torch / transformers, which the verifier-only Space does not need).
RUN pip install --no-cache-dir --no-deps \
    "qverify @ git+https://github.com/Quantum-Labor/qverify.git@main"

COPY app.py .

# HuggingFace Spaces expect the app to listen on 0.0.0.0:7860.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
