# Production Dockerfile for the PyTorch inference API
#
# Prerequisites — Notebook 04 copies these automatically:
#   serve/tokenizer.json
#   serve/model_checkpoint.pt
#
# Build from REPO ROOT (need src/ for model definition):
#   docker build -f serve/Dockerfile -t pytorch-workshop-api .
#
# Run:
#   docker run -p 8000:8000 pytorch-workshop-api

# --- Stage 1: Install Python dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build
COPY serve/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

COPY --from=builder /install /usr/local

# Application code
COPY src/ ./src/
COPY serve/app.py .

# Model artifacts — produced by the training pipeline
# app.py gracefully handles missing files (serves with random weights)
COPY serve/tokenizer.json serve/model_checkpoint.pt ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
