# Production Dockerfile for the PyTorch inference API
# Build from repo root: docker build -f serve/Dockerfile -t pytorch-workshop-api .
#
# Multi-stage build: install dependencies first (cached layer), then copy app code

# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build
COPY serve/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ ./src/
COPY serve/app.py .

# Copy model artifacts (baked into image for workshop simplicity)
# In production, download from GCS/S3 at startup instead
COPY tokenizer.json ./tokenizer.json
COPY checkpoints/ ./checkpoints/

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Single worker for model serving (model is not thread-safe by default)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
