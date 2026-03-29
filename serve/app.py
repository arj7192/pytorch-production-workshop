"""
FastAPI inference server for the transformer language model.

Production patterns demonstrated:
- Model loaded once at startup (lifespan event)
- Health endpoint for load balancer probes
- Input validation with Pydantic
- Request timing middleware
- Structured error responses
- Graceful handling of model loading failures
"""

import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# --- Global state ---
model = None
tokenizer = None
device = None
model_config = None


def _download_from_gcs(bucket_uri: str, blob_name: str, dest: str):
    """Download a file from GCS if it doesn't exist locally."""
    if Path(dest).exists():
        return
    try:
        from google.cloud import storage
        bucket_name = bucket_uri.replace("gs://", "").strip("/")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_name).download_to_filename(dest)
        logger.info(f"Downloaded gs://{bucket_name}/{blob_name} -> {dest}")
    except Exception as e:
        logger.warning(f"GCS download failed for {blob_name}: {e}")


def load_model():
    """Load model and tokenizer at startup."""
    global model, tokenizer, device, model_config

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Pull artifacts from GCS if GCS_BUCKET is set and files are missing
    gcs_bucket = os.environ.get("GCS_BUCKET")
    if gcs_bucket:
        logger.info(f"Checking GCS bucket: {gcs_bucket}")
        _download_from_gcs(gcs_bucket, "tokenizer.json", "tokenizer.json")
        _download_from_gcs(gcs_bucket, "model_checkpoint.pt", "model_checkpoint.pt")

    # Load tokenizer
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "tokenizer.json")
    if not Path(tokenizer_path).exists():
        parent = Path(__file__).resolve().parent
        tokenizer_path = str(parent / "tokenizer.json")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")

    model_config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "d_model": int(os.environ.get("D_MODEL", 256)),
        "n_heads": int(os.environ.get("N_HEADS", 4)),
        "d_ff": int(os.environ.get("D_FF", 512)),
        "n_layers": int(os.environ.get("N_LAYERS", 4)),
        "max_seq_len": int(os.environ.get("MAX_SEQ_LEN", 128)),
        "dropout": 0.0,
    }

    model = build_model(model_config).to(device)
    model.eval()

    # Load checkpoint if available
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "model_checkpoint.pt")
    if not Path(checkpoint_path).exists():
        parent = Path(__file__).resolve().parent
        checkpoint_path = str(parent / "model_checkpoint.pt")

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    else:
        logger.warning("No checkpoint found -  serving with random weights")

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model ready: {param_count:,} parameters")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Shutting down")


# --- FastAPI App ---
app = FastAPI(
    title="PyTorch Workshop -  Inference API",
    description="Text generation with a small transformer language model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response schemas ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt")
    max_tokens: int = Field(50, ge=1, le=200, description="Max tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(40, ge=1, le=200, description="Top-k sampling")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float
    prompt: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    parameters: int


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        parameters=sum(p.numel() for p in model.parameters()) if model else 0,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    try:
        encoded = tokenizer.encode(request.prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )

        generated_text = tokenizer.decode(output_ids[0].tolist())
        tokens_generated = output_ids.size(1) - input_ids.size(1)
        latency_ms = (time.perf_counter() - start) * 1000

        return GenerateResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            latency_ms=round(latency_ms, 2),
            prompt=request.prompt,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": "pytorch-workshop-api",
        "docs": "/docs",
        "health": "/health",
    }
