# Building Production-Ready PyTorch Systems in a Day

A hands-on workshop by **Ashish Ranjan Jha**, author of [Mastering PyTorch](https://www.packtpub.com/product/mastering-pytorch-second-edition/9781801074308).

Train a small transformer model from scratch, optimize it for production throughput, and deploy it as a scalable inference API — all in 3 hours.

---

## What You'll Build

Using a **GPT-style language model** (~8M parameters) trained on WikiText-2 as the anchor project, you'll work through the engineering steps that separate a working model from a production-ready system:

| Module | What you'll do |
|--------|---------------|
| **01 — Model + Training** | Build a modern transformer, structure a reproducible training pipeline with eval, checkpointing, and logging |
| **02 — Speedups + Stability** | Add mixed precision (AMP), optimize DataLoaders, profile bottlenecks, and fix NaNs/OOMs |
| **03 — Inference + Export** | Batch inference, dynamic quantization, TorchScript & ONNX export, benchmark latency |
| **04 — Deploy** | Wrap in FastAPI, containerize with Docker, deploy to Google Cloud Run |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/PacktPublishing/pytorch-production-workshop.git
cd pytorch-production-workshop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify your setup

```bash
python setup_check.py
```

### 3. Open the first notebook

```bash
jupyter lab notebooks/
```

---

## Prerequisites

- Python 3.10+
- PyTorch 2.2+ (CPU is fine for the workshop; GPU recommended for speed)
- ~4 GB free disk space
- Docker (for Module 04)
- `gcloud` CLI (optional, for Cloud Run deployment)

No advanced PyTorch experience required — if you've trained a model before, you're ready.

---

## Repository Layout

```
├── notebooks/                     # Hands-on workshop notebooks (start here)
│   ├── 01_model_and_training.ipynb
│   ├── 02_training_speedups.ipynb
│   ├── 03_inference_and_export.ipynb
│   └── 04_deploy.ipynb
│
├── src/                           # Production-quality Python modules
│   ├── model.py                   # Transformer model definition
│   ├── data.py                    # Dataset and DataLoader utilities
│   ├── train.py                   # Configurable training script (CLI)
│   ├── evaluate.py                # Evaluation and metrics
│   ├── export.py                  # TorchScript / ONNX export
│   └── utils.py                   # Reproducibility, logging, checkpointing
│
├── serve/                         # Inference microservice
│   ├── app.py                     # FastAPI server
│   ├── Dockerfile                 # Production container
│   └── deploy.sh                  # GCP Cloud Run deployment
│
├── configs/                       # Training configurations
│   ├── default.yaml               # Full training config
│   └── fast_debug.yaml            # Quick smoke-test config
│
├── scripts/                       # Standalone utilities
│   ├── benchmark_dataloader.py    # DataLoader perf comparison
│   └── profile_training.py        # Training profiler with Chrome trace
│
├── tests/                         # Smoke tests
│   └── test_model.py
│
├── reference/                     # Takeaway reference cards
│   ├── training_checklist.md      # Production training checklist
│   ├── debugging_guide.md         # NaN/OOM/instability fixes
│   └── deployment_checklist.md    # Deployment readiness checklist
│
├── setup_check.py                 # Pre-workshop environment validator
└── requirements.txt
```

---

## Workshop Agenda (3 hours)

### Welcome + Setup (10 min)
Quick orientation, repo walkthrough, and what "production-ready" means for this workshop.

### Module 1: Build the Anchor Transformer (45 min)
- Define a small GPT-style transformer with modern architecture patterns
- Structure a clean training loop: DataLoader → forward → loss → backward → step
- Add evaluation, checkpointing, and reproducibility from the start
- Understand *why* each component matters for production reliability

### Module 2: Training Speedups + Stability (45 min)
- **Speed**: Mixed precision (AMP + GradScaler), DataLoader tuning (`num_workers`, `pin_memory`, `persistent_workers`), `torch.compile`
- **Stability**: Gradient clipping, NaN/Inf detection, learning rate warmup, OOM prevention
- **Profiling**: Use PyTorch Profiler to find actual bottlenecks (not guessed ones)

### Module 3: Inference Optimization + Export (40 min)
- Batched inference for throughput
- Dynamic quantization to cut model size and latency
- Export via TorchScript (`torch.jit.script`) and ONNX (`torch.onnx.export`)
- Benchmark: eager vs compiled vs quantized vs ONNX

### Module 4: Deploy to Production (30 min)
- Wrap the model in a FastAPI inference service
- Containerize with Docker (multi-stage build, non-root user)
- Deploy to Google Cloud Run with `gcloud`
- Health checks, graceful startup, and what to monitor

### Wrap-up + Q&A (10 min)

---

## What's Included with Registration

- Free copy of [Mastering PyTorch](https://www.packtpub.com/product/mastering-pytorch-second-edition/9781801074308) ebook
- Workshop recording for replay
- This complete code repository
- Reference checklists for training, debugging, and deployment
- Certificate of completion

---

## About the Instructor

**Ashish Ranjan Jha** is a machine learning engineer and author of *Mastering PyTorch* (Packt, 2nd Edition). He has built ML systems at Oracle, Sony, Revolut, and Tractable — from sensor-based transport prediction to insurance fraud detection. He focuses on bridging the gap between ML experimentation and production engineering.

---

## License

MIT — use this code however you like.
