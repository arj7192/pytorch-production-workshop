"""
Evaluation utilities for language model training.

Computes perplexity, generates sample text, and runs full evaluation loops.
"""

import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    """
    Run evaluation loop and return loss + perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(input_ids, targets=targets)
            loss = output["loss"]

        batch_tokens = targets.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

    return {"val_loss": avg_loss, "val_perplexity": perplexity}


@torch.no_grad()
def generate_sample(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate text from a prompt for qualitative evaluation."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    generated_ids = output_ids[0].tolist()
    return tokenizer.decode(generated_ids)


def benchmark_inference(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    n_runs: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """
    Benchmark model inference latency.

    Returns dict with mean, p50, p95, p99 latency in milliseconds.
    """
    model.eval()
    input_ids = input_ids.to(device)
    latencies = []

    for i in range(warmup + n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            model(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

    latencies.sort()
    n = len(latencies)

    return {
        "mean_ms": sum(latencies) / n,
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "n_runs": n,
    }
