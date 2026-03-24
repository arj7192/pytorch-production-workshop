"""
Benchmark DataLoader configurations to find the optimal setup.

Usage:
    python scripts/benchmark_dataloader.py
"""

import time
import torch
from src.data import prepare_wikitext2
from src.utils import set_seed, get_device


def benchmark(dataset, batch_size, num_workers, pin_memory, n_batches=200):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    device = get_device()

    # Warmup
    it = iter(loader)
    for _ in range(min(10, n_batches)):
        batch = next(it)

    # Measure
    start = time.perf_counter()
    it = iter(loader)
    for i in range(n_batches):
        x, y = next(it)
        if pin_memory:
            x.to(device, non_blocking=True)
            y.to(device, non_blocking=True)
    elapsed = time.perf_counter() - start

    return elapsed / n_batches * 1000  # ms per batch


def main():
    set_seed(42)

    print("Loading dataset...")
    train_dataset, _, _, _ = prepare_wikitext2(vocab_size=8192, seq_len=128)

    configs = [
        {"num_workers": 0, "pin_memory": False},
        {"num_workers": 1, "pin_memory": False},
        {"num_workers": 2, "pin_memory": False},
        {"num_workers": 2, "pin_memory": True},
        {"num_workers": 4, "pin_memory": False},
        {"num_workers": 4, "pin_memory": True},
    ]

    print(f"\n{'Workers':<10} {'Pin Memory':<12} {'ms/batch':>10}")
    print("-" * 35)

    for cfg in configs:
        try:
            ms = benchmark(train_dataset, batch_size=64, **cfg)
            print(f"{cfg['num_workers']:<10} {str(cfg['pin_memory']):<12} {ms:>10.2f}")
        except Exception as e:
            print(f"{cfg['num_workers']:<10} {str(cfg['pin_memory']):<12} {'ERROR':>10}")


if __name__ == "__main__":
    main()
