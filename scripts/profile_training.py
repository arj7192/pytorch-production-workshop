"""
Profile a training run and export Chrome trace.

Usage:
    python scripts/profile_training.py

Output:
    logs/profiler/trace.json — open in chrome://tracing
"""

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from src.model import build_model
from src.data import prepare_wikitext2, create_dataloaders
from src.utils import set_seed, get_device


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    print("Loading data...")
    train_dataset, val_dataset, _, tokenizer = prepare_wikitext2(
        vocab_size=8192, seq_len=128
    )
    train_loader, _ = create_dataloaders(train_dataset, val_dataset, batch_size=64)

    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "d_model": 256, "n_heads": 4, "d_ff": 512,
        "n_layers": 4, "max_seq_len": 128, "dropout": 0.1,
    }
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("Profiling...")
    model.train()

    with profile(
        activities=activities,
        schedule=schedule(wait=2, warmup=3, active=10, repeat=1),
        on_trace_ready=tensorboard_trace_handler("logs/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, (x, y) in enumerate(train_loader):
            if step >= 17:
                break
            x, y = x.to(device), y.to(device)
            output = model(x, targets=y)
            output["loss"].backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            prof.step()

    print("\nTop operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    prof.export_chrome_trace("logs/profiler/trace.json")
    print("\nChrome trace saved to logs/profiler/trace.json")
    print("Open chrome://tracing and load the file for visual analysis.")


if __name__ == "__main__":
    main()
