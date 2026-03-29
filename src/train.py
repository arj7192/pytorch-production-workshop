"""
Production training script for the transformer language model.

Usage:
    python -m src.train                          # defaults
    python -m src.train --config configs/fast_debug.yaml
    python -m src.train --epochs 10 --use-amp    # CLI overrides
"""

import argparse
import math
import time
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.model import build_model
from src.data import prepare_wikitext2, create_dataloaders
from src.evaluate import evaluate, generate_sample
from src.utils import (
    TrainConfig,
    set_seed,
    get_device,
    setup_logging,
    CheckpointManager,
    MetricsTracker,
    NaNDetector,
)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay with linear warmup -  the standard production schedule."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
    global_step: int,
    logger,
    metrics: MetricsTracker,
    val_loader=None,
    tokenizer=None,
) -> tuple[float, int]:
    """Single training epoch with logging, NaN detection, and gradient clipping."""
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    nan_detector = NaNDetector()

    for batch_idx, (input_ids, targets) in enumerate(train_loader):
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=config.use_amp,
            dtype=torch.float16 if config.use_amp else torch.float32,
        ):
            output = model(input_ids, targets=targets)
            loss = output["loss"]

        if nan_detector.check_loss(loss, global_step):
            logger.warning(f"NaN/Inf loss at step {global_step} -  skipping batch")
            optimizer.zero_grad()
            global_step += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1
        global_step += 1

        if global_step % config.log_interval == 0:
            avg_loss = epoch_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            ppl = math.exp(min(avg_loss, 20))
            logger.info(
                f"epoch {epoch+1} | step {global_step:>6d} | "
                f"loss {loss.item():.4f} | ppl {ppl:.1f} | lr {lr:.2e}"
            )
            metrics.log(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": loss.item(),
                    "train_ppl": ppl,
                    "lr": lr,
                }
            )

        if val_loader and global_step % config.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device, use_amp=config.use_amp)
            logger.info(
                f"  [eval] val_loss {val_metrics['val_loss']:.4f} | "
                f"val_ppl {val_metrics['val_perplexity']:.1f}"
            )
            metrics.log({"step": global_step, **val_metrics})
            model.train()

    avg_epoch_loss = epoch_loss / max(n_batches, 1)
    return avg_epoch_loss, global_step


def main():
    parser = argparse.ArgumentParser(description="Train transformer language model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--use-amp", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = TrainConfig()
    if args.config:
        config = TrainConfig.from_yaml(args.config)

    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    field_map = {"batch_size": "batch_size", "learning_rate": "learning_rate", "use_amp": "use_amp"}
    for k, v in cli_overrides.items():
        attr = k.replace("-", "_")
        if hasattr(config, attr):
            setattr(config, attr, v)

    set_seed(config.seed)
    device = get_device() if config.device == "auto" else torch.device(config.device)
    logger = setup_logging(config.log_dir)
    metrics = MetricsTracker(config.log_dir)
    ckpt_manager = CheckpointManager(config.checkpoint_dir, config.keep_last_n)

    logger.info("=" * 60)
    logger.info("PRODUCTION TRAINING -  Transformer Language Model")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config.to_dict()}")

    logger.info("Loading data...")
    train_dataset, val_dataset, _, tokenizer = prepare_wikitext2(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    logger.info(
        f"Data ready: {len(train_dataset)} train samples, {len(val_dataset)} val samples"
    )

    actual_vocab_size = tokenizer.get_vocab_size()
    config.vocab_size = actual_vocab_size

    model = build_model(config.to_dict())
    model = model.to(device)
    logger.info(f"Model: {model.count_parameters():,} trainable parameters")

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    scaler = None
    if config.use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        logger.info("AMP enabled with GradScaler")

    logger.info(f"Starting training for {config.epochs} epochs ({total_steps} steps)")
    start_time = time.time()
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        epoch_start = time.time()

        avg_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
            global_step=global_step,
            logger=logger,
            metrics=metrics,
            val_loader=val_loader,
            tokenizer=tokenizer,
        )

        val_metrics = evaluate(model, val_loader, device, use_amp=config.use_amp)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{config.epochs} complete in {epoch_time:.1f}s | "
            f"train_loss {avg_loss:.4f} | val_loss {val_metrics['val_loss']:.4f} | "
            f"val_ppl {val_metrics['val_perplexity']:.1f}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            logger.info(f"  New best val_loss: {best_val_loss:.4f}")

        if (epoch + 1) % config.save_every_n_epochs == 0:
            path = ckpt_manager.save(
                model, optimizer, epoch + 1, global_step,
                val_metrics["val_loss"], config.to_dict(),
            )
            logger.info(f"  Checkpoint saved: {path}")

        sample = generate_sample(model, tokenizer, "The", device, max_new_tokens=30)
        logger.info(f"  Sample: {sample[:120]}...")

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Training complete in {total_time:.1f}s")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")
    logger.info("=" * 60)

    metrics.save()
    logger.info(f"Metrics saved to {config.log_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
