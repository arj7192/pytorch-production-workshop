"""
Production utilities: reproducibility, logging, checkpointing, and diagnostics.
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
import numpy as np
import yaml


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging to both console and file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"train_{timestamp}.log"

    logger = logging.getLogger("workshop")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Model
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

    # Training
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    seed: int = 42

    # Data
    num_workers: int = 0
    pin_memory: bool = False

    # AMP
    use_amp: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    keep_last_n: int = 3

    # Logging
    log_interval: int = 50
    eval_interval: int = 200
    log_dir: str = "logs"

    # Device
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        return asdict(self)


class CheckpointManager:
    """Save and load model checkpoints with automatic cleanup."""

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        val_loss: float,
        config: dict | None = None,
    ) -> str:
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}_loss{val_loss:.4f}.pt"
        torch.save(checkpoint, path)
        self._cleanup()
        return str(path)

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        path: str | None = None,
    ) -> dict:
        if path is None:
            path = self.latest()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def latest(self) -> str | None:
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        return str(checkpoints[-1]) if checkpoints else None

    def _cleanup(self):
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        while len(checkpoints) > self.keep_last_n:
            checkpoints.pop(0).unlink()


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    def log(self, metrics: dict):
        metrics["timestamp"] = time.time()
        self.history.append(metrics)

    def save(self, filename: str = "metrics.jsonl"):
        path = self.log_dir / filename
        with open(path, "w") as f:
            for entry in self.history:
                f.write(json.dumps(entry) + "\n")

    def get_history(self, key: str) -> list[float]:
        return [m[key] for m in self.history if key in m]


class NaNDetector:
    """Detect NaN/Inf in model parameters and gradients."""

    @staticmethod
    def check_loss(loss: torch.Tensor, step: int) -> bool:
        if torch.isnan(loss) or torch.isinf(loss):
            return True
        return False

    @staticmethod
    def check_gradients(model: torch.nn.Module) -> dict:
        stats = {"total_params": 0, "nan_grads": 0, "inf_grads": 0, "max_grad": 0.0}
        for name, p in model.named_parameters():
            if p.grad is not None:
                stats["total_params"] += 1
                if torch.isnan(p.grad).any():
                    stats["nan_grads"] += 1
                if torch.isinf(p.grad).any():
                    stats["inf_grads"] += 1
                grad_max = p.grad.abs().max().item()
                stats["max_grad"] = max(stats["max_grad"], grad_max)
        return stats

    @staticmethod
    def check_weights(model: torch.nn.Module) -> dict:
        stats = {"nan_weights": 0, "inf_weights": 0, "max_weight": 0.0}
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                stats["nan_weights"] += 1
            if torch.isinf(p).any():
                stats["inf_weights"] += 1
            stats["max_weight"] = max(stats["max_weight"], p.abs().max().item())
        return stats
