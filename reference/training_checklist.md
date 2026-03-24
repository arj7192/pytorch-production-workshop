# Production Training Checklist

Use this before every training run. Skip items at your own risk.

---

## Before Training

- [ ] **Reproducibility**: Set all seeds (`random`, `numpy`, `torch`, `cuda`)
- [ ] **Config file**: All hyperparameters in a YAML/JSON config (not hardcoded)
- [ ] **Data splits**: Train/val/test are properly separated (no leakage)
- [ ] **Data validation**: Spot-check samples — are they sane?
- [ ] **Baseline**: Know what "random" loss should be (e.g., `ln(vocab_size)` for LM)
- [ ] **Sanity check**: Can you overfit a single batch? If not, fix bugs first

## Training Loop

- [ ] **Learning rate schedule**: Warmup + decay (cosine or linear)
- [ ] **Gradient clipping**: `clip_grad_norm_` with `max_norm=1.0`
- [ ] **Mixed precision**: AMP enabled for GPU training (`torch.autocast` + `GradScaler`)
- [ ] **`zero_grad(set_to_none=True)`**: Faster than `zero_grad()` alone
- [ ] **NaN detection**: Check `torch.isnan(loss)` every step, skip or stop
- [ ] **Periodic evaluation**: Eval every N steps (not just per epoch)
- [ ] **Logging**: Structured logs with loss, perplexity, learning rate, gradient norm

## Data Loading

- [ ] **`num_workers > 0`**: Use 2-4 workers for GPU training
- [ ] **`pin_memory=True`**: When using CUDA
- [ ] **`persistent_workers=True`**: Avoid worker respawn between epochs
- [ ] **`drop_last=True`** (training): Avoid small last batch issues
- [ ] **`shuffle=True`** (training): Always shuffle training data

## Checkpointing

- [ ] **Save model + optimizer state**: Both are needed to resume training
- [ ] **Save config**: Store hyperparameters with the checkpoint
- [ ] **Auto-cleanup**: Keep only last N checkpoints
- [ ] **Best model**: Track and save best validation checkpoint separately
- [ ] **Atomic saves**: Write to temp file, then rename (prevents corruption)

## After Training

- [ ] **Final evaluation**: Run on held-out test set
- [ ] **Model size**: Check parameter count and disk size
- [ ] **Qualitative check**: Generate/predict samples — do they make sense?
- [ ] **Export metrics**: Save training curves for comparison
- [ ] **Reproducibility test**: Same config + seed → same results?

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| No gradient clipping | Loss spikes, then NaN | Add `clip_grad_norm_(1.0)` |
| LR too high | Loss oscillates wildly | Reduce 10x, add warmup |
| No warmup | Loss explodes at start | Add 100-500 warmup steps |
| No eval during training | Overfitting goes unnoticed | Eval every 200-500 steps |
| `shuffle=False` on train | Model memorizes order | Always shuffle |
| `model.eval()` missing | Dropout active during eval | Call `model.eval()` |
| `torch.no_grad()` missing | Memory leak during eval | Wrap eval in `no_grad()` |
