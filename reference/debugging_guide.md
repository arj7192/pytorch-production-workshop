# Production ML Debugging Guide

When training goes wrong, use this guide to systematically diagnose and fix the problem.

---

## NaN Loss

**Symptoms**: Loss becomes `nan`, model outputs garbage

**Diagnosis checklist**:
1. Is the learning rate too high?
2. Is there a division by zero in the loss function?
3. Are there NaN values in the input data?
4. Is the model's weight initialization appropriate?

**Fixes (try in order)**:
1. **Reduce learning rate** by 10x
2. **Add gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. **Add learning rate warmup**: Start from 0, ramp up over 100-500 steps
4. **Check data**: `assert not torch.isnan(input).any()`
5. **Use pre-norm** (LayerNorm before attention, not after) — more numerically stable
6. **Switch to AMP with GradScaler**: Handles float16 underflow automatically

**Detection code**:
```python
if torch.isnan(loss) or torch.isinf(loss):
    logger.warning(f"Bad loss at step {step}: {loss.item()}")
    optimizer.zero_grad()  # Skip this batch
    continue
```

---

## Exploding Gradients

**Symptoms**: Loss spikes suddenly, gradient norms > 100, eventual NaN

**Diagnosis**:
```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
print(f"Gradient norm: {total_norm:.2f}")
# Healthy: 0.1-10. Warning: > 50. Critical: > 1000
```

**Fixes**:
1. **Gradient clipping** (always use this): `clip_grad_norm_(model.parameters(), 1.0)`
2. **Lower learning rate**
3. **Use AdamW** instead of SGD (built-in per-parameter scaling)
4. **Check for long sequences** — attention scales with sequence length squared

---

## Vanishing Gradients

**Symptoms**: Loss doesn't decrease, early layers don't update, model underfits

**Diagnosis**:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
# First layers should have comparable gradient norms to last layers
```

**Fixes**:
1. **Use residual connections** (already in transformers)
2. **Use pre-norm** instead of post-norm
3. **Increase learning rate** (carefully)
4. **Better initialization**: Xavier uniform or Kaiming

---

## Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Immediate fixes**:
1. **Reduce batch size** (halve it)
2. **Reduce sequence length**
3. **Enable gradient checkpointing**: `model.gradient_checkpointing_enable()`
4. **Use AMP** (halves activation memory)

**Diagnosis**:
```python
# Check current GPU memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max:       {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Memory summary
print(torch.cuda.memory_summary())
```

**Memory estimation formula** (rough):
```
Peak memory ≈ model_params × 4 (FP32)
             + gradients × 4
             + optimizer_states × 8 (Adam has 2 states)
             + activations × batch_size × seq_len × hidden_dim × n_layers × 4
```

**Prevention**:
```python
# Estimate before running
def estimate_memory_gb(model, batch_size, seq_len):
    param_mem = sum(p.numel() * 4 for p in model.parameters()) / 1e9
    activation_mem = batch_size * seq_len * model.d_model * model.transformer.num_layers * 16 / 1e9
    total = param_mem * 4 + activation_mem  # params + grads + optimizer + activations
    return total
```

---

## Training Loss Plateaus

**Symptoms**: Loss stops decreasing but hasn't converged

**Causes and fixes**:
| Cause | Diagnostic | Fix |
|-------|-----------|-----|
| Learning rate too low | Loss decreases very slowly | Increase 2-5x |
| LR not decaying | LR still at initial value late in training | Add cosine/step decay |
| Model too small | Val loss matches train loss but both high | Increase model size |
| Data quality | Random samples look wrong | Clean/filter data |
| Label noise | Train loss has a floor | Improve labels |
| Wrong loss function | N/A | Verify loss matches task |

---

## Overfitting

**Symptoms**: Train loss decreases, val loss increases

**Fixes (try in order)**:
1. **More data** (always the best fix)
2. **Data augmentation**
3. **Increase dropout** (0.1 → 0.2 → 0.3)
4. **Increase weight decay** (0.01 → 0.1)
5. **Reduce model size** (fewer layers/heads)
6. **Early stopping** based on val loss

---

## Slow Training

**Diagnosis with PyTorch Profiler**:
```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Run a few training steps
    pass
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Common bottlenecks**:
| Bottleneck | Diagnostic | Fix |
|-----------|-----------|-----|
| Data loading | CPU at 100%, GPU idle | `num_workers > 0`, `pin_memory` |
| CPU↔GPU transfer | High `to()` time in profiler | `pin_memory`, `non_blocking=True` |
| Small batch size | GPU utilization < 50% | Increase batch size |
| No AMP | All ops in FP32 | Enable `torch.autocast` |
| Python overhead | High "other" CPU time | `torch.compile()` |

---

## Quick Reference: Healthy Training Indicators

```
✓ Loss decreasing smoothly
✓ Gradient norm: 0.1 - 5.0 (stable)
✓ Learning rate following expected schedule
✓ Val loss tracking train loss (not diverging)
✓ GPU utilization > 80%
✓ No NaN/Inf in loss or gradients
✓ Memory usage stable (no growth over time)
```
