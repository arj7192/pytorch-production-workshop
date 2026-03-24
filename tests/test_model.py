"""
Smoke tests for the transformer model and training pipeline.

Run with: pytest tests/ -v
"""

import torch
import pytest
from src.model import TransformerLM, build_model


VOCAB_SIZE = 1000
D_MODEL = 64
N_HEADS = 2
D_FF = 128
N_LAYERS = 2
MAX_SEQ_LEN = 32
BATCH_SIZE = 4


@pytest.fixture
def model():
    return TransformerLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        d_ff=D_FF, n_layers=N_LAYERS, max_seq_len=MAX_SEQ_LEN,
    )


@pytest.fixture
def sample_batch():
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
    return x, y


def test_forward_shape(model, sample_batch):
    x, y = sample_batch
    output = model(x)
    assert output["logits"].shape == (BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE)


def test_forward_with_loss(model, sample_batch):
    x, y = sample_batch
    output = model(x, targets=y)
    assert "loss" in output
    assert output["loss"].dim() == 0  # scalar
    assert not torch.isnan(output["loss"])


def test_backward(model, sample_batch):
    x, y = sample_batch
    output = model(x, targets=y)
    output["loss"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_generate(model):
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 5))
    output = model.generate(input_ids, max_new_tokens=10)
    assert output.shape == (1, 15)  # 5 input + 10 generated


def test_build_model():
    config = {
        "vocab_size": VOCAB_SIZE, "d_model": D_MODEL, "n_heads": N_HEADS,
        "d_ff": D_FF, "n_layers": N_LAYERS, "max_seq_len": MAX_SEQ_LEN,
    }
    model = build_model(config)
    assert isinstance(model, TransformerLM)
    assert model.count_parameters() > 0


def test_weight_tying(model):
    """Verify embedding and output projection share weights."""
    assert model.token_emb.weight is model.output_proj.weight


def test_causal_masking(model):
    """Verify causal masking prevents future token attention."""
    model.eval()  # Disable dropout for deterministic comparison
    x1 = torch.randint(0, VOCAB_SIZE, (1, 10))
    x2 = x1.clone()
    x2[0, 5:] = torch.randint(0, VOCAB_SIZE, (5,))

    with torch.no_grad():
        out1 = model(x1)["logits"]
        out2 = model(x2)["logits"]

    # First 5 tokens should produce identical output regardless of future tokens
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5)
