"""
Model export utilities: TorchScript and ONNX.

Production models often need to run outside Python — in C++ services,
mobile apps, or ONNX-compatible runtimes. This module handles both paths.
"""

import torch
import torch.nn as nn
from pathlib import Path


def export_torchscript(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str = "model_scripted.pt",
    method: str = "script",
) -> str:
    """
    Export model to TorchScript format.

    Args:
        model: The model to export (must be in eval mode).
        sample_input: Example input tensor for tracing.
        output_path: Where to save the exported model.
        method: 'script' for torch.jit.script, 'trace' for torch.jit.trace.

    Returns:
        Path to the exported model file.
    """
    model.eval()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if method == "script":
        scripted = torch.jit.script(model)
    elif method == "trace":
        scripted = torch.jit.trace(model, sample_input)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'script' or 'trace'.")

    scripted.save(str(path))
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"TorchScript model saved to {path} ({file_size_mb:.1f} MB)")
    return str(path)


def export_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str = "model.onnx",
    opset_version: int = 17,
) -> str:
    """
    Export model to ONNX format.

    Args:
        model: The model to export (must be in eval mode).
        sample_input: Example input tensor.
        output_path: Where to save the ONNX model.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX model file.
    """
    model.eval()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        sample_input,
        str(path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        dynamo=False,
    )

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"ONNX model saved to {path} ({file_size_mb:.1f} MB)")
    return str(path)


def verify_torchscript(
    original_model: nn.Module,
    scripted_path: str,
    sample_input: torch.Tensor,
    atol: float = 1e-5,
) -> bool:
    """Verify TorchScript model matches original model output."""
    original_model.eval()
    loaded = torch.jit.load(scripted_path)
    loaded.eval()

    def _get_logits(output):
        return output["logits"] if isinstance(output, dict) else output

    with torch.no_grad():
        orig_out = _get_logits(original_model(sample_input))
        scripted_out = _get_logits(loaded(sample_input))

    match = torch.allclose(orig_out, scripted_out, atol=atol)
    max_diff = (orig_out - scripted_out).abs().max().item()
    print(f"TorchScript verification: {'PASS' if match else 'FAIL'} (max diff: {max_diff:.2e})")
    return match


def verify_onnx(
    original_model: nn.Module,
    onnx_path: str,
    sample_input: torch.Tensor,
    atol: float = 1e-4,
) -> bool:
    """Verify ONNX model matches original model output."""
    import onnxruntime as ort
    import numpy as np

    original_model.eval()
    with torch.no_grad():
        output = original_model(sample_input)
        orig_out = (output["logits"] if isinstance(output, dict) else output).numpy()

    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {"input_ids": sample_input.numpy()})[0]

    max_diff = np.abs(orig_out - onnx_out).max()
    match = max_diff < atol
    print(f"ONNX verification: {'PASS' if match else 'FAIL'} (max diff: {max_diff:.2e})")
    return match
