"""
Pre-workshop environment checker.

Run this before the workshop to verify everything is set up:
    python setup_check.py
"""

import sys
import importlib


def check(name: str, import_name: str | None = None, min_version: str | None = None) -> bool:
    try:
        mod = importlib.import_module(import_name or name)
        version = getattr(mod, "__version__", "unknown")
        status = "OK"
        if min_version and version != "unknown":
            from packaging.version import Version
            if Version(version) < Version(min_version):
                status = f"WARN (need >={min_version})"
        print(f"  [{status}] {name} {version}")
        return True
    except ImportError:
        print(f"  [MISSING] {name} — install with: pip install {name}")
        return False


def main():
    print("=" * 50)
    print("Workshop Environment Check")
    print("=" * 50)

    print(f"\nPython: {sys.version}")
    if sys.version_info < (3, 10):
        print("  [WARN] Python 3.10+ recommended")

    print("\n--- Core Libraries ---")
    all_ok = True
    all_ok &= check("torch", min_version="2.2.0")
    all_ok &= check("torchvision", min_version="0.17.0")

    print("\n--- Data ---")
    all_ok &= check("datasets")
    all_ok &= check("tokenizers")

    print("\n--- Config ---")
    all_ok &= check("yaml", import_name="yaml")

    print("\n--- Serving ---")
    all_ok &= check("fastapi")
    all_ok &= check("uvicorn")
    all_ok &= check("pydantic")

    print("\n--- Export (optional) ---")
    check("onnx")
    check("onnxruntime", import_name="onnxruntime")

    print("\n--- Device ---")
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] CUDA available: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  [OK] Apple MPS available")
    else:
        print("  [OK] CPU mode (workshop works fine without GPU)")

    print("\n--- Docker (optional, for Module 4) ---")
    import shutil
    if shutil.which("docker"):
        print("  [OK] Docker found")
    else:
        print("  [INFO] Docker not found — needed for Module 4 (containerization)")

    print("\n--- gcloud CLI (optional, for Cloud Run deployment) ---")
    if shutil.which("gcloud"):
        print("  [OK] gcloud CLI found")
    else:
        print("  [INFO] gcloud CLI not found — needed for Cloud Run deployment")

    print("\n" + "=" * 50)
    if all_ok:
        print("All core dependencies OK. You're ready for the workshop!")
    else:
        print("Some dependencies missing. Install them with:")
        print("  pip install -r requirements.txt")
    print("=" * 50)


if __name__ == "__main__":
    main()
