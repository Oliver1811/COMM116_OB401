#!/usr/bin/env python3
"""
verify_setup.py — Verify environment setup and model download

Run this after installing dependencies to confirm everything is working.
The Qwen2-VL-2B-Instruct model (~2.5 GB) will download on first run.
"""

import sys

def check_imports():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    required = [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("skimage", "scikit-image"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} — MISSING")
            missing.append(name)
    
    if missing:
        print(f"\nError: Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed\n")
    return True


def check_cuda():
    """Check CUDA availability."""
    import torch
    
    print("Checking CUDA...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠ CUDA not available — will use CPU (slower)")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  ✓ MPS (Apple Silicon) available")
    print()


def test_model_load():
    """Test loading the vision model."""
    print("Testing model load (this will download ~2.5 GB on first run)...")
    
    try:
        from model_loader import generate_response
        
        # Simple text-only test (no image required)
        print("  Loading model...")
        response = generate_response([{
            "role": "user",
            "content": "Hello! Can you confirm you are working?"
        }])
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model response: {response[:100]}...")
        
    except Exception as e:
        print(f"  ✗ Model load failed: {e}")
        return False
    
    print("\n✓ Model is ready\n")
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Environment Setup Verification")
    print("=" * 60)
    print()
    
    # Check Python version
    py_version = sys.version_info
    print(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 10):
        print("⚠ Warning: Python 3.10+ recommended")
    print()
    
    # Run checks
    if not check_imports():
        sys.exit(1)
    
    check_cuda()
    
    if not test_model_load():
        sys.exit(1)
    
    print("=" * 60)
    print("✓ Setup verified — ready to run evaluations!")
    print("=" * 60)
    print()
    print("Try:")
    print("  python run_eval.py --data dev.jsonl --out outputs/test --max-samples 3")


if __name__ == "__main__":
    main()
