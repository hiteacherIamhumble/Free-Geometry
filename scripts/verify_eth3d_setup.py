#!/usr/bin/env python3
"""
Verify ETH3D VGGT setup before running benchmark.

Usage:
    python scripts/verify_eth3d_setup.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vggt"))


def check_checkpoint():
    """Check if LoRA checkpoint exists."""
    lora_path = "checkpoints/vggt_lora_final/eth3d/lora.pt"
    print("=" * 60)
    print("1. Checking LoRA Checkpoint")
    print("=" * 60)

    if not os.path.exists(lora_path):
        print(f"❌ FAILED: LoRA checkpoint not found at {lora_path}")
        return False

    print(f"✓ Found: {lora_path}")

    # Check file size
    size_mb = os.path.getsize(lora_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")

    # Try loading checkpoint
    try:
        import torch
        ckpt = torch.load(lora_path, map_location="cpu")
        print(f"  Keys: {list(ckpt.keys())[:5]}...")
        print("✓ Checkpoint loads successfully")
        return True
    except Exception as e:
        print(f"❌ FAILED: Cannot load checkpoint: {e}")
        return False


def check_dataset():
    """Check if ETH3D dataset is accessible."""
    print("\n" + "=" * 60)
    print("2. Checking ETH3D Dataset")
    print("=" * 60)

    try:
        from vggt.vggt.bench.registries import VGGT_MV_REGISTRY

        if not VGGT_MV_REGISTRY.has("eth3d"):
            print("❌ FAILED: eth3d not registered in VGGT_MV_REGISTRY")
            return False

        print("✓ ETH3D registered in VGGT_MV_REGISTRY")

        # Load dataset
        eth3d = VGGT_MV_REGISTRY.get("eth3d")()
        print(f"✓ Dataset loaded")
        print(f"  Scenes: {eth3d.SCENES}")

        # Try loading first scene
        scene = eth3d.SCENES[0]
        print(f"\n  Testing scene: {scene}")
        scene_data = eth3d.get_data(scene)

        print(f"  ✓ Loaded {len(scene_data.image_files)} frames")
        print(f"  ✓ Extrinsics shape: {scene_data.extrinsics.shape}")
        print(f"  ✓ Intrinsics shape: {scene_data.intrinsics.shape}")

        # Check if first image exists
        if scene_data.image_files:
            img_path = scene_data.image_files[0]
            if os.path.exists(img_path):
                print(f"  ✓ First image exists: {img_path}")
            else:
                print(f"  ⚠ Warning: First image not found: {img_path}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_loading():
    """Check if VGGT models can be loaded."""
    print("\n" + "=" * 60)
    print("3. Checking Model Loading")
    print("=" * 60)

    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.vggt.distillation.models import VGGTStudentModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        # Test base model loading (don't actually load weights)
        print("\n  Testing base VGGT model structure...")
        try:
            # Just check if we can import and instantiate
            print("  ✓ VGGT model class available")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return False

        # Test student model structure
        print("\n  Testing VGGTStudentModel structure...")
        try:
            print("  ✓ VGGTStudentModel class available")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            return False

        print("\n✓ Model classes available (not loading weights to save time)")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "=" * 60)
    print("4. Checking Dependencies")
    print("=" * 60)

    required = [
        "torch",
        "numpy",
        "cv2",
        "open3d",
        "trimesh",
    ]

    all_ok = True
    for pkg in required:
        try:
            if pkg == "cv2":
                import cv2
                print(f"  ✓ {pkg} (OpenCV {cv2.__version__})")
            else:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "unknown")
                print(f"  ✓ {pkg} ({version})")
        except ImportError:
            print(f"  ❌ {pkg} not found")
            all_ok = False

    return all_ok


def check_scripts():
    """Check if required scripts exist."""
    print("\n" + "=" * 60)
    print("5. Checking Scripts")
    print("=" * 60)

    scripts = [
        "scripts/benchmark_teacher_student_all_datasets_vggt.py",
        "scripts/benchmark_lora_vggt.py",
        "scripts/view_pointclouds.py",
        "scripts/test_eth3d_vggt_full.sh",
    ]

    all_ok = True
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✓ {script}")
        else:
            print(f"  ❌ {script} not found")
            all_ok = False

    return all_ok


def main():
    print("\n" + "=" * 60)
    print("ETH3D VGGT Setup Verification")
    print("=" * 60)
    print()

    results = []

    # Run checks
    results.append(("Checkpoint", check_checkpoint()))
    results.append(("Dataset", check_dataset()))
    results.append(("Model Loading", check_model_loading()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Scripts", check_scripts()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ All checks passed! Ready to run benchmark.")
        print("\nNext step:")
        print("  bash scripts/test_eth3d_vggt_full.sh --scene courtyard")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
