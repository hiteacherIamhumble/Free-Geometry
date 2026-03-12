"""
Example script demonstrating TTT usage on VGGT.

This script shows how to:
1. Load a VGGT model with LoRA adapters
2. Run TTT on a test scene
3. Save and load TTT-adapted checkpoints
4. Use the adapted model for inference
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.distillation.models import VGGTStudentModel
from vggt.distillation.dataset import VGGTDistillDataset
from vggt.training.ttt_loss import ttt_consistency_loss, make_triplet_pairs


def example_basic_ttt():
    """Basic TTT example on a single scene."""
    print("="*60)
    print("Example 1: Basic TTT on Single Scene")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model with LoRA
    print("\n1. Loading VGGT model with LoRA adapters...")
    model = VGGTStudentModel(
        model_name="facebook/vggt-1b",
        lora_rank=16,
        lora_alpha=16.0,
        train_camera_token=False,
    ).to(device)

    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 2. Create dummy images (simulating a test scene)
    print("\n2. Creating dummy test scene (8 images)...")
    num_images = 8
    images = torch.randn(num_images, 3, 224, 224).to(device)

    # 3. Generate triplet pairs
    print("\n3. Generating triplet pairs...")
    triplets = make_triplet_pairs(num_images, max_triplets=10)
    print(f"   Generated {len(triplets)} triplets")
    print(f"   Sample: {triplets[:3]}")

    # 4. Freeze everything except LoRA
    print("\n4. Freezing all parameters except LoRA...")
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable after freezing: {trainable:,}")

    # 5. Setup optimizer
    print("\n5. Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.95),
    )

    # 6. Run TTT for 1 epoch
    print("\n6. Running TTT (1 epoch, 3 steps)...")
    model.train()

    for step, (anchor_idx, j_idx, k_idx) in enumerate(triplets[:3]):
        # Prepare pairs
        pair1 = torch.stack([images[anchor_idx], images[j_idx]]).unsqueeze(0)
        pair2 = torch.stack([images[anchor_idx], images[k_idx]]).unsqueeze(0)

        # Forward pass
        _, pred1 = model.forward_with_preds(pair1)
        _, pred2 = model.forward_with_preds(pair2)

        # Extract anchor predictions
        anchor_pred1 = {k: v[:, 0:1] for k, v in pred1.items() if isinstance(v, torch.Tensor)}
        anchor_pred2 = {k: v[:, 0:1] for k, v in pred2.items() if isinstance(v, torch.Tensor)}

        # Compute loss
        loss = ttt_consistency_loss(anchor_pred1, anchor_pred2, depth_weight=0.1)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"   Step {step+1}/3: Loss = {loss.item():.4f}")

    print("\n✓ Basic TTT example completed!")
    return model


def example_save_load_checkpoint():
    """Example of saving and loading TTT checkpoints."""
    print("\n" + "="*60)
    print("Example 2: Save and Load TTT Checkpoint")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create model
    print("\n1. Creating model...")
    model = VGGTStudentModel(
        model_name="facebook/vggt-1b",
        lora_rank=16,
        lora_alpha=16.0,
    ).to(device)

    # 2. Save checkpoint
    print("\n2. Saving LoRA checkpoint...")
    save_path = "/tmp/test_ttt_checkpoint.pt"
    model.save_lora_weights(save_path)
    print(f"   Saved to {save_path}")

    # 3. Create new model and load checkpoint
    print("\n3. Loading checkpoint into new model...")
    model2 = VGGTStudentModel(
        model_name="facebook/vggt-1b",
        lora_rank=16,
        lora_alpha=16.0,
    ).to(device)

    model2.load_lora_weights(save_path)
    print("   ✓ Checkpoint loaded successfully")

    # 4. Verify weights match
    print("\n4. Verifying weights match...")
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        if 'lora' in n1.lower():
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"
    print("   ✓ All LoRA weights match!")

    print("\n✓ Save/load checkpoint example completed!")


def example_with_real_dataset():
    """Example using real ETH3D dataset."""
    print("\n" + "="*60)
    print("Example 3: TTT with Real ETH3D Dataset")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    print("\n1. Loading ETH3D dataset...")
    try:
        dataset = VGGTDistillDataset(
            dataset_name='eth3d',
            num_views=8,
            fixed_subset_seed=42,
            fixed_subset_max_frames=20,  # Use only 20 frames for quick test
            augment=False,
        )
        print(f"   Found {len(dataset.scenes)} scenes")

        # 2. Load first scene
        scene = dataset.scenes[0]
        print(f"\n2. Loading scene: {scene}")

        scene_data = dataset._load_scene_data(scene)
        image_files = scene_data['image_files']

        # Get fixed subset
        subset_indices = dataset._get_fixed_subset_indices(scene, len(image_files))
        image_files = [image_files[i] for i in subset_indices]

        print(f"   Total images: {len(image_files)}")
        print(f"   Using subset: {len(subset_indices)} images")

        # 3. Load images
        print("\n3. Loading images...")
        images = []
        for img_path in image_files[:5]:  # Load only first 5 for quick test
            img = dataset._load_and_preprocess_image(img_path)
            images.append(torch.from_numpy(img).permute(2, 0, 1))

        images = torch.stack(images).to(device)
        print(f"   Loaded {len(images)} images with shape {images.shape}")

        # 4. Create model
        print("\n4. Creating model...")
        model = VGGTStudentModel(
            model_name="facebook/vggt-1b",
            lora_rank=16,
            lora_alpha=16.0,
        ).to(device)

        # 5. Run mini TTT
        print("\n5. Running mini TTT (3 triplets)...")
        triplets = make_triplet_pairs(len(images), max_triplets=3)

        # Freeze except LoRA
        for name, param in model.named_parameters():
            param.requires_grad = 'lora' in name.lower()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-5,
        )

        model.train()
        for step, (anchor_idx, j_idx, k_idx) in enumerate(triplets):
            pair1 = torch.stack([images[anchor_idx], images[j_idx]]).unsqueeze(0)
            pair2 = torch.stack([images[anchor_idx], images[k_idx]]).unsqueeze(0)

            _, pred1 = model.forward_with_preds(pair1)
            _, pred2 = model.forward_with_preds(pair2)

            anchor_pred1 = {k: v[:, 0:1] for k, v in pred1.items() if isinstance(v, torch.Tensor)}
            anchor_pred2 = {k: v[:, 0:1] for k, v in pred2.items() if isinstance(v, torch.Tensor)}

            loss = ttt_consistency_loss(anchor_pred1, anchor_pred2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"   Step {step+1}/3: Loss = {loss.item():.4f}")

        print("\n✓ Real dataset example completed!")

    except Exception as e:
        print(f"\n⚠ Could not run real dataset example: {e}")
        print("   This is expected if ETH3D dataset is not available")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("VGGT Test-Time Training (TTT) Examples")
    print("="*60)

    # Example 1: Basic TTT
    try:
        example_basic_ttt()
    except Exception as e:
        print(f"\n✗ Example 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Example 2: Save/Load
    try:
        example_save_load_checkpoint()
    except Exception as e:
        print(f"\n✗ Example 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Example 3: Real dataset
    try:
        example_with_real_dataset()
    except Exception as e:
        print(f"\n✗ Example 3 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
