"""
ScanNet++ Dataset for Knowledge Distillation Training.

This module provides a PyTorch Dataset for loading ScanNet++ scenes
for knowledge distillation training. Each sample returns:
- 8 views for the teacher model
- 4 views (indices [0,2,4,6]) for the student model

The dataset:
- Reads scene lists from train.txt, val.txt, test.txt
- Uniformly samples views from each scene
- Applies ImageNet normalization
- Handles COLMAP camera parameters (optional)
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from depth_anything_3.utils.read_write_model import read_model


# ImageNet normalization (same as DA3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ScanNetPPDistillDataset(Dataset):
    """
    ScanNet++ Dataset for knowledge distillation.

    Returns 8 uniformly sampled views per scene for teacher,
    and corresponding 4 views (indices [0,2,4,6]) for student.

    Args:
        data_root: Path to data directory containing scene folders
        split: 'train', 'val', or 'test'
        num_views: Number of views for teacher (default: 8)
        image_size: Target image size (H, W), default (518, 518)
        student_indices: Which teacher views to use for student (default: [0,2,4,6])
        augment: Whether to apply data augmentation (only for train)
        load_cameras: Whether to load camera parameters from COLMAP
        cache_metadata: Whether to cache scene metadata in memory
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_views: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        student_indices: List[int] = None,
        augment: bool = True,
        load_cameras: bool = False,
        cache_metadata: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.student_indices = student_indices or [0, 2, 4, 6]
        self.augment = augment and (split == 'train')
        self.load_cameras = load_cameras
        self.cache_metadata = cache_metadata

        # Load scene list from split file
        split_file = os.path.join(data_root, f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                f"Run scripts/create_splits.py first."
            )

        with open(split_file, 'r') as f:
            self.scenes = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.scenes)} scenes for {split} split")

        # Image transforms
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Cache for scene metadata
        self._scene_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_scene_metadata(self, scene: str) -> Dict:
        """Load and cache scene metadata."""
        if self.cache_metadata and scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_path = os.path.join(self.data_root, scene)
        images_path = os.path.join(scene_path, 'images', 'iphone')
        colmap_path = os.path.join(scene_path, 'colmap', 'sparse_render_rgb')

        # Get sorted image names
        image_names = sorted([
            f for f in os.listdir(images_path)
            if f.endswith(('.jpg', '.JPG', '.png', '.PNG'))
        ])

        metadata = {
            'scene': scene,
            'images_path': images_path,
            'image_names': image_names,
            'colmap_path': colmap_path,
        }

        # Optionally load COLMAP data
        if self.load_cameras and os.path.exists(colmap_path):
            try:
                cameras, images, _ = read_model(colmap_path)
                metadata['cameras'] = cameras
                metadata['images'] = images
            except Exception as e:
                print(f"Warning: Failed to load COLMAP for {scene}: {e}")

        if self.cache_metadata:
            self._scene_cache[scene] = metadata

        return metadata

    def _sample_view_indices(self, num_available: int) -> List[int]:
        """
        Uniformly sample view indices from available images.

        Args:
            num_available: Number of available images in scene

        Returns:
            List of indices to sample
        """
        if num_available <= self.num_views:
            # Not enough images, use all and repeat if needed
            indices = list(range(num_available))
            while len(indices) < self.num_views:
                indices.append(random.randint(0, num_available - 1))
            return indices

        # Uniform sampling
        indices = np.linspace(0, num_available - 1, self.num_views, dtype=int).tolist()
        return indices

    def _load_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        img = img.astype(np.float32) / 255.0

        return img

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image."""
        # Color jitter
        if random.random() < 0.5:
            brightness = 0.8 + 0.4 * random.random()
            img = img * brightness
            img = np.clip(img, 0, 1)

        # Random horizontal flip (applied consistently to all views in a scene)
        # Note: This is handled at the batch level, not here

        return img

    def _get_camera_params(
        self,
        metadata: Dict,
        image_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get camera extrinsics and intrinsics for an image.

        Returns:
            Tuple of (extrinsics [4, 4], intrinsics [3, 3]) or (None, None)
        """
        if 'images' not in metadata or 'cameras' not in metadata:
            return None, None

        # Find image in COLMAP data
        images = metadata['images']
        cameras = metadata['cameras']

        for img_id, img_data in images.items():
            if img_data.name == image_name or img_data.name.endswith(image_name):
                # Build extrinsics (world-to-camera)
                ext = np.eye(4, dtype=np.float32)
                ext[:3, :3] = img_data.qvec2rotmat()
                ext[:3, 3] = img_data.tvec

                # Build intrinsics
                cam = cameras[img_data.camera_id]
                ixt = np.eye(3, dtype=np.float32)
                ixt[0, 0] = cam.params[0]  # fx
                ixt[1, 1] = cam.params[1]  # fy
                ixt[0, 2] = cam.params[2]  # cx
                ixt[1, 2] = cam.params[3]  # cy

                return ext, ixt

        return None, None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dict with:
                - teacher_images: [8, 3, H, W] tensor
                - student_images: [4, 3, H, W] tensor
                - scene_id: str
                - (optional) teacher_extrinsics: [8, 4, 4] tensor
                - (optional) teacher_intrinsics: [8, 3, 3] tensor
        """
        scene = self.scenes[idx]
        metadata = self._load_scene_metadata(scene)

        # Sample view indices
        view_indices = self._sample_view_indices(len(metadata['image_names']))

        # Determine if we should flip (consistent across all views)
        do_flip = self.augment and random.random() < 0.5

        # Load images
        images = []
        extrinsics = []
        intrinsics = []

        for view_idx in view_indices:
            img_name = metadata['image_names'][view_idx]
            img_path = os.path.join(metadata['images_path'], img_name)

            # Load image
            img = self._load_image(img_path)

            # Apply augmentation
            if self.augment:
                img = self._augment_image(img)

            # Apply flip
            if do_flip:
                img = np.fliplr(img).copy()

            images.append(img)

            # Load camera params if requested
            if self.load_cameras:
                ext, ixt = self._get_camera_params(metadata, img_name)
                if ext is not None:
                    extrinsics.append(ext)
                    intrinsics.append(ixt)

        # Stack images: [8, H, W, 3] -> [8, 3, H, W]
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()

        # Normalize
        images = self.normalize(images)

        # Teacher gets all 8 views
        teacher_images = images

        # Student gets views at specified indices
        student_images = images[self.student_indices]

        result = {
            'teacher_images': teacher_images,
            'student_images': student_images,
            'scene_id': scene,
            'student_frame_indices': torch.tensor(self.student_indices),
        }

        # Add camera params if available
        if self.load_cameras and len(extrinsics) == self.num_views:
            result['teacher_extrinsics'] = torch.from_numpy(
                np.stack(extrinsics, axis=0)
            ).float()
            result['teacher_intrinsics'] = torch.from_numpy(
                np.stack(intrinsics, axis=0)
            ).float()
            result['student_extrinsics'] = result['teacher_extrinsics'][self.student_indices]
            result['student_intrinsics'] = result['teacher_intrinsics'][self.student_indices]

        return result


def create_dataloader(
    data_root: str,
    split: str,
    batch_size: int = 2,
    num_workers: int = 4,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the ScanNet++ distillation dataset.

    Args:
        data_root: Path to data directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for ScanNetPPDistillDataset

    Returns:
        DataLoader instance
    """
    dataset = ScanNetPPDistillDataset(
        data_root=data_root,
        split=split,
        **dataset_kwargs,
    )

    shuffle = (split == 'train')

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
