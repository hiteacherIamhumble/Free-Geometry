#!/usr/bin/env python3
"""
Benchmark: Feed 4 frames doubled as 8 (i.e. [0,1,2,3,0,1,2,3]) to DA3.
Compare with normal 4v and 8v to see if the gap is about unique views or input count.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator


class DoubledViewDA3:
    """Wrapper that duplicates 4 input images to make 8 before calling inference."""

    def __init__(self, api):
        self.api = api

    def to(self, device):
        self.api = self.api.to(device)
        return self

    def inference(self, image, extrinsics=None, intrinsics=None, **kwargs):
        # Duplicate: [0,1,2,3] -> [0,1,2,3,0,1,2,3]
        if isinstance(image, (list, tuple)):
            doubled_image = list(image) + list(image)
        else:
            doubled_image = image

        if extrinsics is not None:
            import numpy as np
            doubled_ext = np.concatenate([extrinsics, extrinsics], axis=0)
        else:
            doubled_ext = None

        if intrinsics is not None:
            import numpy as np
            doubled_int = np.concatenate([intrinsics, intrinsics], axis=0)
        else:
            doubled_int = None

        result = self.api.inference(
            doubled_image,
            extrinsics=doubled_ext,
            intrinsics=doubled_int,
            **kwargs,
        )

        # Only return the first 4 predictions (the original frames)
        # The result object structure depends on DA3 — let's check what it returns
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='depth-anything/DA3-GIANT-1.1')
    parser.add_argument('--datasets', nargs='+', default=['scannetpp'])
    parser.add_argument('--modes', nargs='+', default=['pose', 'recon_unposed'])
    parser.add_argument('--max_frames', type=int, default=4)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--work_dir', type=str, default='./workspace/da3_4v_doubled')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    api = DepthAnything3.from_pretrained(args.model).to(device)
    doubled_api = DoubledViewDA3(api)

    evaluator = Evaluator(
        work_dir=args.work_dir,
        datas=args.datasets,
        modes=args.modes,
        max_frames=args.max_frames,
        seed=args.seed,
    )

    evaluator.infer(doubled_api)
    metrics = evaluator.eval()
    evaluator.print_metrics(metrics)


if __name__ == '__main__':
    main()
