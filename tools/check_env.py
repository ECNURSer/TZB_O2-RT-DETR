#!/usr/bin/env python3
"""Check the isolated O2-RT-DETR OpenMMLab environment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mmcv
import mmdet
import mmengine
import torch

import ai4rs


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"mmcv={mmcv.__version__}")
    print(f"mmengine={mmengine.__version__}")
    print(f"mmdet={mmdet.__version__}")
    print(f"cv2={cv2.__version__}")
    print(f"ai4rs={ai4rs.__file__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"gpu_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 PyTorch CUDA runtime 与 NVIDIA 驱动")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"gpu{index}={props.name}, {props.total_memory / 1024**3:.1f} GiB")


if __name__ == "__main__":
    main()
