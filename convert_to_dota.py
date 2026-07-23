#!/usr/bin/env python3
"""Convert the existing YOLO OBB dataset into DOTA-style data for O2-RT-DETR."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PROJECT = Path("/home/dihan/TZB-subject1-YOLO26-OBBV1.0")
CLASS_NAMES = [
    "Bus",
    "Cargo-Truck",
    "Dump-Truck",
    "Excavator",
    "Small-Car",
    "Tractor",
    "Trailer",
    "Truck-Tractor",
    "Van",
    "other-vehicle",
]


def link_or_copy(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_path}")
    return width, height


def convert_split(source_split: Path, output_split: Path, copy_images: bool) -> dict[str, int]:
    image_dir = source_split / "images"
    label_dir = source_split / "labels"
    output_images = output_split / "images"
    output_ann = output_split / "annfiles"
    output_images.mkdir(parents=True, exist_ok=True)
    output_ann.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "annotations": 0, "missing_images": 0}
    for label_path in sorted(label_dir.glob("*.txt")):
        source_image = image_dir / f"{label_path.stem}.tif"
        if not source_image.exists():
            stats["missing_images"] += 1
            continue
        width, height = image_size(source_image)
        link_or_copy(source_image, output_images / source_image.name, copy_images)
        lines = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if not values:
                continue
            if len(values) != 9:
                raise ValueError(f"Invalid YOLO OBB label at {label_path}:{line_number}")
            class_id = int(values[0])
            coords = [float(value) for value in values[1:]]
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"Invalid class id at {label_path}:{line_number}: {class_id}")
            pixels = []
            for index, value in enumerate(coords):
                pixels.append(value * (width if index % 2 == 0 else height))
            coord_text = " ".join(f"{value:.3f}" for value in pixels)
            lines.append(f"{coord_text} {CLASS_NAMES[class_id]} 0")
        (output_ann / label_path.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["annotations"] += len(lines)
    return stats


def write_meta(output_root: Path) -> None:
    payload = {
        "class_names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "format": "DOTA labelTxt: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty",
    }
    (output_root / "meta.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO OBB folds to DOTA format")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--all", action="store_true", help="convert fold_0..fold_4")
    parser.add_argument("--source", type=Path, default=SOURCE_PROJECT / "dataset_yolo")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "tzb_dota")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source YOLO dataset does not exist: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    folds = range(5) if args.all else [args.fold if args.fold is not None else 0]
    for fold in folds:
        source_fold = source_root / f"fold_{fold}"
        output_fold = output_root / f"fold_{fold}"
        for split in ("train", "val"):
            stats = convert_split(source_fold / split, output_fold / split, args.copy_images)
            print(f"fold {fold} {split}: {stats}")
        if (source_root / "test").is_dir():
            stats = convert_split(source_root / "test", output_fold / "test", args.copy_images)
            print(f"fold {fold} test: {stats}")
    write_meta(output_root)


if __name__ == "__main__":
    main()
