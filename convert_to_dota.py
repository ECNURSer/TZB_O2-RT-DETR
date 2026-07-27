#!/usr/bin/env python3
"""Convert project OBB annotations into DOTA-style data for O2-RT-DETR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PROJECT = Path("/home/dihan/TZB-subject1-YOLO26-OBBV1.0")
RAW_TZB_DATASET = Path("/data/work1/00_data/TZB/subject1/dataset")
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
LABEL_ALIASES = {
    "Bus": "Bus",
    "Cargo Truck": "Cargo-Truck",
    "Dump Truck": "Dump-Truck",
    "Excavator": "Excavator",
    "Small Car": "Small-Car",
    "Tractor": "Tractor",
    "Trailer": "Trailer",
    "Truck Tractor": "Truck-Tractor",
    "Van": "Van",
    "other-vehicle": "other-vehicle",
}


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


def safe_image_stem(image_path: Path) -> str:
    normalized = image_path.expanduser().as_posix()
    stem = re.sub(r"[^0-9A-Za-z]+", "_", normalized).strip("_")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}" if stem else digest


def normalize_label(label: str) -> str:
    normalized = LABEL_ALIASES.get(label)
    if normalized is None:
        raise ValueError(f"Unsupported label: {label}")
    return normalized


def annotation_line(item: dict) -> str:
    points = item.get("points", [])
    if len(points) < 4:
        raise ValueError(f"Invalid polygon, expected at least 4 points: {points}")
    coords = []
    for point in points[:4]:
        if len(point) < 2:
            raise ValueError(f"Invalid point: {point}")
        coords.extend([float(point[0]), float(point[1])])
    coord_text = " ".join(f"{value:.3f}" for value in coords)
    return f"{coord_text} {normalize_label(item['lab'])} 0"


def convert_json_split(annotation_files: list[Path], output_split: Path, copy_images: bool) -> dict[str, int]:
    if output_split.exists():
        shutil.rmtree(output_split)
    output_images = output_split / "images"
    output_ann = output_split / "annfiles"
    output_images.mkdir(parents=True, exist_ok=True)
    output_ann.mkdir(parents=True, exist_ok=True)

    grouped: OrderedDict[Path, list[str]] = OrderedDict()
    stats = {
        "images": 0,
        "annotations": 0,
        "missing_images": 0,
        "unreadable_images": 0,
        "skipped_missing_image_annotations": 0,
        "skipped_unreadable_image_annotations": 0,
    }
    missing_images: set[Path] = set()

    for annotation_file in annotation_files:
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        for item in payload.get("data", []):
            source_image = Path(item["data_path"]).expanduser()
            if not source_image.is_file():
                if source_image not in missing_images:
                    missing_images.add(source_image)
                    stats["missing_images"] += 1
                stats["skipped_missing_image_annotations"] += 1
                continue
            grouped.setdefault(source_image, []).append(annotation_line(item))

    used_stems: dict[str, Path] = {}
    for source_image, lines in grouped.items():
        try:
            image_size(source_image)
        except ValueError as error:
            stats["unreadable_images"] += 1
            stats["skipped_unreadable_image_annotations"] += len(lines)
            print(f"skip unreadable image: {source_image} ({error})")
            continue
        stem = safe_image_stem(source_image)
        if stem in used_stems and used_stems[stem] != source_image:
            digest = hashlib.sha1(str(source_image).encode("utf-8")).hexdigest()[:12]
            stem = f"{stem}_{digest}"
        used_stems[stem] = source_image
        link_or_copy(source_image, output_images / f"{stem}.tif", copy_images)
        (output_ann / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["annotations"] += len(lines)

    return stats


def convert_full_fair1m(raw_root: Path, output_root: Path, fair_json: str, name: str, copy_images: bool) -> None:
    dataset_root = output_root / name
    split_sources = {
        "train": [
            raw_root / "fold_0" / "train.json",
            raw_root / "fold_0" / "val.json",
            raw_root / fair_json,
        ],
        "val": [raw_root / "test.json"],
        "test": [raw_root / "test.json"],
    }
    for split, annotation_files in split_sources.items():
        for annotation_file in annotation_files:
            if not annotation_file.is_file():
                raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        stats = convert_json_split(annotation_files, dataset_root / split, copy_images)
        print(f"{name} {split}: {stats}")


def write_meta(output_root: Path) -> None:
    payload = {
        "class_names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "format": "DOTA labelTxt: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty",
    }
    (output_root / "meta.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OBB annotations to DOTA format")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--all", action="store_true", help="convert fold_0..fold_4")
    parser.add_argument("--full-fair1m", action="store_true", help="convert full TZB train+val plus FAIR1M train JSON")
    parser.add_argument("--raw-root", type=Path, default=RAW_TZB_DATASET)
    parser.add_argument("--fair-json", default="train_FAIR1M1.0.json")
    parser.add_argument("--full-name", default="full_fair1m")
    parser.add_argument("--source", type=Path, default=SOURCE_PROJECT / "dataset_yolo")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "tzb_dota")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.full_fair1m:
        raw_root = args.raw_root.expanduser().resolve()
        if not raw_root.is_dir():
            raise FileNotFoundError(f"Raw TZB dataset does not exist: {raw_root}")
        convert_full_fair1m(raw_root, output_root, args.fair_json, args.full_name, args.copy_images)
        write_meta(output_root)
        write_meta(output_root / args.full_name)
        return

    source_root = args.source.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source YOLO dataset does not exist: {source_root}")

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
