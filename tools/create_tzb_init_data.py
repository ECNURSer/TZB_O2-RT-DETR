#!/usr/bin/env python3
"""Create a clean TZB DOTA split from original image/XML annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml


RAW_IMAGE_DIR = Path("/data/work1/00_data/TZB/subject1/input_path")
RAW_GT_DIR = Path("/data/work1/00_data/TZB/subject1/gt")
OUTPUT_ROOT = PROJECT_ROOT / "data" / "tzb_dota" / "TZB_init_data"
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
    "Cargo-Truck": "Cargo-Truck",
    "Dump Truck": "Dump-Truck",
    "Dump-Truck": "Dump-Truck",
    "Excavator": "Excavator",
    "Small Car": "Small-Car",
    "Small-Car": "Small-Car",
    "Tractor": "Tractor",
    "Trailer": "Trailer",
    "Truck Tractor": "Truck-Tractor",
    "Truck-Tractor": "Truck-Tractor",
    "Van": "Van",
    "other-vehicle": "other-vehicle",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TZB_init_data in DOTA format")
    parser.add_argument("--image-dir", type=Path, default=RAW_IMAGE_DIR)
    parser.add_argument("--gt-dir", type=Path, default=RAW_GT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--copy-images", action="store_true")
    return parser


def safe_image_stem(image_path: Path) -> str:
    normalized = image_path.expanduser().as_posix()
    stem = re.sub(r"[^0-9A-Za-z]+", "_", normalized).strip("_")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}" if stem else digest


def image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_path}")
    return width, height


def file_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_point(text: str) -> tuple[float, float]:
    left, right = text.split(",", maxsplit=1)
    return float(left), float(right)


def parse_xml(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    lines = []
    for obj in root.findall(".//object"):
        label_node = obj.find("./possibleresult/name")
        if label_node is None or not (label_node.text or "").strip():
            continue
        raw_label = (label_node.text or "").strip()
        label = LABEL_ALIASES.get(raw_label)
        if label is None:
            raise ValueError(f"Unsupported label {raw_label!r} in {xml_path}")
        point_nodes = obj.findall("./points/point")
        if len(point_nodes) < 4:
            raise ValueError(f"Expected at least 4 points in {xml_path}")
        coords = []
        for point_node in point_nodes[:4]:
            x, y = parse_point((point_node.text or "").strip())
            coords.extend([x, y])
        coord_text = " ".join(f"{value:.3f}" for value in coords)
        lines.append(f"{coord_text} {label} 0")
    return lines


def link_or_copy(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def collect_records(image_dir: Path, gt_dir: Path) -> tuple[list[dict], dict]:
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})
    stats = {
        "raw_images": len(image_paths),
        "readable_images": 0,
        "unreadable_images": [],
        "missing_xml": [],
        "unsupported_xml": [],
        "objects": 0,
        "class_counts": Counter(),
    }
    records = []
    for image_path in image_paths:
        xml_path = gt_dir / f"{image_path.stem}.xml"
        if not xml_path.is_file():
            stats["missing_xml"].append(str(xml_path))
            continue
        try:
            width, height = image_size(image_path)
        except ValueError:
            stats["unreadable_images"].append(str(image_path))
            continue
        try:
            lines = parse_xml(xml_path)
        except Exception as error:
            stats["unsupported_xml"].append({"xml": str(xml_path), "error": str(error)})
            continue
        digest = file_sha1(image_path)
        for line in lines:
            stats["class_counts"][line.split()[8]] += 1
        stats["readable_images"] += 1
        stats["objects"] += len(lines)
        records.append(
            {
                "image": image_path,
                "xml": xml_path,
                "width": width,
                "height": height,
                "sha1": digest,
                "lines": lines,
            }
        )
    return records, stats


def split_records(records: list[dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[dict]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must be positive")
    train_ratio, val_ratio, test_ratio = (train_ratio / total_ratio, val_ratio / total_ratio, test_ratio / total_ratio)

    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        groups.setdefault(record["sha1"], []).append(record)
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)

    total_images = len(records)
    targets = {
        "train": round(total_images * train_ratio),
        "val": round(total_images * val_ratio),
    }
    targets["test"] = total_images - targets["train"] - targets["val"]

    splits = {"train": [], "val": [], "test": []}
    for _, group_records in group_items:
        counts = {split: len(items) for split, items in splits.items()}
        if counts["train"] + len(group_records) <= targets["train"]:
            target_split = "train"
        elif counts["val"] + len(group_records) <= targets["val"]:
            target_split = "val"
        else:
            target_split = "test"
        splits[target_split].extend(group_records)

    for split_records_ in splits.values():
        split_records_.sort(key=lambda item: int(item["image"].stem) if item["image"].stem.isdigit() else item["image"].stem)
    return splits


def write_split(output_root: Path, split: str, records: list[dict], copy_images: bool) -> dict:
    image_dir = output_root / split / "images"
    ann_dir = output_root / split / "annfiles"
    if image_dir.parent.exists():
        shutil.rmtree(image_dir.parent)
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    used_stems: dict[str, Path] = {}
    for record in records:
        source_image = record["image"]
        stem = safe_image_stem(source_image)
        if stem in used_stems and used_stems[stem] != source_image:
            stem = f"{stem}_{record['sha1'][:12]}"
        used_stems[stem] = source_image
        link_or_copy(source_image, image_dir / f"{stem}.tif", copy_images)
        (ann_dir / f"{stem}.txt").write_text("\n".join(record["lines"]) + ("\n" if record["lines"] else ""), encoding="utf-8")
        for line in record["lines"]:
            class_counts[line.split()[8]] += 1
    return {
        "images": len(records),
        "annfiles": len(records),
        "objects": sum(class_counts.values()),
        "class_counts": dict(class_counts),
    }


def check_duplicate_leakage(splits: dict[str, list[dict]]) -> dict:
    by_split = {split: {record["sha1"] for record in records} for split, records in splits.items()}
    return {
        "train_val": len(by_split["train"] & by_split["val"]),
        "train_test": len(by_split["train"] & by_split["test"]),
        "val_test": len(by_split["val"] & by_split["test"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    records, stats = collect_records(image_dir, gt_dir)
    if not records:
        raise RuntimeError("No valid records were collected")
    splits = split_records(records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_stats = {split: write_split(output_root, split, split_records_, args.copy_images) for split, split_records_ in splits.items()}
    duplicate_leakage = check_duplicate_leakage(splits)
    meta = {
        "name": output_root.name,
        "format": "DOTA labelTxt: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty",
        "source_images": str(image_dir),
        "source_gt": str(gt_dir),
        "split_strategy": "8:1:1 deterministic shuffle by exact image SHA1 groups",
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "class_names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "collection": {
            **{key: value for key, value in stats.items() if key != "class_counts"},
            "class_counts": dict(stats["class_counts"]),
        },
        "splits": split_stats,
        "duplicate_sha1_leakage": duplicate_leakage,
    }
    (output_root / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
