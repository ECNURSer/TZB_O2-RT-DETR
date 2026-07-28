#!/usr/bin/env python3
"""Merge sharded O2-RT-DETR prediction caches and score competition F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_competition import resolve_split, score_cache
from experiment_results import append_result
from project_utils import PROJECT_ROOT, setup_pythonpath


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge sharded competition caches and score them")
    parser.add_argument("--caches", nargs="+", type=Path, required=True)
    parser.add_argument("--model", default="r50")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--fixed-conf", type=float)
    parser.add_argument("--score-max-det", type=int)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    return parser


def stable_fields(payload: dict) -> dict:
    return {
        key: payload.get(key)
        for key in (
            "schema_version",
            "matching",
            "model",
            "weights",
            "imgsz",
            "min_conf",
            "nms_iou",
            "max_det",
            "tta",
            "tta_flips",
            "class_names",
        )
    }


def merge_payloads(payloads: list[dict], expected_ids: list[str]) -> dict:
    if not payloads:
        raise ValueError("No caches to merge")

    reference = stable_fields(payloads[0])
    images_by_id = {}
    wall_seconds = 0.0
    inference_image_count = 0
    inference_seconds = 0.0
    wall_per_image_seconds = 0.0
    total_images = 0

    for payload in payloads:
        if stable_fields(payload) != reference:
            raise ValueError("Cache metadata does not match")
        images = payload.get("images", [])
        for image in images:
            image_id = image["image_id"]
            if image_id in images_by_id:
                raise ValueError(f"Duplicate image in caches: {image_id}")
            images_by_id[image_id] = image
        image_count = int(payload.get("image_count", len(images)))
        total_images += image_count
        wall_seconds += float(payload.get("wall_seconds", 0.0))
        inference_image_count += int(payload.get("inference_image_count", 0))
        speed = payload.get("speed_ms_per_image", {})
        inference_seconds += float(speed.get("inference", 0.0)) * image_count / 1000.0
        wall_per_image_seconds += float(speed.get("wall", 0.0)) * image_count / 1000.0

    missing = [image_id for image_id in expected_ids if image_id not in images_by_id]
    extra = sorted(set(images_by_id) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"Merged cache does not match split: missing={len(missing)}, extra={len(extra)}")

    merged = dict(payloads[0])
    merged["device"] = "merged"
    merged["shards"] = [payload.get("shard", {}) for payload in payloads]
    merged["image_count"] = len(expected_ids)
    merged["wall_seconds"] = wall_seconds
    merged["inference_image_count"] = inference_image_count
    merged["speed_ms_per_image"] = {
        "inference": 1000.0 * inference_seconds / len(expected_ids) if expected_ids else 0.0,
        "inference_per_augmented_image": 1000.0 * inference_seconds / inference_image_count if inference_image_count else 0.0,
        "wall": 1000.0 * wall_per_image_seconds / total_images if total_images else 0.0,
        "parallel_wall": 1000.0 * max(float(payload.get("wall_seconds", 0.0)) for payload in payloads) / len(expected_ids)
        if expected_ids
        else 0.0,
    }
    merged["images"] = [images_by_id[image_id] for image_id in expected_ids]
    return merged


def main() -> None:
    setup_pythonpath()
    args = build_parser().parse_args()
    if args.fixed_conf is not None and not 0.0 <= args.fixed_conf <= 1.0:
        raise ValueError("--fixed-conf must be between 0 and 1")
    image_paths, label_dir = resolve_split(args.fold, args.split, args.dataset)
    expected_ids = [path.stem for path in image_paths]
    payloads = [json.loads(path.expanduser().resolve().read_text(encoding="utf-8")) for path in args.caches]
    merged = merge_payloads(payloads, expected_ids)

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_cache.write_text(json.dumps(merged, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = score_cache(merged, label_dir, args.match_iou, args.score_max_det, args.fixed_conf)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    score = metrics.get("score") or metrics.get("best", {})
    append_result(
        args.results_csv.expanduser().resolve(),
        {
            "stage": "competition",
            "run_name": args.output.stem,
            "model": args.model,
            "fold": args.dataset or args.fold,
            "split": args.split,
            "weights": merged.get("weights", ""),
            "precision": score.get("precision", ""),
            "recall": score.get("recall", ""),
            "f1": score.get("f1", ""),
            "competition_precision": score.get("precision", ""),
            "competition_recall": score.get("recall", ""),
            "competition_f1_03": score.get("f1", ""),
            "competition_conf": score.get("confidence", ""),
            "inference_ms": metrics.get("speed_ms_per_image", {}).get("inference", ""),
            "results_dir": str(args.output.parent),
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
