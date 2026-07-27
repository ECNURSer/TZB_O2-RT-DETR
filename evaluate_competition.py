#!/usr/bin/env python3
"""Generate O2-RT-DETR OBB prediction caches and score competition F1."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config

from ai4rs.utils import register_all_modules
from competition_scoring import (
    ObjectAnnotation,
    best_class_confidences,
    best_confidence,
    class_scores,
    load_dota_ground_truth,
    merge_matches,
    score_records,
    score_to_dict,
)
from experiment_results import append_result
from inference_utils import infer_paths, infer_paths_tta, parse_tta_flips
from project_utils import CONFIGS, PROJECT_ROOT, require_dataset, resolve_data_root, set_data_root, set_imgsz, set_max_det, setup_pythonpath


def allow_full_checkpoint_loading() -> None:
    original_load = torch.load

    def load_with_full_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_full_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 O2-RT-DETR OBB F1@IoU0.3 评估与置信度搜索")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(CONFIGS), default="r50")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--dataset", help="named DOTA dataset under data/tzb_dota; overrides --fold")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Single inference GPU")
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.7, help="Used for TTA fusion; recorded only when TTA is disabled")
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--tta", action="store_true", help="Enable flip TTA on inference")
    parser.add_argument("--tta-flips", default="h,v,hv", help="Comma-separated flips for TTA: h,v,hv")
    parser.add_argument("--score-max-det", type=int)
    parser.add_argument("--chunk-size", type=int, help="Images processed per inference call; defaults to --batch")
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--fixed-conf", type=float)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--results-csv", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    return parser


def resolve_split(fold: int, split: str, dataset: str | None = None) -> tuple[list[Path], Path]:
    data_root = require_dataset(fold, (split,), dataset=dataset)
    image_dir = data_root / split / "images"
    label_dir = data_root / split / "annfiles"
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file() or path.is_symlink())
    stems = [path.stem for path in image_paths]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Image stems must be unique for cached scoring: {image_dir}")
    return image_paths, label_dir


def build_model(args: argparse.Namespace):
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(CONFIGS[args.model])
    set_data_root(cfg, resolve_data_root(args.fold, args.dataset))
    set_imgsz(cfg, args.imgsz)
    set_max_det(cfg, args.max_det)
    cfg.load_from = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(cfg, str(args.weights), device=device)
    class_names = tuple(cfg.get("class_names", cfg.get("metainfo", {}).get("classes", ())))
    if class_names:
        model.dataset_meta = {**getattr(model, "dataset_meta", {}), "classes": class_names}
    return model


def generate_cache(args: argparse.Namespace, image_paths: list[Path], cache_path: Path) -> dict:
    model = build_model(args)
    if args.tta and not args.tta_flips:
        raise ValueError("--tta requires at least one flip in --tta-flips")
    tta_multiplier = 1 + len(args.tta_flips) if args.tta else 1
    chunk_size = args.chunk_size or max(1, args.batch // tta_multiplier)
    started = time.perf_counter()
    images = []
    inference_seconds = 0.0
    inference_images = 0
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start : start + chunk_size]
        infer_started = time.perf_counter()
        if args.tta:
            chunk_images, inferred = infer_paths_tta(model, chunk, args.min_conf, args.max_det, args.tta_flips, args.nms_iou)
        else:
            chunk_images, inferred = infer_paths(model, chunk, args.min_conf, args.max_det)
        inference_seconds += time.perf_counter() - infer_started
        inference_images += inferred
        images.extend(chunk_images)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"device {args.device}: cached {min(start + len(chunk), len(image_paths))}/{len(image_paths)} images")
    wall_seconds = time.perf_counter() - started
    class_names = getattr(model, "dataset_meta", {}).get("classes", ())
    payload = {
        "schema_version": 1,
        "matching": "same-class confidence-greedy one-to-one polygon IoU",
        "model": args.model,
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "min_conf": args.min_conf,
        "nms_iou": args.nms_iou,
        "nms_note": "O2-RT-DETR inference is DETR-style; nms_iou is used only for TTA fusion.",
        "max_det": args.max_det,
        "tta": bool(args.tta),
        "tta_flips": list(args.tta_flips) if args.tta else [],
        "device": str(args.device),
        "image_count": len(images),
        "wall_seconds": wall_seconds,
        "inference_image_count": inference_images,
        "speed_ms_per_image": {
            "inference": 1000.0 * inference_seconds / len(images) if images else 0.0,
            "inference_per_augmented_image": 1000.0 * inference_seconds / inference_images if inference_images else 0.0,
            "wall": 1000.0 * wall_seconds / len(images) if images else 0.0,
        },
        "class_names": {str(index): name for index, name in enumerate(class_names)},
        "images": images,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def score_cache(
    payload: dict,
    label_dir: Path,
    iou_threshold: float,
    score_max_det: int | None = None,
    fixed_conf: float | None = None,
) -> dict:
    names = {int(key): value for key, value in payload.get("class_names", {}).items()}
    class_to_id = {name: class_id for class_id, name in names.items()}
    images = []
    for image in payload["images"]:
        cached_predictions = image["predictions"][:score_max_det] if score_max_det else image["predictions"]
        predictions = [
            ObjectAnnotation(
                class_id=int(item["class_id"]),
                confidence=float(item["confidence"]),
                polygon=tuple((float(x), float(y)) for x, y in item["polygon"]),
            )
            for item in cached_predictions
        ]
        targets = load_dota_ground_truth(label_dir / f"{image['image_id']}.txt", class_to_id)
        images.append((predictions, targets))
    records, total_gt_by_class = merge_matches(images, iou_threshold=iou_threshold)
    total_gt = sum(total_gt_by_class.values())
    if fixed_conf is None:
        selected = best_confidence(records, total_gt)
        class_thresholds, best_by_class = best_class_confidences(records, total_gt_by_class)
        optimization = {
            "threshold_mode": "optimized_on_evaluated_split",
            "best": score_to_dict(selected),
            "best_per_class": {
                "score": score_to_dict(best_by_class),
                "thresholds": {names.get(class_id, str(class_id)): confidence for class_id, confidence in class_thresholds.items()},
            },
        }
    else:
        selected = score_records(records, total_gt, fixed_conf)
        optimization = {
            "threshold_mode": "fixed",
            "fixed_conf": fixed_conf,
            "score": score_to_dict(selected),
        }
    per_class = class_scores(records, total_gt_by_class, selected.confidence)
    return {
        "metric": f"class-aware F1@polygon-IoU{iou_threshold:g}",
        "matching": "predictions sorted by confidence; same class; one GT matched at most once",
        **optimization,
        "per_class": {names.get(class_id, str(class_id)): score_to_dict(score) for class_id, score in per_class.items()},
        "total_ground_truths": total_gt,
        "prediction_cache": {
            key: payload.get(key)
            for key in (
                "model",
                "weights",
                "imgsz",
                "min_conf",
                "nms_iou",
                "max_det",
                "tta",
                "tta_flips",
                "image_count",
                "inference_image_count",
                "wall_seconds",
            )
        },
        "score_max_det": score_max_det or payload["max_det"],
        "speed_ms_per_image": payload.get("speed_ms_per_image", {}),
    }


def main() -> None:
    setup_pythonpath()
    allow_full_checkpoint_loading()
    args = build_parser().parse_args()
    if "," in str(args.device):
        raise ValueError("Competition inference uses one GPU; pass a single --device such as 7")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
    if args.score_max_det is not None and args.score_max_det <= 0:
        raise ValueError("--score-max-det must be positive")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be between 0 and 1")
    if args.fixed_conf is not None and not 0.0 <= args.fixed_conf <= 1.0:
        raise ValueError("--fixed-conf must be between 0 and 1")
    args.tta_flips = parse_tta_flips(args.tta_flips)
    if args.tta and not args.tta_flips:
        raise ValueError("--tta requires at least one flip in --tta-flips")
    args.weights = args.weights.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    image_paths, label_dir = resolve_split(args.fold, args.split, args.dataset)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        image_paths = image_paths[: args.limit]
    if args.reuse_cache:
        payload = json.loads(args.cache.read_text(encoding="utf-8"))
    else:
        payload = generate_cache(args, image_paths, args.cache)
    if bool(payload.get("tta", False)) != bool(args.tta):
        raise ValueError("Prediction cache TTA setting does not match --tta")
    if args.tta and tuple(payload.get("tta_flips", [])) != tuple(args.tta_flips):
        raise ValueError("Prediction cache TTA flips do not match --tta-flips")
    if args.tta and float(payload.get("nms_iou", args.nms_iou)) != float(args.nms_iou):
        raise ValueError("Prediction cache TTA NMS IoU does not match --nms-iou")
    if payload.get("image_count") != len(image_paths):
        raise ValueError("Prediction cache image count does not match the selected data split")
    expected_ids = [path.stem for path in image_paths]
    cached_ids = [image["image_id"] for image in payload.get("images", [])]
    if cached_ids != expected_ids:
        raise ValueError("Prediction cache image IDs do not match the selected data split")
    cached_weights = Path(payload.get("weights", "")).expanduser().resolve()
    if cached_weights != args.weights:
        raise ValueError(f"Prediction cache weights do not match --weights: {cached_weights}")
    if args.score_max_det is not None and args.score_max_det > int(payload["max_det"]):
        raise ValueError("--score-max-det cannot exceed the max_det used to generate the prediction cache")
    metrics = score_cache(payload, label_dir, args.match_iou, args.score_max_det, args.fixed_conf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
            "imgsz": args.imgsz,
            "batch": args.batch,
            "weights": str(args.weights),
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
