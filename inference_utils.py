#!/usr/bin/env python3
"""Shared O2-RT-DETR inference helpers, including flip TTA fusion."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector

from ai4rs.structures.bbox import rbox2qbox
from competition_scoring import polygon_iou


FLIP_CHOICES = {"h", "v", "hv"}


def parse_tta_flips(value: str) -> tuple[str, ...]:
    if not value or value.lower() in {"none", "off", "false"}:
        return ()
    flips = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = sorted(set(flips) - FLIP_CHOICES)
    if invalid:
        raise ValueError(f"Unsupported TTA flips: {invalid}. Supported: h,v,hv")
    return flips


def tta_variants(enabled: bool, flips: tuple[str, ...]) -> tuple[str, ...]:
    return ("orig", *flips) if enabled else ("orig",)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image cannot be read: {path}")
    return image


def image_shape(path: Path) -> tuple[int, int]:
    image = load_image(path)
    return int(image.shape[0]), int(image.shape[1])


def result_shape(result, fallback_path: Path) -> tuple[int, int]:
    metainfo = getattr(result, "metainfo", {}) or {}
    shape = metainfo.get("ori_shape") or metainfo.get("img_shape")
    if shape:
        return int(shape[0]), int(shape[1])
    return image_shape(fallback_path)


def result_predictions(result, min_conf: float, max_det: int) -> list[dict]:
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes
    if hasattr(bboxes, "tensor"):
        bboxes = bboxes.tensor
    bboxes = bboxes.detach().cpu().float()
    scores = pred_instances.scores.detach().cpu().float()
    labels = pred_instances.labels.detach().cpu().long()
    if bboxes.numel() == 0:
        return []
    if bboxes.shape[-1] == 5:
        qboxes = rbox2qbox(bboxes).reshape(-1, 4, 2)
    elif bboxes.shape[-1] == 8:
        qboxes = bboxes.reshape(-1, 4, 2)
    else:
        raise ValueError(f"Expected rbox or qbox predictions, got shape {tuple(bboxes.shape)}")
    keep = torch.nonzero(scores >= min_conf, as_tuple=False).flatten()
    if keep.numel() == 0:
        return []
    keep = keep[torch.argsort(scores[keep], descending=True)][:max_det]
    return [
        {
            "class_id": int(labels[index]),
            "confidence": float(scores[index]),
            "polygon": [[float(x), float(y)] for x, y in qboxes[index]],
        }
        for index in keep.tolist()
    ]


def flip_image(image: np.ndarray, variant: str) -> np.ndarray:
    if variant == "orig":
        return image
    if variant == "h":
        return cv2.flip(image, 1)
    if variant == "v":
        return cv2.flip(image, 0)
    if variant == "hv":
        return cv2.flip(image, -1)
    raise ValueError(f"Unsupported TTA variant: {variant}")


def invert_flip_polygon(polygon: list[list[float]], width: int, height: int, variant: str) -> list[list[float]]:
    restored = []
    for x, y in polygon:
        if "h" in variant:
            x = width - x
        if "v" in variant:
            y = height - y
        restored.append([max(0.0, min(float(width), x)), max(0.0, min(float(height), y))])
    return restored


def invert_flip_predictions(predictions: list[dict], width: int, height: int, variant: str) -> list[dict]:
    if variant == "orig":
        return predictions
    return [
        {
            "class_id": prediction["class_id"],
            "confidence": prediction["confidence"],
            "polygon": invert_flip_polygon(prediction["polygon"], width, height, variant),
        }
        for prediction in predictions
    ]


def polygon_bounds(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def bounds_overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def polygon_nms(predictions: list[dict], iou_threshold: float, max_det: int) -> list[dict]:
    if not predictions:
        return []
    if iou_threshold <= 0:
        return sorted(predictions, key=lambda item: item["confidence"], reverse=True)[:max_det]
    grouped: dict[int, list[dict]] = {}
    for prediction in predictions:
        grouped.setdefault(int(prediction["class_id"]), []).append(prediction)

    kept_all: list[dict] = []
    for class_predictions in grouped.values():
        kept: list[dict] = []
        kept_bounds: list[tuple[float, float, float, float]] = []
        for prediction in sorted(class_predictions, key=lambda item: item["confidence"], reverse=True):
            current_bounds = polygon_bounds(prediction["polygon"])
            suppressed = False
            for existing, existing_bounds in zip(kept, kept_bounds, strict=True):
                if not bounds_overlap(current_bounds, existing_bounds):
                    continue
                if polygon_iou(prediction["polygon"], existing["polygon"]) >= iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(prediction)
                kept_bounds.append(current_bounds)
        kept_all.extend(kept)
    return sorted(kept_all, key=lambda item: item["confidence"], reverse=True)[:max_det]


def infer_paths(
    model,
    image_paths: list[Path],
    min_conf: float,
    max_det: int,
) -> tuple[list[dict], float]:
    results = inference_detector(model, [str(path) for path in image_paths])
    if not isinstance(results, (list, tuple)):
        results = [results]
    images = []
    for source_path, result in zip(image_paths, results, strict=True):
        height, width = result_shape(result, source_path)
        images.append(
            {
                "image_id": source_path.stem,
                "width": width,
                "height": height,
                "predictions": result_predictions(result, min_conf, max_det),
            }
        )
    return images, len(image_paths)


def infer_paths_tta(
    model,
    image_paths: list[Path],
    min_conf: float,
    max_det: int,
    flips: tuple[str, ...],
    nms_iou: float,
) -> tuple[list[dict], float]:
    variants = tta_variants(True, flips)
    inputs = []
    specs = []
    for source_path in image_paths:
        image = load_image(source_path)
        height, width = int(image.shape[0]), int(image.shape[1])
        for variant in variants:
            inputs.append(flip_image(image, variant))
            specs.append((source_path, width, height, variant))

    results = inference_detector(model, inputs)
    if not isinstance(results, (list, tuple)):
        results = [results]

    by_image: dict[Path, dict] = {}
    for (source_path, width, height, variant), result in zip(specs, results, strict=True):
        record = by_image.setdefault(
            source_path,
            {
                "image_id": source_path.stem,
                "width": width,
                "height": height,
                "predictions": [],
            },
        )
        predictions = result_predictions(result, min_conf, max_det)
        record["predictions"].extend(invert_flip_predictions(predictions, width, height, variant))

    images = []
    for source_path in image_paths:
        record = by_image[source_path]
        record["predictions"] = polygon_nms(record["predictions"], nms_iou, max_det)
        images.append(record)
    return images, len(inputs)
