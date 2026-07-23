#!/usr/bin/env python3
"""Competition-aligned OBB F1 scoring at polygon IoU 0.3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectAnnotation:
    class_id: int
    polygon: tuple[tuple[float, float], ...]
    confidence: float = 1.0


@dataclass(frozen=True)
class MatchRecord:
    confidence: float
    class_id: int
    true_positive: bool


@dataclass(frozen=True)
class Score:
    confidence: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def polygon_iou(first: Iterable[Iterable[float]], second: Iterable[Iterable[float]]) -> float:
    """Return exact convex-polygon IoU for two OBB quadrilaterals."""
    a = cv2.convexHull(np.asarray(first, dtype=np.float32).reshape(-1, 2))
    b = cv2.convexHull(np.asarray(second, dtype=np.float32).reshape(-1, 2))
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - float(intersection)
    return max(0.0, min(1.0, float(intersection) / union)) if union > 0.0 else 0.0


def match_image(
    predictions: Iterable[ObjectAnnotation],
    ground_truths: Iterable[ObjectAnnotation],
    iou_threshold: float = 0.3,
) -> tuple[list[MatchRecord], dict[int, int]]:
    """Greedily match confidence-sorted predictions to same-class unmatched GT boxes."""
    predictions = list(predictions)
    ground_truths = list(ground_truths)
    gt_by_class: dict[int, list[ObjectAnnotation]] = defaultdict(list)
    for target in ground_truths:
        gt_by_class[target.class_id].append(target)
    gt_counts = {class_id: len(items) for class_id, items in gt_by_class.items()}

    matched = {class_id: np.zeros(len(items), dtype=bool) for class_id, items in gt_by_class.items()}
    target_bounds = {
        class_id: np.asarray(
            [
                (
                    min(point[0] for point in target.polygon),
                    min(point[1] for point in target.polygon),
                    max(point[0] for point in target.polygon),
                    max(point[1] for point in target.polygon),
                )
                for target in items
            ],
            dtype=np.float32,
        )
        for class_id, items in gt_by_class.items()
    }
    records: list[MatchRecord] = []
    ordered = sorted(enumerate(predictions), key=lambda item: (-item[1].confidence, item[0]))
    for _, prediction in ordered:
        targets = gt_by_class.get(prediction.class_id, [])
        available = np.flatnonzero(~matched.get(prediction.class_id, np.empty(0, dtype=bool)))
        is_tp = False
        if len(available):
            points = np.asarray(prediction.polygon, dtype=np.float32)
            px1, py1 = points.min(axis=0)
            px2, py2 = points.max(axis=0)
            bounds = target_bounds[prediction.class_id][available]
            overlaps = (bounds[:, 0] < px2) & (bounds[:, 2] > px1) & (bounds[:, 1] < py2) & (bounds[:, 3] > py1)
            available = available[overlaps]
        if len(available):
            ious = np.asarray(
                [polygon_iou(prediction.polygon, targets[index].polygon) for index in available], dtype=np.float32
            )
            best_position = int(ious.argmax())
            if float(ious[best_position]) >= iou_threshold:
                matched[prediction.class_id][available[best_position]] = True
                is_tp = True
        records.append(MatchRecord(prediction.confidence, prediction.class_id, is_tp))
    return records, gt_counts


def score_records(records: Iterable[MatchRecord], total_gt: int, confidence: float) -> Score:
    """Score pre-matched predictions at one inclusive confidence threshold."""
    selected = [record for record in records if record.confidence >= confidence]
    tp = sum(record.true_positive for record in selected)
    fp = len(selected) - tp
    fn = max(total_gt - tp, 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return Score(confidence, tp, fp, fn, precision, recall, f1)


def best_confidence(records: Iterable[MatchRecord], total_gt: int) -> Score:
    """Find the exact best global confidence among all distinct prediction scores."""
    records = sorted(records, key=lambda record: record.confidence, reverse=True)
    if not records:
        return score_records([], total_gt, 1.0)

    best = score_records([], total_gt, 1.0)
    tp = fp = index = 0
    while index < len(records):
        confidence = records[index].confidence
        while index < len(records) and records[index].confidence == confidence:
            tp += int(records[index].true_positive)
            fp += int(not records[index].true_positive)
            index += 1
        fn = max(total_gt - tp, 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        candidate = Score(confidence, tp, fp, fn, precision, recall, f1)
        if (candidate.f1, candidate.confidence) > (best.f1, best.confidence):
            best = candidate
    return best


def score_to_dict(score: Score) -> dict[str, float | int]:
    """Convert a score to a JSON-serializable dictionary."""
    return asdict(score)
