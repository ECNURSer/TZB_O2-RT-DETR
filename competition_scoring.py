#!/usr/bin/env python3
"""Competition-aligned OBB F1 scoring at polygon IoU 0.3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
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


def merge_matches(
    images: Iterable[tuple[Iterable[ObjectAnnotation], Iterable[ObjectAnnotation]]],
    iou_threshold: float = 0.3,
) -> tuple[list[MatchRecord], dict[int, int]]:
    """Match all images and aggregate GT class counts."""
    records: list[MatchRecord] = []
    total_gt: dict[int, int] = defaultdict(int)
    for predictions, ground_truths in images:
        image_records, image_gt = match_image(predictions, ground_truths, iou_threshold)
        records.extend(image_records)
        for class_id, count in image_gt.items():
            total_gt[class_id] += count
    return records, dict(total_gt)


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


def _confidence_states(records: Iterable[MatchRecord], total_gt: int) -> list[Score]:
    """Return cumulative TP/FP states at every distinct confidence for one class."""
    records = sorted(records, key=lambda record: record.confidence, reverse=True)
    states = [score_records([], total_gt, 1.0)]
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
        states.append(Score(confidence, tp, fp, fn, precision, recall, f1))
    return states


def best_class_confidences(records: Iterable[MatchRecord], total_gt_by_class: dict[int, int]) -> tuple[dict[int, float], Score]:
    """Find class-specific thresholds that maximize aggregate micro-F1."""
    grouped: dict[int, list[MatchRecord]] = defaultdict(list)
    records = list(records)
    for record in records:
        grouped[record.class_id].append(record)
    class_ids = sorted(set(grouped) | set(total_gt_by_class))
    states = {class_id: _confidence_states(grouped[class_id], total_gt_by_class.get(class_id, 0)) for class_id in class_ids}
    total_gt = sum(total_gt_by_class.values())
    target_f1 = best_confidence(records, total_gt).f1
    selected: dict[int, Score] = {}
    for _ in range(100):
        selected = {
            class_id: max(
                class_states,
                key=lambda score: ((2.0 - target_f1) * score.tp - target_f1 * score.fp, score.confidence),
            )
            for class_id, class_states in states.items()
        }
        tp = sum(score.tp for score in selected.values())
        fp = sum(score.fp for score in selected.values())
        fn = max(total_gt - tp, 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if abs(f1 - target_f1) < 1e-15:
            break
        target_f1 = f1
    thresholds = {class_id: score.confidence for class_id, score in selected.items()}
    return thresholds, Score(0.0, tp, fp, fn, precision, recall, f1)


def class_scores(records: Iterable[MatchRecord], total_gt_by_class: dict[int, int], confidence: float) -> dict[int, Score]:
    """Return per-class diagnostic scores at the selected global confidence."""
    grouped: dict[int, list[MatchRecord]] = defaultdict(list)
    for record in records:
        grouped[record.class_id].append(record)
    return {
        class_id: score_records(grouped[class_id], total_gt_by_class.get(class_id, 0), confidence)
        for class_id in sorted(set(grouped) | set(total_gt_by_class))
    }


def score_to_dict(score: Score) -> dict[str, float | int]:
    """Convert a score to a JSON-serializable dictionary."""
    return asdict(score)


def load_dota_ground_truth(label_file: Path, class_to_id: dict[str, int]) -> list[ObjectAnnotation]:
    """Load one DOTA-style label file into scoring annotations."""
    if not label_file.is_file():
        return []
    objects: list[ObjectAnnotation] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        class_name = parts[8]
        if class_name not in class_to_id:
            continue
        coords = [float(value) for value in parts[:8]]
        polygon = tuple((coords[index], coords[index + 1]) for index in range(0, 8, 2))
        objects.append(ObjectAnnotation(class_id=class_to_id[class_name], polygon=polygon))
    return objects
