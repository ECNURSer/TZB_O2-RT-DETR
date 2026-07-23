"""MMEngine metric for the project competition F1@polygon-IoU0.3."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from ai4rs.registry import METRICS
from ai4rs.structures.bbox import rbox2qbox
from competition_scoring import ObjectAnnotation, best_confidence, match_image, score_records


@METRICS.register_module()
class CompetitionF1Metric(BaseMetric):
    """Compute class-aware, confidence-optimized F1 with one-to-one polygon matching."""

    default_prefix: Optional[str] = "competition"

    def __init__(
        self,
        iou_threshold: float = 0.3,
        fixed_conf: float | None = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if fixed_conf is not None and not 0.0 <= fixed_conf <= 1.0:
            raise ValueError("fixed_conf must be in [0, 1]")
        self.iou_threshold = iou_threshold
        self.fixed_conf = fixed_conf

    @staticmethod
    def _qboxes(boxes) -> np.ndarray:
        """Return boxes as Nx8 quadrilateral coordinates."""
        if hasattr(boxes, "tensor"):
            boxes = boxes.tensor
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes)
        boxes = boxes.detach().cpu().float()
        if boxes.numel() == 0:
            return np.zeros((0, 4, 2), dtype=np.float32)
        if boxes.shape[-1] == 5:
            boxes = rbox2qbox(boxes)
        elif boxes.shape[-1] != 8:
            raise ValueError(f"Expected rbox or qbox, got shape {tuple(boxes.shape)}")
        return boxes.reshape(-1, 4, 2).numpy()

    @staticmethod
    def _objects(boxes, labels, scores=None) -> list[ObjectAnnotation]:
        qboxes = CompetitionF1Metric._qboxes(boxes)
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.detach().cpu().numpy().astype(int)
        if scores is None:
            scores_array = np.ones(len(labels), dtype=np.float32)
        else:
            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores)
            scores_array = scores.detach().cpu().numpy().astype(float)
        return [
            ObjectAnnotation(
                class_id=int(class_id),
                confidence=float(score),
                polygon=tuple((float(x), float(y)) for x, y in polygon),
            )
            for polygon, class_id, score in zip(qboxes, labels, scores_array, strict=True)
        ]

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        """Collect per-image matched prediction records."""
        for data_sample in data_samples:
            gt_instances = data_sample["gt_instances"]
            pred_instances = data_sample["pred_instances"]
            targets = self._objects(gt_instances["bboxes"], gt_instances["labels"])
            predictions = self._objects(
                pred_instances["bboxes"],
                pred_instances["labels"],
                pred_instances["scores"],
            )
            records, gt_counts = match_image(predictions, targets, iou_threshold=self.iou_threshold)
            self.results.append((records, gt_counts))

    def compute_metrics(self, results: list) -> dict:
        """Aggregate matches and return TensorBoard-friendly scalar metrics."""
        all_records = []
        total_gt_by_class: dict[int, int] = defaultdict(int)
        for records, gt_counts in results:
            all_records.extend(records)
            for class_id, count in gt_counts.items():
                total_gt_by_class[int(class_id)] += int(count)
        total_gt = sum(total_gt_by_class.values())
        score = (
            score_records(all_records, total_gt, self.fixed_conf)
            if self.fixed_conf is not None
            else best_confidence(all_records, total_gt)
        )
        logger: MMLogger = MMLogger.get_current_instance()
        mode = "fixed" if self.fixed_conf is not None else "optimized"
        logger.info(
            "Competition F1@0.3: "
            f"P={score.precision:.5f}, R={score.recall:.5f}, F1={score.f1:.5f}, "
            f"conf={score.confidence:.5f}, TP={score.tp}, FP={score.fp}, FN={score.fn}, mode={mode}"
        )
        return {
            "precision": score.precision,
            "recall": score.recall,
            "F1@0.3": score.f1,
            "best_conf@0.3": score.confidence,
            "tp": score.tp,
            "fp": score.fp,
            "fn": score.fn,
        }
