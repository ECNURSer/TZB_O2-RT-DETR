#!/usr/bin/env python3
"""Shared experiment result recording for O2-RT-DETR runs."""

from __future__ import annotations

import csv
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULT_FIELDS = [
    "timestamp_utc",
    "stage",
    "run_name",
    "model",
    "fold",
    "split",
    "epochs",
    "imgsz",
    "batch",
    "weights",
    "precision",
    "recall",
    "f1",
    "map50",
    "map75",
    "map",
    "competition_precision",
    "competition_recall",
    "competition_f1_03",
    "competition_conf",
    "params_m",
    "results_dir",
]


def metric_values(metrics: dict[str, Any] | None) -> dict[str, float | str]:
    metrics = metrics or {}

    def pick(*keys: str) -> float | str:
        for key in keys:
            if key in metrics:
                value = metrics[key]
                if value is None:
                    continue
                return float(value)
        return ""

    precision = pick("competition/precision", "precision")
    recall = pick("competition/recall", "recall")
    f1 = pick("competition/F1@0.3", "F1@0.3")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": pick("dota/AP50", "AP50"),
        "map75": pick("dota/AP75", "AP75"),
        "map": pick("dota/mAP", "mAP"),
        "competition_precision": precision,
        "competition_recall": recall,
        "competition_f1_03": f1,
        "competition_conf": pick("competition/best_conf@0.3", "best_conf@0.3"),
    }


def append_result(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
    normalized["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with csv_path.open("a+", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        reader = csv.DictReader(stream)
        existing_rows = list(reader)
        existing_fields = reader.fieldnames or []
        if existing_fields and existing_fields != RESULT_FIELDS:
            stream.seek(0)
            stream.truncate()
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({field: existing.get(field, "") for field in RESULT_FIELDS})
        stream.seek(0, 2)
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(normalized)
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
