#!/usr/bin/env python3
"""Project helpers shared by train and evaluation wrappers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "r18": PROJECT_ROOT / "configs" / "o2_rtdetr_r18vd_tzb.py",
    "r34": PROJECT_ROOT / "configs" / "o2_rtdetr_r34vd_tzb.py",
    "r50": PROJECT_ROOT / "configs" / "o2_rtdetr_r50vd_tzb.py",
}


def setup_pythonpath() -> None:
    paths = [str(PROJECT_ROOT)]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)


def fold_data_root(fold: int) -> Path:
    return PROJECT_ROOT / "data" / "tzb_dota" / f"fold_{fold}"


def require_dataset(fold: int, splits: tuple[str, ...] = ("train", "val")) -> Path:
    data_root = fold_data_root(fold)
    missing = []
    for split in splits:
        for child in ("images", "annfiles"):
            path = data_root / split / child
            if not path.is_dir():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "DOTA data is not prepared. Run: "
            f"python convert_to_dota.py --fold {fold}. Missing: {missing[0]}"
        )
    return data_root


def set_data_root(cfg: Any, data_root: Path) -> None:
    root = str(data_root.resolve()) + "/"
    cfg.data_root = root
    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if loader_name in cfg:
            cfg[loader_name]["dataset"]["data_root"] = root


def set_loader_options(cfg: Any, batch: int | None = None, workers: int | None = None) -> None:
    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if loader_name not in cfg:
            continue
        if batch is not None:
            cfg[loader_name]["batch_size"] = batch
        if workers is not None:
            cfg[loader_name]["num_workers"] = workers
            cfg[loader_name]["persistent_workers"] = workers > 0


def set_imgsz(cfg: Any, imgsz: int | None) -> None:
    if imgsz is None:
        return
    cfg.imgsz = imgsz
    for pipeline_name in ("train_pipeline", "val_pipeline"):
        if pipeline_name in cfg:
            for transform in cfg[pipeline_name]:
                if transform.get("type") == "mmdet.Resize":
                    transform["scale"] = (imgsz, imgsz)
                if transform.get("type") == "mmdet.Pad":
                    transform["size"] = (imgsz, imgsz)
    for loader_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if loader_name not in cfg:
            continue
        for transform in cfg[loader_name]["dataset"]["pipeline"]:
            if transform.get("type") == "mmdet.Resize":
                transform["scale"] = (imgsz, imgsz)
            if transform.get("type") == "mmdet.Pad":
                transform["size"] = (imgsz, imgsz)


def set_max_det(cfg: Any, max_det: int | None) -> None:
    if max_det is not None:
        cfg.model.setdefault("test_cfg", {})["max_per_img"] = max_det


def latest_metrics(work_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    scalar_files = sorted(work_dir.rglob("scalars.json"), key=lambda path: path.stat().st_mtime)
    for scalar_file in scalar_files:
        for line in scalar_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, value in record.items():
                if key in {"step", "epoch", "iter"}:
                    continue
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
    return metrics


def best_checkpoint(work_dir: Path) -> Path:
    best = sorted(work_dir.glob("best_*.pth"), key=lambda path: path.stat().st_mtime)
    if best:
        return best[-1]
    latest = work_dir / "latest.pth"
    if latest.exists():
        return latest
    epochs = sorted(work_dir.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime)
    return epochs[-1] if epochs else latest
