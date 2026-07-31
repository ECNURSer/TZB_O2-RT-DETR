#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


DEFAULT_KEEP_KEYS = (
    "train/loss_bbox",
    "train/loss_cls",
    "train/loss_iou",
    "val/loss_bbox",
    "val/loss_cls",
    "val/loss_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild a filtered TensorBoard log from MMEngine scalars.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar_path(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("*/vis_data/scalars.json"))
    if not matches:
        raise FileNotFoundError(f"未找到 scalars.json: {run_dir}")
    return matches[-1]


def filtered_items(row: dict) -> dict[str, float]:
    items = {key: row[key] for key in DEFAULT_KEEP_KEYS if key in row}
    for key, value in row.items():
        if key.startswith("competition/"):
            items[key] = value
    return items


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output))
    kept = 0
    source = scalar_path(run_dir)
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            for key, value in filtered_items(row).items():
                writer.add_scalar(key, value, step)
                kept += 1
    writer.flush()
    writer.close()
    print(f"source={source}")
    print(f"output={output}")
    print(f"scalars_written={kept}")


if __name__ == "__main__":
    main()
