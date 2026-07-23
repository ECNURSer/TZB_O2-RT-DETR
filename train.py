#!/usr/bin/env python3
"""Train O2-RT-DETR with project-compatible data, logging, and metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from ai4rs.utils import register_all_modules
from experiment_results import append_result, metric_values
from project_utils import (
    CONFIGS,
    PROJECT_ROOT,
    best_checkpoint,
    latest_metrics,
    require_dataset,
    set_data_root,
    set_imgsz,
    set_loader_options,
    set_max_det,
    setup_pythonpath,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train O2-RT-DETR on the TZB OBB dataset")
    parser.add_argument("--model", choices=sorted(CONFIGS), default="r50")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--batch", type=int, default=4, help="per-GPU batch size; paper uses 4 on 2 GPUs")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", help="handled by run.sh through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--name")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--amp", action="store_true", help="enable MMEngine AmpOptimWrapper")
    parser.add_argument("--patience", type=int, help="optional EarlyStoppingHook patience on F1@0.3")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--results-csv", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    setup_pythonpath()
    args = build_parser().parse_args()
    if args.device and "," not in args.device:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    data_root = require_dataset(args.fold, ("train", "val"))

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(CONFIGS[args.model])
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    set_data_root(cfg, data_root)
    set_loader_options(cfg, batch=args.batch, workers=args.workers)
    set_imgsz(cfg, args.imgsz)
    set_max_det(cfg, args.max_det)

    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = args.val_interval
    cfg.default_hooks.checkpoint.interval = args.save_period
    run_name = args.name or f"o2_rtdetr_{args.model}vd_fold{args.fold}"
    cfg.work_dir = str(PROJECT_ROOT / "runs" / run_name)

    if args.amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"
    if args.patience is not None and args.patience > 0:
        cfg.custom_hooks.append(
            dict(
                type="EarlyStoppingHook",
                monitor="competition/F1@0.3",
                rule="greater",
                patience=args.patience,
            )
        )
    if args.resume:
        cfg.resume = True
        if args.resume != "auto":
            cfg.load_from = str(Path(args.resume).expanduser().resolve())

    print(cfg.pretty_text)
    if args.dry_run:
        print("配置检查通过；dry-run 未启动训练。")
        return

    runner = Runner.from_cfg(cfg) if "runner_type" not in cfg else RUNNERS.build(cfg)
    runner.train()

    work_dir = Path(cfg.work_dir)
    metrics = latest_metrics(work_dir)
    params_m = sum(parameter.numel() for parameter in runner.model.parameters()) / 1_000_000
    best = best_checkpoint(work_dir)
    (work_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_result(
        args.results_csv.expanduser().resolve(),
        {
            "stage": "train_val",
            "run_name": run_name,
            "model": args.model,
            "fold": args.fold,
            "split": "val",
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "weights": str(best),
            "params_m": params_m,
            "results_dir": str(work_dir),
            **metric_values(metrics),
        },
    )
    print(f"训练完成: {work_dir}")
    print(f"TensorBoard: tensorboard --logdir {work_dir} --port 6006")


if __name__ == "__main__":
    main()
