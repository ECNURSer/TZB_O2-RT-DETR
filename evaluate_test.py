#!/usr/bin/env python3
"""Run OpenMMLab test diagnostics with DOTA mAP and project F1@0.3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from ai4rs.utils import register_all_modules
from experiment_results import append_result, metric_values
from project_utils import (
    CONFIGS,
    PROJECT_ROOT,
    require_dataset,
    set_data_root,
    set_imgsz,
    set_loader_options,
    set_max_det,
    setup_pythonpath,
)


def allow_full_checkpoint_loading() -> None:
    original_load = torch.load

    def load_with_full_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_full_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate O2-RT-DETR with DOTA mAP and F1@0.3 diagnostics")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(CONFIGS), default="r50")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--fixed-conf", type=float)
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-csv", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    return parser


def set_fixed_conf(cfg: Config, fixed_conf: float | None) -> None:
    if fixed_conf is None:
        return
    evaluators = cfg.test_evaluator if isinstance(cfg.test_evaluator, list) else [cfg.test_evaluator]
    for evaluator in evaluators:
        if evaluator.get("type") == "CompetitionF1Metric":
            evaluator["fixed_conf"] = fixed_conf


def main() -> None:
    setup_pythonpath()
    allow_full_checkpoint_loading()
    args = build_parser().parse_args()
    if "," in args.device:
        raise ValueError("Evaluation uses one GPU. Pass a single --device, such as 0 or 7.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    data_root = require_dataset(args.fold, (args.split,))

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(CONFIGS[args.model])
    set_data_root(cfg, data_root)
    set_loader_options(cfg, batch=args.batch, workers=args.workers)
    set_imgsz(cfg, args.imgsz)
    set_max_det(cfg, args.max_det)
    if args.split == "val":
        cfg.test_dataloader = cfg.val_dataloader
    set_fixed_conf(cfg, args.fixed_conf)
    run_name = args.name or f"{args.split}_o2_rtdetr_{args.model}vd_fold{args.fold}"
    cfg.work_dir = str(PROJECT_ROOT / "runs" / "test" / run_name)
    cfg.load_from = str(weights)

    runner = Runner.from_cfg(cfg) if "runner_type" not in cfg else RUNNERS.build(cfg)
    metrics = runner.test()
    output = args.output.expanduser().resolve() if args.output else Path(cfg.work_dir) / f"{args.split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_result(
        args.results_csv.expanduser().resolve(),
        {
            "stage": "test",
            "run_name": run_name,
            "model": args.model,
            "fold": args.fold,
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "weights": str(weights),
            "params_m": sum(parameter.numel() for parameter in runner.model.parameters()) / 1_000_000,
            "results_dir": str(Path(cfg.work_dir)),
            **metric_values(metrics),
        },
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"结果已保存: {output}")


if __name__ == "__main__":
    main()
