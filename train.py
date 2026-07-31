#!/usr/bin/env python3
"""Train O2-RT-DETR with project-compatible data, logging, and metrics."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
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


DEFAULT_TB_SCALAR_KEYS = (
    "train/loss_bbox",
    "train/loss_cls",
    "train/loss_iou",
    "val/loss_bbox",
    "val/loss_cls",
    "val/loss_iou",
    "competition/*",
)


FIXED_TRAINING_PRESET = {
    "dataset": "full_fair1m",
    "epochs": 80,
    "batch": 4,
    "imgsz": 1024,
    "workers": 8,
    "save_period": 5,
    "max_keep_ckpts": 5,
    "val_interval": 1,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "warmup_epochs": None,
    "lrf": 0.005,
    "cos_lr": False,
    "amp": False,
    "aug_profile": "fair1m-paper",
    "mosaic_epochs": 0,
    "patience": 0,
}


def allow_full_checkpoint_loading() -> None:
    original_load = torch.load

    def load_with_full_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_full_checkpoint


YOLO26M_FULL_PRESET = {
    "epochs": 1500,
    "batch": 12,
    "imgsz": 1280,
    "workers": 8,
    "save_period": 50,
    "val_interval": 1,
    "lr": 0.0012,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,
    "lrf": 0.005,
    "cos_lr": True,
    "amp": True,
    "aug_profile": "yolo26m",
    "mosaic_epochs": 40,
}


TRAINING_PRESETS = {
    "fixed": FIXED_TRAINING_PRESET,
    "yolo26m-full": YOLO26M_FULL_PRESET,
}


def option_passed(*names: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:] for name in names)


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset = TRAINING_PRESETS.get(args.preset)
    if preset is None:
        return
    option_map = {
        "dataset": ("--dataset", "--fold"),
        "epochs": ("--epochs",),
        "batch": ("--batch",),
        "imgsz": ("--imgsz",),
        "workers": ("--workers",),
        "save_period": ("--save-period",),
        "max_keep_ckpts": ("--max-keep-ckpts",),
        "val_interval": ("--val-interval",),
        "lr": ("--lr",),
        "weight_decay": ("--weight-decay",),
        "warmup_epochs": ("--warmup-epochs",),
        "lrf": ("--lrf",),
        "cos_lr": ("--cos-lr",),
        "amp": ("--amp",),
        "aug_profile": ("--aug-profile",),
        "mosaic_epochs": ("--mosaic-epochs",),
        "patience": ("--patience",),
    }
    for key, option_names in option_map.items():
        if key in preset and not option_passed(*option_names):
            setattr(args, key, preset[key])


def set_optimizer_schedule(cfg: Config, args: argparse.Namespace) -> None:
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
    if args.weight_decay is not None:
        cfg.optim_wrapper.optimizer.weight_decay = args.weight_decay

    if args.cos_lr or args.warmup_epochs is not None:
        lr = float(cfg.optim_wrapper.optimizer.lr)
        warmup_epochs = int(args.warmup_epochs or 0)
        schedulers = []
        if warmup_epochs > 0:
            schedulers.append(
                dict(
                    type="LinearLR",
                    start_factor=0.001,
                    by_epoch=True,
                    begin=0,
                    end=warmup_epochs,
                    convert_to_iter_based=True,
                )
            )
        if args.cos_lr:
            schedulers.append(
                dict(
                    type="CosineAnnealingLR",
                    eta_min=lr * float(args.lrf),
                    by_epoch=True,
                    begin=warmup_epochs,
                    end=args.epochs,
                    T_max=max(1, args.epochs - warmup_epochs),
                )
            )
        cfg.param_scheduler = schedulers


def yolo26m_load_pipeline() -> list[dict]:
    return [
        dict(type="mmdet.LoadImageFromFile", backend_args=None),
        dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
        dict(type="ai4rs.ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    ]


def yolo26m_post_pipeline(imgsz: int) -> list[dict]:
    return [
        dict(type="mmdet.YOLOXHSVRandomAug", hue_delta=3, saturation_delta=102, value_delta=77),
        dict(
            type="mmdet.RandomAffine",
            max_rotate_degree=0.0,
            max_translate_ratio=0.05,
            scaling_ratio_range=(0.8, 1.2),
            max_shear_degree=0.0,
            border=(0, 0),
            border_val=(114, 114, 114),
        ),
        dict(type="mmdet.RandomFlip", prob=0.5, direction="horizontal"),
        dict(type="mmdet.RandomFlip", prob=0.5, direction="vertical"),
        dict(type="ai4rs.RandomRotate", prob=1.0, angle_range=180),
        dict(type="ai4rs.RegularizeRotatedBox", angle_version="le90"),
        dict(type="mmdet.FilterAnnotations", min_gt_bbox_wh=(2, 2), keep_empty=False),
        dict(type="mmdet.Resize", scale=(imgsz, imgsz), keep_ratio=True),
        dict(type="mmdet.Pad", size=(imgsz, imgsz), pad_val=dict(img=(114, 114, 114))),
        dict(type="mmdet.PackDetInputs"),
    ]


def apply_yolo26m_augmentation(cfg: Config, imgsz: int, mosaic_epochs: int) -> None:
    post_pipeline = yolo26m_post_pipeline(imgsz)
    if mosaic_epochs <= 0:
        train_pipeline = yolo26m_load_pipeline() + post_pipeline
        cfg.train_pipeline = train_pipeline
        cfg.train_dataloader.dataset.pipeline = train_pipeline
        return

    inner_dataset = copy.deepcopy(cfg.train_dataloader.dataset)
    inner_dataset.type = "ai4rs.DOTADataset"
    inner_dataset.pipeline = yolo26m_load_pipeline()
    mosaic_pipeline = [
        dict(type="mmdet.Mosaic", img_scale=(imgsz, imgsz), prob=0.25, pad_val=114.0),
        *post_pipeline,
    ]
    cfg.train_pipeline = mosaic_pipeline
    cfg.train_pipeline_after_mosaic = post_pipeline
    cfg.train_dataloader.dataset = dict(
        type="mmdet.MultiImageMixDataset",
        dataset=inner_dataset,
        pipeline=mosaic_pipeline,
    )
    cfg.custom_hooks.append(
        dict(
            type="mmdet.PipelineSwitchHook",
            switch_epoch=mosaic_epochs,
            switch_pipeline=post_pipeline,
        )
    )


def fair1m_paper_pipeline(imgsz: int) -> list[dict]:
    return [
        dict(type="mmdet.LoadImageFromFile", backend_args=None),
        dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
        dict(type="ai4rs.ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
        dict(type="mmdet.Resize", scale=(imgsz, imgsz), keep_ratio=True),
        dict(type="mmdet.RandomFlip", prob=0.75, direction=["horizontal", "vertical", "diagonal"]),
        dict(type="mmdet.Pad", size=(imgsz, imgsz), pad_val=dict(img=(114, 114, 114))),
        dict(type="mmdet.PackDetInputs"),
    ]


def apply_fair1m_paper_augmentation(cfg: Config, imgsz: int) -> None:
    train_pipeline = fair1m_paper_pipeline(imgsz)
    cfg.train_pipeline = train_pipeline
    cfg.train_dataloader.dataset.pipeline = train_pipeline


def parse_scalar_keys(value: str) -> tuple[str, ...] | None:
    if value.lower() == "all":
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def scalar_key_allowed(key: str, patterns: tuple[str, ...] | None) -> bool:
    if patterns is None:
        return True
    for pattern in patterns:
        if pattern.endswith("*") and key.startswith(pattern[:-1]):
            return True
        if key == pattern:
            return True
    return False


def filter_visualizer_scalars(runner: Runner, scalar_keys: tuple[str, ...] | None) -> None:
    if scalar_keys is None:
        return
    add_scalars = runner.visualizer.add_scalars

    def add_filtered_scalars(scalars: dict, *args, **kwargs):
        filtered = {key: value for key, value in scalars.items() if scalar_key_allowed(key, scalar_keys)}
        if filtered:
            return add_scalars(filtered, *args, **kwargs)
        return None

    runner.visualizer.add_scalars = add_filtered_scalars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train O2-RT-DETR on the TZB OBB dataset")
    parser.add_argument("--model", choices=sorted(CONFIGS), default="r50")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--dataset", help="named DOTA dataset under data/tzb_dota; overrides --fold, e.g. full_fair1m")
    parser.add_argument(
        "--preset",
        choices=("fixed", "paper", "yolo26m-full"),
        default="fixed",
        help="fixed locks the current full_fair1m paper strategy used by the latest epoch80 run",
    )
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--batch", type=int, default=4, help="per-GPU batch size; paper uses 4 on 2 GPUs")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", help="handled by run.sh through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--name")
    parser.add_argument("--aug-profile", choices=("paper", "fair1m-paper", "yolo26m"), default="paper")
    parser.add_argument("--mosaic-epochs", type=int, default=0, help="YOLO-style mosaic epochs before switching it off")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--max-keep-ckpts", type=int, help="maximum checkpoints to keep; fixed preset uses 5")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--lr", type=float, help="AdamW base learning rate")
    parser.add_argument("--weight-decay", type=float, help="AdamW weight decay")
    parser.add_argument("--warmup-epochs", type=float, help="linear warmup epochs; yolo26m-full uses 5")
    parser.add_argument("--lrf", type=float, default=0.005, help="final LR ratio for cosine scheduler")
    parser.add_argument("--cos-lr", action="store_true", help="enable cosine LR scheduler")
    parser.add_argument("--amp", action="store_true", help="enable MMEngine AmpOptimWrapper")
    parser.add_argument("--patience", type=int, default=20, help="EarlyStoppingHook patience on F1@0.3; set 0 to disable")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--results-csv", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    parser.add_argument("--tb-scalar-keys", default=",".join(DEFAULT_TB_SCALAR_KEYS), help='comma-separated TensorBoard scalar allowlist, or "all"')
    parser.add_argument("--no-epoch-tb", action="store_true", help="disable epoch-level train/val TensorBoard scalar hook")
    parser.add_argument("--no-val-loss", action="store_true", help="do not compute validation loss scalars after each train epoch")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    setup_pythonpath()
    args = build_parser().parse_args()
    if args.resume:
        allow_full_checkpoint_loading()
    apply_preset_defaults(args)
    if args.device and "," not in args.device:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    data_root = require_dataset(args.fold, ("train", "val"), dataset=args.dataset)

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
    set_optimizer_schedule(cfg, args)
    if args.aug_profile == "fair1m-paper":
        apply_fair1m_paper_augmentation(cfg, args.imgsz)
    elif args.aug_profile == "yolo26m":
        apply_yolo26m_augmentation(cfg, args.imgsz, args.mosaic_epochs)

    cfg.max_epochs = args.epochs
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = args.val_interval
    cfg.default_hooks.checkpoint.interval = args.save_period
    if args.max_keep_ckpts is not None:
        cfg.default_hooks.checkpoint.max_keep_ckpts = args.max_keep_ckpts
    data_label = args.dataset or f"fold{args.fold}"
    run_name = args.name or f"o2_rtdetr_{args.model}vd_{data_label}"
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
    if not args.no_epoch_tb:
        cfg.custom_hooks.append(dict(type="ai4rs.EpochTensorboardScalarHook", log_val_loss=not args.no_val_loss))
    if args.resume:
        cfg.resume = True
        if args.resume != "auto":
            cfg.load_from = str(Path(args.resume).expanduser().resolve())

    print(cfg.pretty_text)
    if args.dry_run:
        print("配置检查通过；dry-run 未启动训练。")
        return

    runner = Runner.from_cfg(cfg) if "runner_type" not in cfg else RUNNERS.build(cfg)
    filter_visualizer_scalars(runner, parse_scalar_keys(args.tb_scalar_keys))
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
            "fold": args.dataset or args.fold,
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
