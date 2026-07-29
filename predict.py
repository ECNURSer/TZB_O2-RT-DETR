#!/usr/bin/env python3
"""Run standalone O2-RT-DETR OBB inference with YOLO26-style outputs."""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
from pathlib import Path

import cv2
import torch
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config

from ai4rs.utils import register_all_modules
from inference_utils import infer_paths, infer_paths_tta, parse_tta_flips
from project_utils import CONFIGS, PROJECT_ROOT, resolve_data_root, set_data_root, set_imgsz, set_max_det, setup_pythonpath


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
COLORS = (
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
)


def allow_full_checkpoint_loading() -> None:
    original_load = torch.load

    def load_with_full_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_full_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="O2-RT-DETR OBB 独立推理")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Image file, directory, or glob pattern")
    parser.add_argument("--model", choices=sorted(CONFIGS), default="r50")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--dataset", help="named DOTA dataset under data/tzb_dota; overrides --fold")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.4781915247440338)
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tta", action="store_true", help="Enable flip TTA on inference")
    parser.add_argument("--tta-flips", default="h,v,hv", help="Comma-separated flips for TTA: h,v,hv")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="Polygon NMS IoU for TTA fusion")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "predict")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--nosave", action="store_true", help="Do not save visualization images")
    parser.add_argument("--no-txt", action="store_true", help="Do not save YOLO-OBB txt predictions")
    parser.add_argument("--txt-format", choices=("yolo", "pixel"), default="yolo")
    return parser


def resolve_source(source: str) -> list[Path]:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        paths = [path for path in source_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    elif source_path.is_file():
        paths = [source_path] if source_path.suffix.lower() in IMAGE_SUFFIXES else []
    else:
        paths = [
            Path(path)
            for path in glob.glob(source, recursive=True)
            if Path(path).is_file() and Path(path).suffix.lower() in IMAGE_SUFFIXES
        ]
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No supported images found from --source: {source}")
    return paths


def build_model(args: argparse.Namespace):
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(CONFIGS[args.model])
    set_data_root(cfg, resolve_data_root(args.fold, args.dataset))
    set_imgsz(cfg, args.imgsz)
    set_max_det(cfg, args.max_det)
    cfg.load_from = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(cfg, str(args.weights), device=device)
    class_names = tuple(cfg.get("class_names", cfg.get("metainfo", {}).get("classes", ())))
    if class_names:
        model.dataset_meta = {**getattr(model, "dataset_meta", {}), "classes": class_names}
    return model


def normalized_polygon(polygon: list[list[float]], width: int, height: int) -> list[float]:
    values: list[float] = []
    for x, y in polygon:
        values.extend([max(0.0, min(1.0, x / width)), max(0.0, min(1.0, y / height))])
    return values


def write_label(path: Path, predictions: list[dict], width: int, height: int, txt_format: str) -> None:
    lines = []
    for prediction in predictions:
        if txt_format == "yolo":
            values = normalized_polygon(prediction["polygon"], width, height)
            coords = " ".join(f"{value:.6f}" for value in values)
        else:
            coords = " ".join(f"{value:.3f}" for point in prediction["polygon"] for value in point)
        lines.append(f"{prediction['class_id']} {coords} {prediction['confidence']:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_drawable_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image cannot be read: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def draw_predictions(source_path: Path, output_path: Path, predictions: list[dict], class_names: tuple[str, ...]) -> None:
    image = load_drawable_image(source_path)
    for prediction in predictions:
        class_id = int(prediction["class_id"])
        confidence = float(prediction["confidence"])
        polygon = prediction["polygon"]
        points = torch.tensor(polygon, dtype=torch.float32).round().int().numpy()
        color = COLORS[class_id % len(COLORS)]
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        label = f"{class_names[class_id] if class_id < len(class_names) else class_id} {confidence:.2f}"
        x, y = int(points[:, 0].min()), int(points[:, 1].min())
        y = max(y - 4, 12)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def unique_stems(paths: list[Path]) -> dict[Path, str]:
    used: dict[str, int] = {}
    names = {}
    for path in paths:
        stem = path.stem
        index = used.get(stem, 0)
        used[stem] = index + 1
        names[path] = stem if index == 0 else f"{stem}_{index}"
    return names


def main() -> None:
    setup_pythonpath()
    allow_full_checkpoint_loading()
    args = build_parser().parse_args()
    if "," in str(args.device):
        raise ValueError("Prediction uses one GPU; pass a single --device such as 0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
    args.weights = args.weights.expanduser().resolve()
    args.project = args.project.expanduser().resolve()
    if not args.weights.is_file():
        raise FileNotFoundError(f"权重不存在: {args.weights}")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be between 0 and 1")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")

    image_paths = resolve_source(args.source)
    tta_flips = parse_tta_flips(args.tta_flips)
    if args.tta and not tta_flips:
        raise ValueError("--tta requires at least one flip in --tta-flips")
    tta_multiplier = 1 + len(tta_flips) if args.tta else 1
    chunk_size = max(1, args.batch // tta_multiplier) if args.tta else args.batch
    run_dir = args.project / args.name
    label_dir = run_dir / "labels"
    name_map = unique_stems(image_paths)
    model = build_model(args)
    class_names = tuple(getattr(model, "dataset_meta", {}).get("classes", ()))
    records = []
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start : start + chunk_size]
        if args.tta:
            chunk_records, _ = infer_paths_tta(model, chunk, args.conf, args.max_det, tta_flips, args.nms_iou)
        else:
            chunk_records, _ = infer_paths(model, chunk, args.conf, args.max_det)
        for image_path, record in zip(chunk, chunk_records, strict=True):
            height, width = int(record["height"]), int(record["width"])
            predictions = record["predictions"]
            output_stem = name_map[image_path]
            if not args.no_txt:
                write_label(label_dir / f"{output_stem}.txt", predictions, width, height, args.txt_format)
            if not args.nosave:
                draw_predictions(image_path, run_dir / f"{output_stem}{image_path.suffix}", predictions, class_names)
            records.append(
                {
                    "image_id": image_path.stem,
                    "source": str(image_path),
                    "width": width,
                    "height": height,
                    "predictions": predictions,
                }
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"predicted {min(start + len(chunk), len(image_paths))}/{len(image_paths)} images")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.json").write_text(
        json.dumps(
            {
                "weights": str(args.weights),
                "model": args.model,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "max_det": args.max_det,
                "tta": bool(args.tta),
                "tta_flips": list(tta_flips) if args.tta else [],
                "nms_iou": args.nms_iou,
                "class_names": {str(index): name for index, name in enumerate(class_names)},
                "images": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"结果目录: {run_dir}")


if __name__ == "__main__":
    main()
