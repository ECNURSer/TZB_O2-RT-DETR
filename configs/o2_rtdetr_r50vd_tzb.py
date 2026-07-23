from mmengine.config import read_base
with read_base():
    from projects.rotated_rtdetr.configs.o2_rtdetr_r50vd_2xb4_72e_dota import *


data_root = "data/tzb_dota/fold_0/"
class_names = (
    "Bus",
    "Cargo-Truck",
    "Dump-Truck",
    "Excavator",
    "Small-Car",
    "Tractor",
    "Trailer",
    "Truck-Tractor",
    "Van",
    "other-vehicle",
)
metainfo = dict(classes=class_names)
imgsz = 1280

model.update(bbox_head=dict(num_classes=10), test_cfg=dict(max_per_img=600))

train_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=None),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.Resize", scale=(imgsz, imgsz), keep_ratio=True),
    dict(type="mmdet.RandomFlip", prob=0.75, direction=["horizontal", "vertical", "diagonal"]),
    dict(type="RandomRotate", prob=0.5, angle_range=180),
    dict(type="mmdet.Pad", size=(imgsz, imgsz), pad_val=dict(img=(114, 114, 114))),
    dict(type="mmdet.PackDetInputs"),
]
val_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=None),
    dict(type="mmdet.Resize", scale=(imgsz, imgsz), keep_ratio=True),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.Pad", size=(imgsz, imgsz), pad_val=dict(img=(114, 114, 114))),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type="DOTADataset",
        data_root=data_root,
        ann_file="train/annfiles/",
        data_prefix=dict(img_path="train/images/"),
        img_suffix="tif",
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DOTADataset",
        data_root=data_root,
        ann_file="val/annfiles/",
        data_prefix=dict(img_path="val/images/"),
        img_suffix="tif",
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DOTADataset",
        data_root=data_root,
        ann_file="test/annfiles/",
        data_prefix=dict(img_path="test/images/"),
        img_suffix="tif",
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

val_evaluator = [
    dict(type="DOTAMetric", metric="mAP", iou_thrs=[0.5, 0.75]),
    dict(type="CompetitionF1Metric", iou_threshold=0.3),
]
test_evaluator = val_evaluator

train_cfg.update(max_epochs=72, val_interval=1)
default_hooks.update(
    logger=dict(interval=50),
    checkpoint=dict(interval=1, max_keep_ckpts=99999, save_best="competition/F1@0.3", rule="greater"),
)
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(type="RotLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
work_dir = "runs/o2_rtdetr_r50vd_fold0"
