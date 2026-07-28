# TZB Subject1 O2-RT-DETR OBB

本项目是在原始 [wokaikaixinxin/O2-RT-DETR](https://github.com/wokaikaixinxin/O2-RT-DETR) 基础上，为 `TZB subject1` 旋转框目标检测任务适配的 OpenMMLab 版本工程。项目目标是保持原 O2-RT-DETR 模型结构、损失函数和匹配策略，同时对齐原 YOLO26 项目的数据组织、训练入口、日志输出、TensorBoard 监控、F1@0.3 评估和推理流程。

当前主线实验使用 `O2-RTDETR-R50`，数据格式为 DOTA OBB，核心指标为：

- `dota/mAP`
- `dota/AP50`
- `dota/AP75`
- `competition/F1@0.3`
- `competition/precision`
- `competition/recall`
- `competition/best_conf@0.3`

## 项目目录

```text
TZB-subject1-O2-RT-DETR/
├── ai4rs/                         # 原项目 AI4RS 核心模块
├── projects/rotated_rtdetr/        # O2-RTDETR 模型、head、loss、assigner 配置
├── configs/                        # 本项目 r18/r34/r50 训练配置入口
├── data/tzb_dota/                  # 转换后的 DOTA 格式数据，默认被 git 忽略
│   ├── fold_0/
│   └── full_fair1m/
├── runs/                           # 训练日志、权重、TensorBoard events，默认被 git 忽略
├── results/experiments.csv         # 实验结果汇总
├── convert_to_dota.py              # TZB / FAIR1M JSON 转 DOTA OBB
├── train.py                        # 训练入口，支持论文策略、YOLO 风格增强、早停、TB 过滤
├── evaluate_test.py                # DOTA mAP + F1@0.3 诊断评估
├── evaluate_competition.py         # 固定/搜索 conf 的 F1@0.3 评估，支持 TTA 和分片缓存
├── merge_competition_caches.py     # 多卡分片推理缓存合并与评分
├── predict.py                      # 单图/目录推理与可视化
└── run.sh                          # 统一命令入口
```

## 环境

推荐环境名：

```bash
conda activate o2-rtdetr-obb
```

主要依赖：

- Python `3.10`
- PyTorch：按本机 CUDA 版本使用 PyTorch 官网命令安装
- OpenMMLab：`mmengine`、`mmcv`、`mmdet`
- 常用库：`opencv-python`、`numpy`、`tensorboard`、`pandas`、`pypdf`

环境检查：

```bash
conda run --no-capture-output -n o2-rtdetr-obb python tools/check_env.py
```

说明：

- 不建议混用 YOLO 项目的 conda 环境。
- `setup_env.sh` 使用清华 PyPI 镜像安装普通 Python 包；PyTorch 建议按本机 CUDA 用官网命令手动安装。
- 如果 OpenMMLab 没有对应 CUDA/PyTorch 的 `mmcv` 预编译包，需要本机有可用 `nvcc` 才能源码编译。

## 数据组织

项目训练数据统一转换为 DOTA OBB 格式：

```text
data/tzb_dota/<dataset_name>/
├── train/
│   ├── images/
│   └── annfiles/
├── val/
│   ├── images/
│   └── annfiles/
└── test/
    ├── images/
    └── annfiles/
```

标注格式为 DOTA 多边形旋转框：

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

当前支持的数据集：

- `fold_0`：TZB 原始五折中的 fold0 组织。
- `full_fair1m`：TZB 全量训练数据合并 FAIR1M1.0 补充数据。

转换 fold0：

```bash
bash run.sh convert --fold 0
```

转换全量 TZB + FAIR1M：

```bash
bash run.sh convert \
  --full-fair1m \
  --full-name full_fair1m \
  --fair-json /data/work1/00_data/TZB/subject1/dataset/train_FAIR1M1.0.json
```

## 当前推荐训练：论文 FAIR1M 策略

当前主线训练不使用 YOLO 风格增强，不使用早停，按论文 FAIR1M 描述使用 random flip。

启动后四张卡训练：

```bash
tmux new-session -d -s o2_full_fair1m_paper_train \
"cd /home/dihan/TZB-subject1-O2-RT-DETR && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
CONDA_ENV=o2-rtdetr-obb bash run.sh train-r50 \
  --dataset full_fair1m \
  --device 4,5,6,7 \
  --epochs 80 \
  --batch 4 \
  --imgsz 1024 \
  --workers 8 \
  --save-period 5 \
  --val-interval 1 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --aug-profile fair1m-paper \
  --patience 0 \
  --name o2_rtdetr_r50vd_full_fair1m_paper_e80_b4_4g"
```

当前论文策略对应关系：

| 项目 | 当前配置 | 说明 |
|---|---:|---|
| 模型 | `O2-RTDETR-R50` | 原 O2-RTDETR R50 |
| 输入尺寸 | `1024` | 对齐论文 DOTA/FAIR1M patch 尺寸 |
| Optimizer | `AdamW` | 对齐论文 |
| Base LR | `1e-4` | 对齐论文 O2-RTDETR R50 |
| Backbone LR | `1e-5` | 通过 `lr_mult=0.1` 实现 |
| Weight decay | `1e-4` | 对齐论文 |
| Grad clip | `0.1` | 对齐论文 |
| Queries | `300` | 对齐论文 |
| Decoder layers | `6` | 对齐论文 |
| LR schedule | `LinearLR` warmup `2000 iters` | 对齐原项目配置 |
| 数据增强 | random flip only | 对齐论文 FAIR1M 描述 |
| Epochs | `80` | 用户指定；论文表中为 `72` |
| Batch | `4/卡 × 4卡 = 16` | 用户指定；论文为总 batch `8` |
| AMP | 关闭 | 当前 KLD/Hungarian 相关计算不启用半精度 |
| Early stopping | 关闭 | `--patience 0` |

## TensorBoard

当前 TensorBoard 只建议监控当前 run，避免旧实验事件文件污染曲线：

```bash
bash run.sh tensorboard \
  --logdir runs/o2_rtdetr_r50vd_full_fair1m_paper_e80_b4_4g_20260728_150704 \
  --port 6008
```

默认 TensorBoard 只保留核心标量：

- `base_lr`
- `lr`
- `loss`
- `loss_cls`
- `loss_bbox`
- `loss_iou`
- `grad_norm`
- `memory`
- `dota/mAP`
- `dota/AP50`
- `dota/AP75`
- `competition/precision`
- `competition/recall`
- `competition/F1@0.3`
- `competition/best_conf@0.3`

如果需要记录所有标量：

```bash
bash run.sh train-r50 ... --tb-scalar-keys all
```

## 训练与断点续训

普通 fold0 训练：

```bash
bash run.sh train-r50 \
  --fold 0 \
  --device 4,5,6,7 \
  --epochs 80 \
  --batch 4 \
  --name o2_rtdetr_r50vd_fold0_paper_e80
```

从完整 checkpoint 断点续训：

```bash
bash run.sh train-r50 \
  --dataset full_fair1m \
  --device 4,5,6,7 \
  --epochs 80 \
  --batch 4 \
  --imgsz 1024 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --aug-profile fair1m-paper \
  --patience 0 \
  --resume runs/<run_name>/epoch_35.pth \
  --name <run_name>
```

注意：

- `best_*.pth` 通常只保存模型权重，不一定包含 optimizer 和 scheduler。
- 真正完整断点续训应优先使用 `epoch_*.pth` 或 `last_checkpoint` 指向的 checkpoint。
- PyTorch 2.6+ 默认 `torch.load(weights_only=True)` 会影响 MMEngine 完整 checkpoint 读取；项目已在 `--resume` 时兼容完整 checkpoint 加载。

## 评估

### DOTA mAP + F1@0.3 诊断

```bash
bash run.sh test \
  --model r50 \
  --dataset full_fair1m \
  --split test \
  --weights runs/<run_name>/best_competition_F1@0.3_epoch_xx.pth \
  --fixed-conf <val_best_conf> \
  --output results/test_metrics.json \
  --device 4
```

### Competition F1@0.3

在验证集搜索最佳 conf：

```bash
bash run.sh competition \
  --model r50 \
  --dataset full_fair1m \
  --split val \
  --weights runs/<run_name>/best_competition_F1@0.3_epoch_xx.pth \
  --cache runs/competition/val_cache.json \
  --output runs/competition/val_metrics.json \
  --device 4
```

在 test 集使用验证集 conf，不重新寻优：

```bash
bash run.sh competition \
  --model r50 \
  --dataset full_fair1m \
  --split test \
  --fixed-conf <val_best_conf> \
  --weights runs/<run_name>/best_competition_F1@0.3_epoch_xx.pth \
  --cache runs/competition/test_cache.json \
  --output runs/competition/test_metrics.json \
  --device 4
```

## TTA 推理评估

TTA 作用在推理端，默认使用水平、垂直、水平+垂直翻转：

```bash
bash run.sh competition \
  --model r50 \
  --dataset full_fair1m \
  --split test \
  --tta \
  --fixed-conf <val_tta_best_conf> \
  --weights runs/<run_name>/best_competition_F1@0.3_epoch_xx.pth \
  --cache runs/competition/test_tta_cache.json \
  --output runs/competition/test_tta_metrics.json \
  --device 4
```

多卡分片生成 TTA cache 后合并：

```bash
for SHARD in 0 1 2 3; do
  GPU=$((4 + SHARD))
  bash run.sh competition \
    --model r50 \
    --dataset full_fair1m \
    --split test \
    --tta \
    --fixed-conf <val_tta_best_conf> \
    --num-shards 4 \
    --shard-index "$SHARD" \
    --weights runs/<run_name>/best_competition_F1@0.3_epoch_xx.pth \
    --cache runs/competition/test_tta_shard${SHARD}.json \
    --output runs/competition/test_tta_shard${SHARD}_metrics.json \
    --device "$GPU" &
done
wait

conda run --no-capture-output -n o2-rtdetr-obb python merge_competition_caches.py \
  --model r50 \
  --dataset full_fair1m \
  --split test \
  --fixed-conf <val_tta_best_conf> \
  --caches \
    runs/competition/test_tta_shard0.json \
    runs/competition/test_tta_shard1.json \
    runs/competition/test_tta_shard2.json \
    runs/competition/test_tta_shard3.json \
  --output-cache runs/competition/test_tta_cache.json \
  --output runs/competition/test_tta_metrics.json
```

注意：

- no-TTA 的最佳 conf 不一定适合 TTA。
- TTA 会增加候选框，通常需要重新在 val 上得到 TTA 对应的 fixed conf。
- 不建议在 test 上寻优 conf，除非只是诊断上限。

## 独立推理

单图或目录推理：

```bash
bash run.sh predict \
  --model r50 \
  --dataset full_fair1m \
  --weights runs/<run_name>/best_competition_F1@0.3_epoch_xx.pth \
  --source /path/to/images_or_one_image \
  --conf <val_best_conf> \
  --device 4 \
  --name predict
```

TTA 推理：

```bash
bash run.sh predict \
  --model r50 \
  --dataset full_fair1m \
  --tta \
  --weights runs/<run_name>/best_competition_F1@0.3_epoch_xx.pth \
  --source /path/to/images_or_one_image \
  --conf <val_tta_best_conf> \
  --device 4 \
  --name predict_tta
```

输出位置默认在：

```text
runs/predict/<name>/
```

## 结果文件

训练和评估结果会写入：

- `runs/<run_name>/`：配置、日志、权重、TensorBoard events、验证指标。
- `runs/competition/`：F1@0.3 评估 cache 和 metrics。
- `results/experiments.csv`：实验摘要。

`runs/`、`logs/`、`data/`、权重文件和 TensorBoard events 默认被 `.gitignore` 忽略，不会提交到 GitHub。

## 常用 tmux 命令

查看会话：

```bash
tmux ls
```

进入训练会话：

```bash
tmux attach -t o2_full_fair1m_paper_train
```

关闭训练：

```bash
tmux kill-session -t o2_full_fair1m_paper_train
```

关闭 TensorBoard：

```bash
tmux kill-session -t o2_full_fair1m_paper_tb_6008
```

## 关键实现说明

- 模型结构、OCD denoising、KLD loss、Chamfer matching cost 继承自 O2-RT-DETR / AI4RS。
- 本项目新增 `CompetitionF1Metric`，在训练验证阶段输出 `competition/F1@0.3`。
- `train.py` 支持 `fair1m-paper` 增强策略，用于 FAIR1M 论文风格训练。
- `train.py` 支持 TensorBoard 标量 allowlist，避免过多 decoder 层损失污染曲线。
- `evaluate_competition.py` 支持固定 conf、自动搜索 conf、TTA 和多卡分片 cache。
- `merge_competition_caches.py` 用于合并多卡分片推理结果并统一评分。

## 引用

```bibtex
@ARTICLE{11424629,
  author={Ding, Zeyu and Zhou, Yong and Zhao, Jiaqi and Du, Wen-Liang and Li, Xixi and Yao, Rui and Saddik, Abdulmotaleb El},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Real-Time Oriented Object Detection Transformer in Remote Sensing Images},
  year={2026},
  volume={64},
  number={5613014},
  pages={1-14},
  doi={10.1109/TGRS.2026.3671683}
}
```
