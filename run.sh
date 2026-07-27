#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${CONDA_ENV:-o2-rtdetr-obb}"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
MODE="${1:-help}"
shift || true

extract_device() {
    DEVICE=""
    ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device)
                [[ $# -ge 2 ]] || { echo "错误: --device 缺少参数" >&2; exit 2; }
                DEVICE="$2"
                shift 2
                ;;
            *)
                ARGS+=("$1")
                shift
                ;;
        esac
    done
}

run_python() {
    conda run --no-capture-output -n "$ENV_NAME" python "$@"
}

run_train() {
    MODEL="$1"
    shift
    extract_device "$@"
    if [[ -n "$DEVICE" ]]; then
        export CUDA_VISIBLE_DEVICES="$DEVICE"
    fi
    if [[ "$DEVICE" == *,* ]]; then
        GPU_COUNT="$(awk -F',' '{print NF}' <<<"$DEVICE")"
        conda run --no-capture-output -n "$ENV_NAME" \
            torchrun --standalone --nproc_per_node "$GPU_COUNT" \
            "$PROJECT/train.py" --model "$MODEL" --launcher pytorch "${ARGS[@]}"
    else
        run_python "$PROJECT/train.py" --model "$MODEL" "${ARGS[@]}"
    fi
}

run_eval() {
    SCRIPT="$1"
    shift
    extract_device "$@"
    if [[ "$DEVICE" == *,* ]]; then
        echo "错误: 评估只支持单 GPU，请传 --device 0 或 --device 7" >&2
        exit 2
    fi
    if [[ -n "$DEVICE" ]]; then
        export CUDA_VISIBLE_DEVICES="$DEVICE"
    fi
    run_python "$PROJECT/$SCRIPT" "${ARGS[@]}"
}

case "$MODE" in
    convert)
        run_python "$PROJECT/convert_to_dota.py" "$@"
        ;;
    train-r18)
        run_train r18 "$@"
        ;;
    train-r34)
        run_train r34 "$@"
        ;;
    train-r50)
        run_train r50 "$@"
        ;;
    test)
        run_eval evaluate_test.py "$@"
        ;;
    competition)
        run_eval evaluate_competition.py "$@"
        ;;
    predict)
        run_eval predict.py "$@"
        ;;
    tensorboard)
        LOGDIR=""
        PORT="6006"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --logdir)
                    [[ $# -ge 2 ]] || { echo "错误: --logdir 缺少目录" >&2; exit 2; }
                    LOGDIR="$2"
                    shift 2
                    ;;
                --port)
                    [[ $# -ge 2 ]] || { echo "错误: --port 缺少端口号" >&2; exit 2; }
                    PORT="$2"
                    shift 2
                    ;;
                *)
                    echo "错误: tensorboard 不支持参数 $1" >&2
                    exit 2
                    ;;
            esac
        done
        [[ -n "$LOGDIR" ]] || {
            echo "错误: 必须指定 --logdir，例如 runs/o2_rtdetr_r50vd_fold0" >&2
            exit 2
        }
        [[ "$LOGDIR" = /* ]] || LOGDIR="$PROJECT/$LOGDIR"
        [[ -d "$LOGDIR" ]] || { echo "错误: TensorBoard 日志目录不存在: $LOGDIR" >&2; exit 2; }
        EVENT_FILE="$(find "$LOGDIR" -type f -name 'events.out.tfevents.*' -print -quit)"
        [[ -n "$EVENT_FILE" ]] || {
            echo "错误: 目录中没有 events.out.tfevents.* 文件: $LOGDIR" >&2
            exit 2
        }
        echo "TensorBoard 日志目录: $LOGDIR"
        echo "检测到 events 文件: $EVENT_FILE"
        echo "访问地址: http://服务器IP:$PORT"
        conda run --no-capture-output -n "$ENV_NAME" tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
        ;;
    help|*)
        cat <<'EOF'
用法:
  bash run.sh convert --fold 0
  bash run.sh convert --full-fair1m --full-name full_fair1m
  bash run.sh train-r50 --dataset full_fair1m --preset yolo26m-full --device 0,1,2,3,4,5,6,7 --name o2_rtdetr_r50vd_full_fair1m_yoloaug_mosaic40
  bash run.sh train-r50 --fold 0 --device 0,1 --batch 4 --name o2_rtdetr_r50vd_fold0
  bash run.sh train-r34 --fold 0 --device 0,1 --batch 4 --name o2_rtdetr_r34vd_fold0
  bash run.sh test --model r50 --dataset full_fair1m --split test --fixed-conf <val_conf> --weights <best.pth> --output results/test_metrics.json --device 0
  bash run.sh competition --model r50 --dataset full_fair1m --split val --weights <best.pth> --cache runs/competition/val_cache.json --output runs/competition/val_metrics.json --device 0
  bash run.sh competition --model r50 --dataset full_fair1m --split test --fixed-conf <val_conf> --weights <best.pth> --cache runs/competition/test_cache.json --output runs/competition/test_metrics.json --device 0
  bash run.sh competition --model r50 --dataset full_fair1m --split test --tta --fixed-conf <val_conf> --weights <best.pth> --cache runs/competition/test_tta_cache.json --output runs/competition/test_tta_metrics.json --device 0
  bash run.sh predict --model r50 --dataset full_fair1m --weights <best.pth> --source <image-or-dir> --device 0 --name predict
  bash run.sh predict --model r50 --dataset full_fair1m --tta --weights <best.pth> --source <image-or-dir> --device 0 --name predict_tta
  bash run.sh tensorboard --logdir runs/o2_rtdetr_r50vd_fold0 --port 6007
EOF
        ;;
esac
