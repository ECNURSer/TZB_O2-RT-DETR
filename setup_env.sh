#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${CONDA_ENV:-o2-rtdetr-obb}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
CONDA_MAIN_CHANNEL="${CONDA_MAIN_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
CONDA_R_CHANNEL="${CONDA_R_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_TAG}}"
MMENGINE_VERSION="${MMENGINE_VERSION:-0.10.7}"
MMCV_VERSION="${MMCV_VERSION:-2.1.0}"
MMDET_VERSION="${MMDET_VERSION:-3.3.0}"
MAX_JOBS="${MAX_JOBS:-8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

PIP_ARGS=(--timeout 300 --retries 20 -i "$PIP_MIRROR" --trusted-host "$PIP_TRUSTED_HOST")
CONDA_BASE_ARGS=(--override-channels -c "$CONDA_MAIN_CHANNEL" -c "$CONDA_R_CHANNEL")

detect_driver_cuda() {
    nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1
}

openmmlab_wheel_index_exists() {
    local index_url="$1"
    python - "$index_url" <<'PY'
import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=20) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except (urllib.error.URLError, TimeoutError):
    raise SystemExit(1)
PY
}

detect_cuda_home() {
    conda run -n "$ENV_NAME" python -c "import os, shutil; from pathlib import Path
try:
    from torch.utils.cpp_extension import CUDA_HOME
except Exception:
    CUDA_HOME = None
candidates = []
if CUDA_HOME:
    candidates.append(CUDA_HOME)
if os.environ.get('CUDA_HOME'):
    candidates.append(os.environ['CUDA_HOME'])
nvcc = shutil.which('nvcc')
if nvcc:
    candidates.append(str(Path(nvcc).resolve().parents[1]))
for path in candidates:
    if path and Path(path, 'bin', 'nvcc').exists():
        print(path)
        raise SystemExit(0)
raise SystemExit(1)"
}

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "${CONDA_BASE_ARGS[@]}" python=3.10 pip
fi

ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"

export PIP_INDEX_URL="$PIP_MIRROR"
export PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST"

conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${PIP_ARGS[@]}" --upgrade pip "setuptools<80" wheel
if conda run -n "$ENV_NAME" python -c "import torch, torchvision; ok=torch.__version__.startswith('${TORCH_VERSION}+') and torchvision.__version__.startswith('${TORCHVISION_VERSION}+'); raise SystemExit(0 if ok and torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "检测到兼容的 CUDA PyTorch，保留当前版本。"
else
    DRIVER_CUDA="$(detect_driver_cuda)"
    echo "检测到 NVIDIA Driver CUDA Version: ${DRIVER_CUDA:-unknown}"
    echo "使用 OpenMMLab 兼容的 PyTorch 官网命令安装: pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION --index-url $TORCH_INDEX_URL"
    conda run --no-capture-output -n "$ENV_NAME" python -m pip install --timeout 300 --retries 20 \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
        --index-url "$TORCH_INDEX_URL"
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${PIP_ARGS[@]}" \
    "setuptools<80" wheel ninja psutil cython "numpy==1.26.4" \
    "opencv-python==4.11.0.86" "opencv-python-headless==4.11.0.86" \
    openmim tensorboard pandas pypdf

MMCV_INDEX_URL="https://download.openmmlab.com/mmcv/dist/${TORCH_CUDA_TAG}/torch${TORCH_VERSION}/index.html"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${PIP_ARGS[@]}" \
    "mmengine==$MMENGINE_VERSION" "mmdet==$MMDET_VERSION"

if openmmlab_wheel_index_exists "$MMCV_INDEX_URL"; then
    conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${PIP_ARGS[@]}" \
        "mmcv==$MMCV_VERSION" -f "$MMCV_INDEX_URL"
else
    echo "OpenMMLab 未提供 $MMCV_INDEX_URL，改为按当前 PyTorch/CUDA 源码编译 mmcv==$MMCV_VERSION。"
    if [[ ! -x "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]]; then
        conda install -y -n "$ENV_NAME" "${CONDA_BASE_ARGS[@]}" "gxx_linux-64=11.2.0"
    fi
    CUDA_HOME_DETECTED="$(detect_cuda_home || true)"
    if [[ -z "$CUDA_HOME_DETECTED" ]]; then
        echo "错误：当前 torch/cuda 组合没有 mmcv 预编译轮子，源码编译需要 nvcc/CUDA_HOME。"
        echo "请先在当前环境安装 CUDA 12.8 编译工具链，或改用有 OpenMMLab 轮子的 torch/cu 组合。"
        exit 1
    fi
    echo "CUDA_HOME=$CUDA_HOME_DETECTED"
    CUDA_HOME="$CUDA_HOME_DETECTED" \
        PATH="$ENV_PREFIX/bin:$CUDA_HOME_DETECTED/bin:$PATH" \
        LD_LIBRARY_PATH="$CUDA_HOME_DETECTED/lib64:${LD_LIBRARY_PATH:-}" \
        CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
        CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
        CUDAHOSTCXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
        MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS="$MAX_JOBS" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        conda run --no-capture-output -n "$ENV_NAME" python -m pip install --no-build-isolation --no-cache-dir \
        "mmcv==$MMCV_VERSION"
fi
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${PIP_ARGS[@]}" -e "$PROJECT"

echo "环境创建完成: $ENV_NAME"
conda run --no-capture-output -n "$ENV_NAME" python "$PROJECT/tools/check_env.py"
