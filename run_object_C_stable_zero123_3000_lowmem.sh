#!/usr/bin/env bash
set -euo pipefail

source /mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh
conda activate ts3d

export WORK_ROOT=/mnt/data/disk3/zhouyu
export PROJECT_ROOT="$WORK_ROOT/projects/gs_aigc_scene_fusion"
export TS_ROOT="$WORK_ROOT/projects/threestudio"

# 物理 GPU。想用 GPU1 时运行前 export PHYSICAL_GPU=1
export PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"

export IMAGE_PATH="$PROJECT_ROOT/data/object_C_image/processed/object_C_cup_rgba_512.png"
export ZERO123_CKPT="$TS_ROOT/load/zero123/stable_zero123.ckpt"

export TRAIN_TAG="cup_bear_3000_lowmem_gpu${PHYSICAL_GPU}"
export TEST_TAG="cup_bear_3000_lowmem_gpu${PHYSICAL_GPU}_test"

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/outputs/object_C_image3d"
mkdir -p "$PROJECT_ROOT/final_assets/object_C_image3d"
mkdir -p "$PROJECT_ROOT/report/figures/object_C"

# ============================================================
# CUDA / build environment
# ============================================================

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

export LD_LIBRARY_PATH="$PROJECT_ROOT/local_lib:/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$PROJECT_ROOT/local_lib:/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LIBRARY_PATH:-}"
export LDFLAGS="-L$PROJECT_ROOT/local_lib -L/usr/lib/x86_64-linux-gnu -L$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib64 ${LDFLAGS:-}"

export TORCH_CUDA_ARCH_LIST="8.6"
export TCNN_CUDA_ARCHITECTURES=86
export PYTHONPATH="$TS_ROOT:${PYTHONPATH:-}"

# 减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

# 离线模式
export HF_HOME="$WORK_ROOT/hf_cache/threestudio"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

echo "============================================================"
echo "[CHECK] image path: $IMAGE_PATH"
ls -lh "$IMAGE_PATH"

echo "============================================================"
echo "[CHECK] stable zero123 ckpt: $ZERO123_CKPT"
ls -lh "$ZERO123_CKPT"

python - <<'PY'
from PIL import Image
from pathlib import Path
import os

p = Path(os.environ["IMAGE_PATH"])
img = Image.open(p)
print("[IMAGE CHECK] mode:", img.mode)
print("[IMAGE CHECK] size:", img.size)
if img.mode == "RGBA":
    print("[IMAGE CHECK] alpha extrema:", img.getchannel("A").getextrema())
else:
    raise RuntimeError("Input image is not RGBA.")
PY

python - <<'PY'
import os, torch
print("[CUDA CHECK] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[CUDA CHECK] torch =", torch.__version__)
print("[CUDA CHECK] torch cuda =", torch.version.cuda)
print("[CUDA CHECK] cuda available =", torch.cuda.is_available())
print("[CUDA CHECK] device count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[CUDA CHECK] gpu =", torch.cuda.get_device_name(0))
    print("[CUDA CHECK] capability =", torch.cuda.get_device_capability(0))
else:
    raise RuntimeError("CUDA is not available.")
PY

cd "$TS_ROOT"

echo "============================================================"
echo "[TRAIN] stable-zero123 3000 steps, low-memory mode"
echo "[TRAIN] tag = $TRAIN_TAG"
echo "============================================================"

python launch.py \
  --config configs/stable-zero123.yaml \
  --train \
  --gpu 0 \
  exp_root_dir="$PROJECT_ROOT/outputs/object_C_image3d" \
  name="stable_zero123" \
  tag="$TRAIN_TAG" \
  data.image_path="$IMAGE_PATH" \
  system.guidance.pretrained_model_name_or_path="$ZERO123_CKPT" \
  data.random_camera.batch_size=1 \
  data.random_camera.eval_batch_size=1 \
  system.guidance.half_precision_weights=true \
  trainer.precision=16-mixed \
  trainer.max_steps=3000 \
  checkpoint.save_last=true \
  checkpoint.every_n_train_steps=500

echo "============================================================"
echo "[FIND CKPT]"
echo "============================================================"

CKPT=$(find "$PROJECT_ROOT/outputs/object_C_image3d/stable_zero123" \
  -type f -name "*.ckpt" | grep "$TRAIN_TAG" | sort | tail -n 1 || true)

if [ -z "$CKPT" ]; then
  echo "[ERROR] No checkpoint found for tag: $TRAIN_TAG"
  echo "[DEBUG] all ckpts:"
  find "$PROJECT_ROOT/outputs/object_C_image3d" -type f -name "*.ckpt" | sort || true
  exit 1
fi

echo "[INFO] CKPT = $CKPT"
ls -lh "$CKPT"

echo "============================================================"
echo "[TEST] render video from checkpoint"
echo "[TEST] tag = $TEST_TAG"
echo "============================================================"

python launch.py \
  --config configs/stable-zero123.yaml \
  --test \
  --gpu 0 \
  exp_root_dir="$PROJECT_ROOT/outputs/object_C_image3d" \
  name="stable_zero123" \
  tag="$TEST_TAG" \
  resume="$CKPT" \
  data.image_path="$IMAGE_PATH" \
  system.guidance.pretrained_model_name_or_path="$ZERO123_CKPT" \
  data.random_camera.eval_batch_size=1 \
  system.guidance.half_precision_weights=true \
  trainer.precision=16-mixed

echo "============================================================"
echo "[RESULT FILES]"
echo "============================================================"

find "$PROJECT_ROOT/outputs/object_C_image3d/stable_zero123" \
  -type f \( -name "*.mp4" -o -name "*.gif" -o -name "*.png" -o -name "*.jpg" -o -name "*.ckpt" \) | \
  grep -E "$TRAIN_TAG|$TEST_TAG" | sort | tail -n 150

LATEST_MP4=$(find "$PROJECT_ROOT/outputs/object_C_image3d/stable_zero123" \
  -type f -name "*.mp4" | grep -E "$TRAIN_TAG|$TEST_TAG" | sort | tail -n 1 || true)

if [ -n "$LATEST_MP4" ]; then
  cp "$LATEST_MP4" "$PROJECT_ROOT/final_assets/object_C_image3d/object_C_cup_bear_3000_lowmem_final.mp4"
  echo "[DONE] copied final video:"
  echo "$PROJECT_ROOT/final_assets/object_C_image3d/object_C_cup_bear_3000_lowmem_final.mp4"
else
  echo "[WARN] No mp4 found. Check png/gif outputs above."
fi

echo "============================================================"
echo "[DONE] Object C low-memory stable-zero123 pipeline finished."
echo "============================================================"
