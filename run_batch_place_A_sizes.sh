#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
GS_ROOT=/mnt/data/disk3/zhouyu/projects/gaussian-splatting
CONDA_SH=/mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh

# 如需指定 GPU，可取消下一行注释
export CUDA_VISIBLE_DEVICES=0

BG_MODEL="$PROJECT_ROOT/outputs/background_3dgs/garden_30000"
BG_PLY="$BG_MODEL/point_cloud/iteration_30000/point_cloud.ply"

OBJ_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply"
TABLE_XYZ="$PROJECT_ROOT/final_assets/fusion_debug/table_pick_xyz.npy"

OUT_BASE="$PROJECT_ROOT/outputs/fusion_scene/object_A_size_batch"
LOG_DIR="$PROJECT_ROOT/logs"
PREVIEW_OUT="$PROJECT_ROOT/final_assets/fusion_debug/object_A_size_compare.jpg"

# 这次固定试三个尺寸
SIZES=(0.18 0.22 0.26)

# 轻微抬高，避免插进桌面
LIFT=0.03

# 预览图默认取这个视角；若不存在会自动回退
RENDER_IDX="00012.png"

mkdir -p "$OUT_BASE" "$LOG_DIR" "$PROJECT_ROOT/final_assets/fusion_debug"

source "$CONDA_SH"
conda activate gs_splatting

echo "=================================================="
echo "[INFO] Batch placing Object A with sizes: ${SIZES[*]}"
echo "[INFO] Background model: $BG_MODEL"
echo "[INFO] Object A ply    : $OBJ_PLY"
echo "[INFO] Table xyz       : $TABLE_XYZ"
echo "[INFO] Output base     : $OUT_BASE"
echo "=================================================="

for SIZE in "${SIZES[@]}"; do
  TAG=$(python - <<PY
size = float("$SIZE")
print(f"{int(round(size * 100)):03d}")
PY
)

  MODEL_DIR="$OUT_BASE/garden_with_A_s${TAG}"
  OUT_PLY="$MODEL_DIR/point_cloud/iteration_30000/point_cloud.ply"

  echo
  echo "========== SIZE = $SIZE (tag=s${TAG}) =========="
  rm -rf "$MODEL_DIR"
  mkdir -p "$MODEL_DIR"

  # 复制背景模型结构（保留相机等元数据，不复制旧 point_cloud）
  rsync -a --exclude='point_cloud' "$BG_MODEL"/ "$MODEL_DIR"/
  mkdir -p "$MODEL_DIR/point_cloud/iteration_30000"

  export BG_PLY OBJ_PLY TABLE_XYZ OUT_PLY SIZE LIFT

  python - <<'PY'
import os
import numpy as np
from plyfile import PlyData, PlyElement

bg_ply = os.environ["BG_PLY"]
obj_ply = os.environ["OBJ_PLY"]
table_xyz_path = os.environ["TABLE_XYZ"]
out_ply = os.environ["OUT_PLY"]
size = float(os.environ["SIZE"])
lift = float(os.environ["LIFT"])

bg = PlyData.read(bg_ply)
obj = PlyData.read(obj_ply)

bg_v = bg["vertex"].data
obj_v = obj["vertex"].data.copy()

xyz = np.stack([obj_v["x"], obj_v["y"], obj_v["z"]], axis=1).astype(np.float32)

bb_min = xyz.min(axis=0)
bb_max = xyz.max(axis=0)
dims = bb_max - bb_min

target = np.load(table_xyz_path).astype(np.float32)

# 以物体最大边控制尺寸
scale = size / float(np.max(dims))

# 让物体在 xy 上居中到目标点，在 z 上以底面对齐桌面并加 lift
center_xy = 0.5 * (bb_min[:2] + bb_max[:2])

xyz_new = xyz.copy()
xyz_new[:, 0] = (xyz[:, 0] - center_xy[0]) * scale + target[0]
xyz_new[:, 1] = (xyz[:, 1] - center_xy[1]) * scale + target[1]
xyz_new[:, 2] = (xyz[:, 2] - bb_min[2]) * scale + target[2] + lift

obj_v["x"] = xyz_new[:, 0]
obj_v["y"] = xyz_new[:, 1]
obj_v["z"] = xyz_new[:, 2]

merged = np.concatenate([bg_v, obj_v], axis=0)
el = PlyElement.describe(merged, "vertex")
PlyData([el], text=False).write(out_ply)

print("[DONE] merged ply saved to:", out_ply)
print("[INFO] target size:", size)
print("[INFO] scale      :", scale)
print("[INFO] table xyz  :", target.tolist())
print("[INFO] lift       :", lift)
print("[INFO] obj dims before:", dims.tolist())
print("[INFO] obj dims after :", (dims * scale).tolist())
print("[INFO] bg points  :", len(bg_v))
print("[INFO] obj points :", len(obj_v))
print("[INFO] total      :", len(merged))
PY

  cd "$GS_ROOT"
  python render.py \
    -m "$MODEL_DIR" \
    --iteration 30000 \
    2>&1 | tee "$LOG_DIR/render_A_size_s${TAG}.log"
  cd "$PROJECT_ROOT"
done

# 生成三栏对比图
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

project = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
out_base = project / "outputs/fusion_scene/object_A_size_batch"
preview_out = project / "final_assets/fusion_debug/object_A_size_compare.jpg"

size_tags = [("018", "0.18"), ("022", "0.22"), ("026", "0.26")]
render_idx = "00012.png"

tiles = []
for tag, size in size_tags:
    model = out_base / f"garden_with_A_s{tag}"

    cand_paths = [
        model / "test/ours_30000/renders" / render_idx,
        model / "train/ours_30000/renders" / render_idx,
    ]

    img_path = None
    for p in cand_paths:
        if p.exists():
            img_path = p
            break

    if img_path is None:
        # fallback：找一个中间视角
        all_imgs = []
        for d in [
            model / "test/ours_30000/renders",
            model / "train/ours_30000/renders",
        ]:
            if d.exists():
                all_imgs.extend(sorted(d.glob("*.png")))
        if not all_imgs:
            continue
        img_path = all_imgs[len(all_imgs) // 2]

    img = Image.open(img_path).convert("RGB")
    tiles.append((size, img, img_path))

if not tiles:
    raise SystemExit("No rendered images found. Cannot build preview sheet.")

w, h = tiles[0][1].size
title_h = 50
sheet = Image.new("RGB", (w * len(tiles), h + title_h), (255, 255, 255))
draw = ImageDraw.Draw(sheet)

for i, (size, img, path) in enumerate(tiles):
    x = i * w
    sheet.paste(img, (x, title_h))
    draw.text((x + 12, 12), f"TARGET_SIZE = {size}", fill=(0, 0, 0))
    draw.text((x + 12, h + title_h - 24), path.name, fill=(0, 0, 0))

sheet.save(preview_out, quality=95)
print("[DONE] preview sheet saved to:", preview_out)
PY

echo
echo "=================================================="
echo "[DONE] All batch jobs finished."
echo "[RESULT] Preview sheet:"
echo "  $PREVIEW_OUT"
echo
echo "[RESULT] Per-size fused models:"
echo "  $OUT_BASE/garden_with_A_s018"
echo "  $OUT_BASE/garden_with_A_s022"
echo "  $OUT_BASE/garden_with_A_s026"
echo "=================================================="
