#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
GS_ROOT=/mnt/data/disk3/zhouyu/projects/gaussian-splatting
CONDA_SH=/mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0

BG_MODEL="$PROJECT_ROOT/outputs/background_3dgs/garden_30000"
BG_PLY="$BG_MODEL/point_cloud/iteration_30000/point_cloud.ply"
OBJ_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply"

OUT_ROOT="$PROJECT_ROOT/outputs/fusion_scene/A_local_search_known_anchor"
PREVIEW_ROOT="$PROJECT_ROOT/final_assets/fusion_debug/A_local_search_known_anchor"
LOG_DIR="$PROJECT_ROOT/logs/A_local_search_known_anchor"

mkdir -p "$OUT_ROOT" "$PREVIEW_ROOT" "$LOG_DIR"

# 这是前面已经验证“能看到 A”的桌面锚点，不再用新反查坐标
BASE_X=-0.17198236
BASE_Y=1.46132815
BASE_Z=1.10103631

# 局部搜索范围：在这个 3D 点附近轻微挪动
# 这 9 个位置会覆盖左/右/前/后/中心
OFFSETS=(
  "center:0.00:0.00"
  "x_m015:-0.15:0.00"
  "x_p015:0.15:0.00"
  "y_m015:0.00:-0.15"
  "y_p015:0.00:0.15"
  "xm015_ym015:-0.15:-0.15"
  "xm015_yp015:-0.15:0.15"
  "xp015_ym015:0.15:-0.15"
  "xp015_yp015:0.15:0.15"
)

# 0.18 太小，这里先试中等和偏大
SIZES=(0.22 0.26)

# 固定轻微抬高，避免陷进桌面
LIFT=0.035

set +u
source "$CONDA_SH"
conda activate gs_splatting
set -u

echo "=================================================="
echo "[INFO] Local search around known visible anchor"
echo "[INFO] BASE = ($BASE_X, $BASE_Y, $BASE_Z)"
echo "[INFO] sizes = ${SIZES[*]}"
echo "[INFO] lift  = $LIFT"
echo "[INFO] out   = $OUT_ROOT"
echo "=================================================="

export BG_PLY OBJ_PLY OUT_ROOT PREVIEW_ROOT
export BASE_X BASE_Y BASE_Z LIFT
export OFFSET_LIST="${OFFSETS[*]}"
export SIZE_LIST="${SIZES[*]}"

python - <<'PY'
from pathlib import Path
import os, json
import numpy as np
from plyfile import PlyData, PlyElement

bg_ply = Path(os.environ["BG_PLY"])
obj_ply = Path(os.environ["OBJ_PLY"])
out_root = Path(os.environ["OUT_ROOT"])
preview_root = Path(os.environ["PREVIEW_ROOT"])

base = np.array([
    float(os.environ["BASE_X"]),
    float(os.environ["BASE_Y"]),
    float(os.environ["BASE_Z"]),
], dtype=np.float32)

lift = float(os.environ["LIFT"])

offsets = []
for item in os.environ["OFFSET_LIST"].split():
    tag, dx, dy = item.split(":")
    offsets.append((tag, float(dx), float(dy)))

sizes = [float(x) for x in os.environ["SIZE_LIST"].split()]

bg = PlyData.read(bg_ply)
obj = PlyData.read(obj_ply)

bg_v = bg["vertex"].data
obj_v_base = obj["vertex"].data

xyz = np.stack([obj_v_base["x"], obj_v_base["y"], obj_v_base["z"]], axis=1).astype(np.float32)
bb_min = xyz.min(axis=0)
bb_max = xyz.max(axis=0)
dims = bb_max - bb_min
center_xy = 0.5 * (bb_min[:2] + bb_max[:2])

meta = []

for pos_tag, dx, dy in offsets:
    target = base + np.array([dx, dy, 0.0], dtype=np.float32)

    for size in sizes:
        size_tag = f"s{int(round(size * 100)):03d}"
        cand_tag = f"{pos_tag}_{size_tag}"

        model_dir = out_root / cand_tag
        out_ply = model_dir / "point_cloud/iteration_30000/point_cloud.ply"
        model_dir.mkdir(parents=True, exist_ok=True)
        out_ply.parent.mkdir(parents=True, exist_ok=True)

        obj_v = obj_v_base.copy()

        scale = size / float(np.max(dims))
        log_s = np.log(scale)

        xyz_new = xyz.copy()
        xyz_new[:, 0] = (xyz[:, 0] - center_xy[0]) * scale + target[0]
        xyz_new[:, 1] = (xyz[:, 1] - center_xy[1]) * scale + target[1]
        xyz_new[:, 2] = (xyz[:, 2] - bb_min[2]) * scale + target[2] + lift

        obj_v["x"] = xyz_new[:, 0]
        obj_v["y"] = xyz_new[:, 1]
        obj_v["z"] = xyz_new[:, 2]

        # 关键：同步缩放 Gaussian 本身尺度
        for k in ["scale_0", "scale_1", "scale_2"]:
            if k in obj_v.dtype.names:
                obj_v[k] = obj_v[k] + log_s

        merged = np.concatenate([bg_v, obj_v], axis=0)
        PlyData([PlyElement.describe(merged, "vertex")], text=False).write(out_ply)

        meta.append({
            "tag": cand_tag,
            "position_tag": pos_tag,
            "size": size,
            "lift": lift,
            "target_xyz": target.tolist(),
            "model_dir": str(model_dir),
            "scale": float(scale),
        })

        print("[DONE_MODEL]", cand_tag, "target=", target.tolist(), "size=", size, "scale=", float(scale))

preview_root.mkdir(parents=True, exist_ok=True)
with open(preview_root / "candidate_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("[DONE] meta:", preview_root / "candidate_meta.json")
PY

# 渲染所有候选
for MODEL_DIR in "$OUT_ROOT"/*; do
  [ -d "$MODEL_DIR" ] || continue
  TAG=$(basename "$MODEL_DIR")

  echo
  echo "========== RENDER $TAG =========="

  rsync -a --exclude='point_cloud' "$BG_MODEL"/ "$MODEL_DIR"/

  cd "$GS_ROOT"
  python render.py \
    -m "$MODEL_DIR" \
    --iteration 30000 \
    --skip_train \
    2>&1 | tee "$LOG_DIR/render_${TAG}.log"

  cd "$PROJECT_ROOT"
done

# 生成固定 00012 对比图 + best-diff 对比图
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import json, math

project = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
preview_root = project / "final_assets/fusion_debug/A_local_search_known_anchor"
out_root = project / "outputs/fusion_scene/A_local_search_known_anchor"

meta = json.loads((preview_root / "candidate_meta.json").read_text())

bg_dir = project / "outputs/background_3dgs/garden_30000/test/ours_30000/renders"
bg_imgs = {p.name: p for p in bg_dir.glob("*.png")}

def make_sheet(items, out_path, title_key):
    tiles = []
    for item, img_path, score in items:
        im = Image.open(img_path).convert("RGB")
        im.thumbnail((360, 230))
        canvas = Image.new("RGB", (360, 278), "white")
        canvas.paste(im, ((360 - im.width)//2, 0))

        d = ImageDraw.Draw(canvas)
        d.text((6, 232), item["tag"], fill=(0,0,0))
        d.text((6, 248), f"size={item['size']} lift={item['lift']}", fill=(0,0,0))
        d.text((6, 264), f"{title_key}={score:.3f}", fill=(0,0,0))
        tiles.append(canvas)

    cols = 3
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 360, rows * 278), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 360, (i // cols) * 278))
    sheet.save(out_path, quality=95)
    print("[DONE]", out_path)

fixed_items = []
bestdiff_items = []

for item in meta:
    model_dir = Path(item["model_dir"])
    render_dir = model_dir / "test/ours_30000/renders"
    if not render_dir.exists():
        continue

    fixed = render_dir / "00012.png"
    if fixed.exists():
        fixed_items.append((item, fixed, 0.0))

    # 对每个候选，找和背景差异最大的帧
    best_score = -1.0
    best_img = None
    for p in sorted(render_dir.glob("*.png")):
        bg_p = bg_imgs.get(p.name)
        if bg_p is None:
            continue

        im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
        bg = np.asarray(Image.open(bg_p).convert("RGB")).astype(np.float32)

        # 只关注桌面下半部分和中心附近，避免背景树叶变化干扰
        h, w = im.shape[:2]
        roi = (slice(int(h*0.30), int(h*0.85)), slice(int(w*0.20), int(w*0.80)))

        diff = np.mean(np.abs(im[roi] - bg[roi]))

        if diff > best_score:
            best_score = diff
            best_img = p

    if best_img is not None:
        bestdiff_items.append((item, best_img, best_score))

make_sheet(
    fixed_items,
    preview_root / "A_local_search_fixed_00012.jpg",
    "fixed"
)

make_sheet(
    bestdiff_items,
    preview_root / "A_local_search_bestdiff.jpg",
    "diff"
)

print("[INFO] fixed candidates:", len(fixed_items))
print("[INFO] bestdiff candidates:", len(bestdiff_items))
PY

echo
echo "=================================================="
echo "[DONE] local search finished."
echo "Open:"
echo "  $PREVIEW_ROOT/A_local_search_fixed_00012.jpg"
echo "  $PREVIEW_ROOT/A_local_search_bestdiff.jpg"
echo "  $PREVIEW_ROOT/candidate_meta.json"
echo "=================================================="
