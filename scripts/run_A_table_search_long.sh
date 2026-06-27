#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
GS_ROOT=/mnt/data/disk3/zhouyu/projects/gaussian-splatting
CONDA_SH=/mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0

BG_MODEL="$PROJECT_ROOT/outputs/background_3dgs/garden_30000"
BG_PLY="$BG_MODEL/point_cloud/iteration_30000/point_cloud.ply"
OBJ_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply"

IMG="$PROJECT_ROOT/outputs/background_3dgs/garden_30000/test/ours_30000/renders/00012.png"
CAM_JSON="$BG_MODEL/cameras.json"

OUT_ROOT="$PROJECT_ROOT/outputs/fusion_scene/A_table_search_long"
PREVIEW_ROOT="$PROJECT_ROOT/final_assets/fusion_debug/A_table_search_long"
LOG_DIR="$PROJECT_ROOT/logs/A_table_search_long"

mkdir -p "$OUT_ROOT" "$PREVIEW_ROOT" "$LOG_DIR"

# 这几个点都在桌面木板上，尽量避开中央花瓶、金属圆盘中心和桌边缘。
# 格式：tag:x:y
ANCHORS=(
  "left_mid:610:520"
  "front_mid:760:625"
  "right_front:980:610"
  "right_mid:1070:550"
)

# 尺寸：0.18 偏小，0.22 中等，0.26 偏大
SIZES=(0.18 0.22 0.26)

# lift：轻微抬高，防止陷进桌面
LIFTS=(0.03)

# test/00012.png 对应 test split 的第 12 张
SPLIT=test
RENDER_INDEX=12
LLFFHOLD=8

set +u
source "$CONDA_SH"
conda activate gs_splatting
set -u

echo "=================================================="
echo "[INFO] long search for Object A table placement"
echo "[INFO] anchors: ${ANCHORS[*]}"
echo "[INFO] sizes  : ${SIZES[*]}"
echo "[INFO] lifts  : ${LIFTS[*]}"
echo "[INFO] output : $OUT_ROOT"
echo "=================================================="

# ============================================================
# Step 1. 根据多个桌面像素点反查 3D 坐标
# ============================================================
export PROJECT_ROOT BG_PLY OBJ_PLY IMG CAM_JSON PREVIEW_ROOT SPLIT RENDER_INDEX LLFFHOLD
export ANCHOR_LIST="${ANCHORS[*]}"

python - <<'PY'
from pathlib import Path
import os, json
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

project = Path(os.environ["PROJECT_ROOT"])
bg_ply = Path(os.environ["BG_PLY"])
img_path = Path(os.environ["IMG"])
cam_json = Path(os.environ["CAM_JSON"])
preview_root = Path(os.environ["PREVIEW_ROOT"])
split = os.environ["SPLIT"]
render_index = int(os.environ["RENDER_INDEX"])
llffhold = int(os.environ["LLFFHOLD"])

anchor_items = os.environ["ANCHOR_LIST"].split()
anchors = []
for item in anchor_items:
    tag, x, y = item.split(":")
    anchors.append((tag, float(x), float(y)))

img = Image.open(img_path).convert("RGB")
W, H = img.size

cams = json.loads(cam_json.read_text())
test_cams = [c for i, c in enumerate(cams) if i % llffhold == 0]
train_cams = [c for i, c in enumerate(cams) if i % llffhold != 0]
cam_list = test_cams if split == "test" else train_cams

if render_index >= len(cam_list):
    raise RuntimeError(f"render_index={render_index} out of range, len={len(cam_list)}")

cam = cam_list[render_index]
print("[INFO] total cams:", len(cams))
print("[INFO] test cams :", len(test_cams))
print("[INFO] selected camera:", cam.get("img_name", "unknown"))

fx = float(cam["fx"])
fy = float(cam["fy"])
cx = W / 2.0
cy = H / 2.0

C = np.array(cam["position"], dtype=np.float32)
R = np.array(cam["rotation"], dtype=np.float32)

ply = PlyData.read(bg_ply)
v = ply["vertex"].data
xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

# 先判断 R / R.T 哪个投影更合理：哪个可见点更多就用哪个。
variants = []
for variant in [0, 1]:
    if variant == 0:
        cam_xyz = (xyz - C) @ R
    else:
        cam_xyz = (xyz - C) @ R.T

    z = cam_xyz[:, 2]
    valid = z > 1e-6
    u = fx * (cam_xyz[:, 0] / z) + cx
    vv = fy * (cam_xyz[:, 1] / z) + cy
    valid &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    variants.append((variant, int(valid.sum()), cam_xyz, u, vv, valid))
    print("[INFO] variant", variant, "valid:", int(valid.sum()))

variant, _, cam_xyz, u, vv, valid = max(variants, key=lambda t: t[1])
print("[INFO] use projection variant:", variant)

anchor_results = []
debug = img.copy()
draw = ImageDraw.Draw(debug)

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
]

for i, (tag, px, py) in enumerate(anchors):
    du = u - px
    dv = vv - py
    pix_dist = np.sqrt(du * du + dv * dv)

    near = valid & (pix_dist < 8.0)
    if near.sum() < 5:
        near = valid & (pix_dist < 15.0)
    if near.sum() < 5:
        print("[WARN]", tag, "too few nearby points:", int(near.sum()))
        continue

    idxs = np.where(near)[0]
    depths = cam_xyz[idxs, 2]

    # 像素附近最前面的点，通常就是可见桌面
    best_idx = idxs[np.argmin(depths)]

    table_xyz = xyz[best_idx]
    proj_u = float(u[best_idx])
    proj_v = float(vv[best_idx])
    pd = float(pix_dist[best_idx])

    anchor_results.append({
        "tag": tag,
        "pixel_x": px,
        "pixel_y": py,
        "xyz": table_xyz.tolist(),
        "proj_u": proj_u,
        "proj_v": proj_v,
        "pix_dist": pd,
        "near_count": int(near.sum()),
    })

    color = colors[i % len(colors)]

    # 手选像素点：十字
    draw.ellipse((px - 9, py - 9, px + 9, py + 9), outline=color, width=3)
    draw.line((px - 16, py, px + 16, py), fill=color, width=2)
    draw.line((px, py - 16, px, py + 16), fill=color, width=2)
    draw.text((px + 10, py + 10), tag, fill=color)

    # 反投影点：小绿圈
    draw.ellipse((proj_u - 5, proj_v - 5, proj_u + 5, proj_v + 5), outline=(0, 255, 0), width=3)

    print(f"[ANCHOR] {tag}: pixel=({px:.1f},{py:.1f}) xyz={table_xyz.tolist()} proj=({proj_u:.1f},{proj_v:.1f}) pix_dist={pd:.2f} near={int(near.sum())}")

preview_root.mkdir(parents=True, exist_ok=True)
(debug).save(preview_root / "anchor_query_debug.jpg")

import csv
csv_path = preview_root / "anchor_xyz.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "pixel_x", "pixel_y", "x", "y", "z", "proj_u", "proj_v", "pix_dist", "near_count"])
    for r in anchor_results:
        x, y, z = r["xyz"]
        writer.writerow([r["tag"], r["pixel_x"], r["pixel_y"], x, y, z, r["proj_u"], r["proj_v"], r["pix_dist"], r["near_count"]])

np.save(preview_root / "anchor_xyz.npy", np.array([r["xyz"] for r in anchor_results], dtype=np.float32))

print("[DONE] anchor debug:", preview_root / "anchor_query_debug.jpg")
print("[DONE] anchor csv  :", csv_path)
PY

# ============================================================
# Step 2. 对 anchor x size x lift 生成并渲染候选
# ============================================================
python - <<'PY'
from pathlib import Path
import csv
import os
import numpy as np
from plyfile import PlyData, PlyElement
import json

project = Path(os.environ["PROJECT_ROOT"])
preview_root = Path(os.environ["PREVIEW_ROOT"])
bg_ply = Path(os.environ["BG_PLY"])
obj_ply = Path(os.environ["OBJ_PLY"])
out_root = project / "outputs/fusion_scene/A_table_search_long"

sizes = [0.18, 0.22, 0.26]
lifts = [0.02, 0.04]

anchor_csv = preview_root / "anchor_xyz.csv"

bg = PlyData.read(bg_ply)
obj = PlyData.read(obj_ply)
bg_v = bg["vertex"].data
obj_v_base = obj["vertex"].data

xyz = np.stack([obj_v_base["x"], obj_v_base["y"], obj_v_base["z"]], axis=1).astype(np.float32)
bb_min = xyz.min(axis=0)
bb_max = xyz.max(axis=0)
dims = bb_max - bb_min
center_xy = 0.5 * (bb_min[:2] + bb_max[:2])

rows = []
with open(anchor_csv) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

meta = []

for r in rows:
    tag = r["tag"]
    target = np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=np.float32)

    for size in sizes:
        for lift in lifts:
            size_tag = f"s{int(round(size * 100)):03d}"
            lift_tag = f"l{int(round(lift * 1000)):03d}"
            cand_tag = f"{tag}_{size_tag}_{lift_tag}"

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

            for k in ["scale_0", "scale_1", "scale_2"]:
                if k in obj_v.dtype.names:
                    obj_v[k] = obj_v[k] + log_s

            merged = np.concatenate([bg_v, obj_v], axis=0)
            PlyData([PlyElement.describe(merged, "vertex")], text=False).write(out_ply)

            meta.append({
                "tag": cand_tag,
                "anchor": tag,
                "size": size,
                "lift": lift,
                "target_xyz": target.tolist(),
                "model_dir": str(model_dir),
                "scale": float(scale),
            })

            print("[DONE_MODEL]", cand_tag, "size=", size, "lift=", lift, "target=", target.tolist())

with open(preview_root / "candidate_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("[DONE] candidate meta:", preview_root / "candidate_meta.json")
PY

# 复制背景模型非 point_cloud 文件并渲染
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

  # 删除大 PLY，保留渲染图片，节省空间。后续选中后可按 meta 重新生成。
  rm -rf "$MODEL_DIR/point_cloud"

  cd "$PROJECT_ROOT"
done

# ============================================================
# Step 3. 生成总览图
# ============================================================
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw
import json
import math

project = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
preview_root = project / "final_assets/fusion_debug/A_table_search_long"
out_root = project / "outputs/fusion_scene/A_table_search_long"

meta = json.loads((preview_root / "candidate_meta.json").read_text())

# 固定看 test 00012，便于比较同一视角
render_idx = "00012.png"

tiles = []

for item in meta:
    model_dir = Path(item["model_dir"])
    cand = [
        model_dir / "test/ours_30000/renders" / render_idx,
        model_dir / "train/ours_30000/renders" / render_idx,
    ]

    img_path = None
    for p in cand:
        if p.exists():
            img_path = p
            break

    if img_path is None:
        all_imgs = []
        for d in [
            model_dir / "test/ours_30000/renders",
            model_dir / "train/ours_30000/renders",
        ]:
            if d.exists():
                all_imgs.extend(sorted(d.glob("*.png")))
        if not all_imgs:
            continue
        img_path = all_imgs[len(all_imgs)//2]

    im = Image.open(img_path).convert("RGB")
    im.thumbnail((360, 230))

    canvas = Image.new("RGB", (360, 270), "white")
    canvas.paste(im, ((360 - im.width)//2, 0))

    d = ImageDraw.Draw(canvas)
    d.text((6, 234), item["tag"], fill=(0, 0, 0))
    d.text((6, 250), f"size={item['size']} lift={item['lift']}", fill=(0, 0, 0))

    tiles.append(canvas)

cols = 4
rows = math.ceil(len(tiles) / cols)
sheet = Image.new("RGB", (cols * 360, rows * 270), "white")

for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * 360, (i // cols) * 270))

out_path = preview_root / "A_table_search_overview.jpg"
sheet.save(out_path, quality=95)

print("[DONE] overview:", out_path)
print("[INFO] candidates:", len(tiles))
PY

echo
echo "=================================================="
echo "[DONE] Long table placement search finished."
echo "Open these files:"
echo "1) $PREVIEW_ROOT/anchor_query_debug.jpg"
echo "2) $PREVIEW_ROOT/A_table_search_overview.jpg"
echo "3) $PREVIEW_ROOT/candidate_meta.json"
echo "=================================================="
