#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
GS_ROOT=/mnt/data/disk3/zhouyu/projects/gaussian-splatting
CONDA_SH=/mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=1

BG_MODEL="$PROJECT_ROOT/outputs/background_3dgs/garden_30000"
BG_PLY="$BG_MODEL/point_cloud/iteration_30000/point_cloud.ply"

A_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply"
B_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_B/object_B_gaussian_asset.ply"
C_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_C/object_C_gaussian_asset.ply"

OUT_ROOT="$PROJECT_ROOT/outputs/fusion_scene/final_ABC_quick"
PREVIEW_DIR="$PROJECT_ROOT/final_assets/fusion_final_quick"
LOG_DIR="$PROJECT_ROOT/logs/final_ABC_quick"

mkdir -p "$OUT_ROOT" "$PREVIEW_DIR" "$LOG_DIR"

set +u
source "$CONDA_SH"
conda activate gs_splatting
set -u

echo "=================================================="
echo "[INFO] Quick final fusion: garden + A + B + C"
echo "[INFO] GPU: physical GPU1 via CUDA_VISIBLE_DEVICES=1"
echo "=================================================="

# ============================================================
# Step 1. 生成 3 个 final 候选版本
# ============================================================
export PROJECT_ROOT BG_MODEL BG_PLY A_PLY B_PLY C_PLY OUT_ROOT PREVIEW_DIR

python - <<'PY'
from pathlib import Path
import os
import json
import shutil
import subprocess
import numpy as np
from plyfile import PlyData, PlyElement

project = Path(os.environ["PROJECT_ROOT"])
bg_model = Path(os.environ["BG_MODEL"])
bg_ply = Path(os.environ["BG_PLY"])
a_ply = Path(os.environ["A_PLY"])
b_ply = Path(os.environ["B_PLY"])
c_ply = Path(os.environ["C_PLY"])
out_root = Path(os.environ["OUT_ROOT"])
preview_dir = Path(os.environ["PREVIEW_DIR"])

# 这个是之前已经验证能看到桌面附近的锚点
TABLE_XYZ = np.array([-0.17198236, 1.46132815, 1.10103631], dtype=np.float32)

# 三个物体的桌面位置：围绕该锚点错开放置
# A：左侧 / 前一点
# B：右前，苹果
# C：右后，Zero123 杯子
PLACEMENTS = {
    "A": {
        "ply": a_ply,
        "target_size": 0.30,
        "offset_xy": np.array([-0.18, -0.03], dtype=np.float32),
    },
    "B": {
        "ply": b_ply,
        "target_size": 0.16,
        "offset_xy": np.array([0.12, -0.08], dtype=np.float32),
    },
    "C": {
        "ply": c_ply,
        "target_size": 0.22,
        "offset_xy": np.array([0.20, 0.08], dtype=np.float32),
    },
}

# 只扫 3 个 lift 候选：宁可稍微浮一点，也不要埋进桌子
VARIANTS = [
    {
        "tag": "final_v1_low",
        "lift": {"A": 0.06, "B": 0.04, "C": 0.06},
    },
    {
        "tag": "final_v2_mid",
        "lift": {"A": 0.09, "B": 0.06, "C": 0.09},
    },
    {
        "tag": "final_v3_high",
        "lift": {"A": 0.12, "B": 0.08, "C": 0.12},
    },
]

def read_v(path: Path):
    ply = PlyData.read(str(path))
    return ply["vertex"].data

def xyz_of(v):
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

def set_xyz(v, xyz):
    v["x"] = xyz[:, 0]
    v["y"] = xyz[:, 1]
    v["z"] = xyz[:, 2]

def normalize_bottom_xycenter(xyz):
    """
    关键修复：
    x/y 以中心归零，z 以最低点归零。
    这样 anchor + lift 才是真正把物体底部放到桌面上方。
    """
    xyz = xyz.copy().astype(np.float32)
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    xy_center = 0.5 * (mn[:2] + mx[:2])
    z_min = mn[2]

    xyz[:, 0] -= xy_center[0]
    xyz[:, 1] -= xy_center[1]
    xyz[:, 2] -= z_min
    return xyz

def transform_object(v_in, target_size, anchor_xyz, lift):
    v = np.array(v_in, dtype=v_in.dtype)
    xyz = xyz_of(v)

    xyz0 = normalize_bottom_xycenter(xyz)
    dims = xyz0.max(axis=0) - xyz0.min(axis=0)
    max_dim = float(np.max(dims))

    if max_dim <= 1e-8:
        raise RuntimeError("object has invalid dimension")

    scale = float(target_size) / max_dim
    xyz_new = xyz0 * scale
    xyz_new += anchor_xyz.astype(np.float32)
    xyz_new[:, 2] += float(lift)

    set_xyz(v, xyz_new)

    # 3DGS 的 scale_* 是 log-space，坐标缩放后要同步加 log(scale)
    names = v.dtype.names
    if all(k in names for k in ["scale_0", "scale_1", "scale_2"]):
        log_s = np.float32(np.log(scale))
        v["scale_0"] = v["scale_0"] + log_s
        v["scale_1"] = v["scale_1"] + log_s
        v["scale_2"] = v["scale_2"] + log_s

    return v, {
        "scale": scale,
        "dims_before": dims.tolist(),
        "target_size": target_size,
        "anchor_xyz": anchor_xyz.tolist(),
        "lift": lift,
    }

def main():
    print("[INFO] loading background:", bg_ply)
    bg_v = read_v(bg_ply)
    print("[INFO] background points:", len(bg_v))

    obj_vs = {}
    for key, info in PLACEMENTS.items():
        print(f"[INFO] loading {key}:", info["ply"])
        obj_vs[key] = read_v(info["ply"])
        print(f"[INFO] {key} points:", len(obj_vs[key]))

    meta_all = []

    for variant in VARIANTS:
        tag = variant["tag"]
        model_dir = out_root / tag
        out_ply = model_dir / "point_cloud/iteration_30000/point_cloud.ply"

        if model_dir.exists():
            shutil.rmtree(model_dir)

        # 复制背景模型配置，但排除 point_cloud
        subprocess.run(
            ["rsync", "-a", "--exclude=point_cloud", str(bg_model) + "/", str(model_dir) + "/"],
            check=True,
        )
        out_ply.parent.mkdir(parents=True, exist_ok=True)

        merged_list = [bg_v]
        meta = {"tag": tag, "objects": {}}

        for key, place in PLACEMENTS.items():
            anchor = TABLE_XYZ.copy()
            anchor[0] += place["offset_xy"][0]
            anchor[1] += place["offset_xy"][1]

            v_new, info = transform_object(
                obj_vs[key],
                target_size=place["target_size"],
                anchor_xyz=anchor,
                lift=variant["lift"][key],
            )

            assert bg_v.dtype == v_new.dtype, f"dtype mismatch for {key}"
            merged_list.append(v_new)
            meta["objects"][key] = info

            print(f"[{tag}] {key}: target_size={place['target_size']} lift={variant['lift'][key]} anchor={anchor.tolist()} scale={info['scale']}")

        merged = np.concatenate(merged_list, axis=0)
        PlyData([PlyElement.describe(merged, "vertex")], text=False).write(str(out_ply))

        meta["point_cloud"] = str(out_ply)
        meta["total_points"] = int(len(merged))
        meta_all.append(meta)

        print("[DONE]", tag, out_ply, "points:", len(merged))

    preview_dir.mkdir(parents=True, exist_ok=True)
    with open(preview_dir / "final_ABC_quick_meta.json", "w") as f:
        json.dump(meta_all, f, indent=2)

    print("[DONE] meta:", preview_dir / "final_ABC_quick_meta.json")

if __name__ == "__main__":
    main()
PY

# ============================================================
# Step 2. 渲染 3 个候选版本
# ============================================================
cd "$GS_ROOT"

for MODEL_DIR in "$OUT_ROOT"/*; do
  [ -d "$MODEL_DIR" ] || continue
  TAG=$(basename "$MODEL_DIR")

  echo
  echo "========== RENDER $TAG on GPU1 =========="

  python render.py \
    -m "$MODEL_DIR" \
    --iteration 30000 \
    --skip_train \
    2>&1 | tee "$LOG_DIR/render_${TAG}.log"
done

# ============================================================
# Step 3. 生成预览总览图
# ============================================================
cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

project = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
out_root = project / "outputs/fusion_scene/final_ABC_quick"
preview_dir = project / "final_assets/fusion_final_quick"
preview_dir.mkdir(parents=True, exist_ok=True)

models = sorted([p for p in out_root.iterdir() if p.is_dir()])

# 固定取 00012 视角，同时每个版本再取几张不同视角
tiles = []

for model in models:
    render_dir = model / "test/ours_30000/renders"
    imgs = sorted(render_dir.glob("*.png"))
    if not imgs:
        print("[WARN] no renders:", model)
        continue

    # 优先 00012，如果没有就取中间
    fixed = render_dir / "00012.png"
    pick = fixed if fixed.exists() else imgs[len(imgs)//2]

    im = Image.open(pick).convert("RGB")
    im = ImageOps.contain(im, (420, 280))

    canvas = Image.new("RGB", (440, 330), "white")
    canvas.paste(im, ((440 - im.width)//2, 10))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 292), model.name, fill=(0, 0, 0))
    draw.text((10, 310), pick.name, fill=(0, 0, 0))
    tiles.append(canvas)

cols = 3
rows = math.ceil(len(tiles) / cols)
sheet = Image.new("RGB", (cols * 440, rows * 330), "white")

for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * 440, (i // cols) * 330))

out = preview_dir / "final_ABC_quick_fixed00012_overview.jpg"
sheet.save(out, quality=95)
print("[DONE]", out)

# 额外生成每个版本的多视角小图，方便检查物体有没有从别的角度可见
for model in models:
    render_dir = model / "test/ours_30000/renders"
    imgs = sorted(render_dir.glob("*.png"))
    if not imgs:
        continue

    picks = []
    if len(imgs) <= 9:
        picks = imgs
    else:
        idxs = [0, 2, 4, 6, 8, 10, 12, 14, 16]
        picks = [imgs[i] for i in idxs if i < len(imgs)]

    thumbs = []
    for p in picks:
        im = Image.open(p).convert("RGB")
        im = ImageOps.contain(im, (300, 200))
        c = Image.new("RGB", (320, 240), "white")
        c.paste(im, ((320 - im.width)//2, 10))
        d = ImageDraw.Draw(c)
        d.text((10, 215), p.name, fill=(0, 0, 0))
        thumbs.append(c)

    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 320, rows * 240), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 320, (i // cols) * 240))

    out = preview_dir / f"{model.name}_multiview_preview.jpg"
    sheet.save(out, quality=95)
    print("[DONE]", out)
PY

echo
echo "=================================================="
echo "[DONE] Quick final ABC fusion finished."
echo "Open:"
echo "$PREVIEW_DIR/final_ABC_quick_fixed00012_overview.jpg"
echo "$PREVIEW_DIR/final_v1_low_multiview_preview.jpg"
echo "$PREVIEW_DIR/final_v2_mid_multiview_preview.jpg"
echo "$PREVIEW_DIR/final_v3_high_multiview_preview.jpg"
echo "=================================================="
