#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
GS_ROOT=/mnt/data/disk3/zhouyu/projects/gaussian-splatting
CONDA_SH=/mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=1

BG_MODEL="$PROJECT_ROOT/outputs/background_3dgs/garden_30000"
BG_PLY="$BG_MODEL/point_cloud/iteration_30000/point_cloud.ply"

A_RAW="$PROJECT_ROOT/final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply"
A_TAG="${A_TAG:-center_s026}"
A_CAND="$PROJECT_ROOT/outputs/fusion_scene/A_local_search_known_anchor/$A_TAG/point_cloud/iteration_30000/point_cloud.ply"

B_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_B/object_B_gaussian_asset.ply"
C_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_C/object_C_gaussian_asset.ply"

OUT_ROOT="$PROJECT_ROOT/outputs/fusion_scene/final_ABC_repair_v2"
PREVIEW_DIR="$PROJECT_ROOT/final_assets/fusion_repair_v2"
LOG_DIR="$PROJECT_ROOT/logs/final_ABC_repair_v2"

mkdir -p "$OUT_ROOT" "$PREVIEW_DIR" "$LOG_DIR"

set +u
source "$CONDA_SH"
conda activate gs_splatting
set -u

echo "=================================================="
echo "[INFO] Fast repair ABC v2"
echo "[INFO] A_TAG=$A_TAG"
echo "[INFO] A_CAND=$A_CAND"
echo "[INFO] GPU: physical GPU1"
echo "=================================================="

export PROJECT_ROOT BG_MODEL BG_PLY A_RAW A_CAND B_PLY C_PLY OUT_ROOT PREVIEW_DIR

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
a_raw = Path(os.environ["A_RAW"])
a_cand = Path(os.environ["A_CAND"])
b_ply = Path(os.environ["B_PLY"])
c_ply = Path(os.environ["C_PLY"])
out_root = Path(os.environ["OUT_ROOT"])
preview_dir = Path(os.environ["PREVIEW_DIR"])

TABLE_XYZ = np.array([-0.17198236, 1.46132815, 1.10103631], dtype=np.float32)

# B/C 的桌面位置：尽量远离花瓶，三者分开
B_ANCHOR = TABLE_XYZ + np.array([0.30, -0.14, 0.0], dtype=np.float32)
C_ANCHOR = TABLE_XYZ + np.array([0.42,  0.14, 0.0], dtype=np.float32)

# 尺寸：比上一版更大一点，便于看清
B_TARGET_SIZE = 0.20
C_TARGET_SIZE = 0.34

# 高度：宁可轻微浮一点，也不要埋进桌子
B_LIFT = 0.13
C_LIFT = 0.18

# A 从可见候选里抽出来后，只做局部放大和抬高
A_SCALE_UP = 1.35
A_EXTRA_LIFT = 0.22
A_EXTRA_XY = np.array([-0.05, 0.00], dtype=np.float32)

C_ROT_VARIANTS = {
    "c_none": "none",
    "c_rx90": "rx90",
    "c_rxm90": "rxm90",
    "c_flipz": "flipz",
}

def read_v(path):
    return PlyData.read(str(path))["vertex"].data

def xyz_of(v):
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

def set_xyz(v, xyz):
    v["x"] = xyz[:, 0]
    v["y"] = xyz[:, 1]
    v["z"] = xyz[:, 2]

def add_log_scale(v, s, extra_sharp=0.0):
    names = v.dtype.names
    if all(k in names for k in ["scale_0", "scale_1", "scale_2"]):
        delta = np.float32(np.log(s) + extra_sharp)
        v["scale_0"] = v["scale_0"] + delta
        v["scale_1"] = v["scale_1"] + delta
        v["scale_2"] = v["scale_2"] + delta

def boost_opacity(v, bias):
    if "opacity" in v.dtype.names:
        v["opacity"] = np.clip(v["opacity"] + np.float32(bias), -8.0, 8.0)

def rotate_xyz(xyz, mode):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if mode == "none":
        out = np.stack([x, y, z], axis=1)
    elif mode == "rx90":
        # 绕 X 轴 +90°：常用于 Y-up -> Z-up 尝试
        out = np.stack([x, -z, y], axis=1)
    elif mode == "rxm90":
        # 绕 X 轴 -90°
        out = np.stack([x, z, -y], axis=1)
    elif mode == "flipz":
        # 上下翻转候选
        out = np.stack([x, y, -z], axis=1)
    else:
        raise ValueError(mode)
    return out.astype(np.float32)

def normalize_bottom(v, rot_mode="none"):
    """
    先把物体转到候选姿态，再做底部对齐：
    x/y 居中，z 的 2% 分位数作为底部。
    """
    v2 = np.array(v, dtype=v.dtype)
    xyz = xyz_of(v2)

    # 先居中，避免旋转时绕远处转
    center0 = np.median(xyz, axis=0)
    xyz = xyz - center0

    xyz = rotate_xyz(xyz, rot_mode)

    # 去离群点估计主体
    med = np.median(xyz, axis=0)
    dist = np.linalg.norm(xyz - med, axis=1)
    mask = dist < np.quantile(dist, 0.995)
    core = xyz[mask] if mask.sum() > 200 else xyz

    xy_center = np.median(core[:, :2], axis=0)
    z_bottom = np.quantile(core[:, 2], 0.02)

    xyz[:, 0] -= xy_center[0]
    xyz[:, 1] -= xy_center[1]
    xyz[:, 2] -= z_bottom

    set_xyz(v2, xyz)
    return v2

def place_asset(v, anchor, target_size, lift, rot_mode="none", opacity_bias=0.0, extra_sharp=-0.35):
    v2 = normalize_bottom(v, rot_mode=rot_mode)
    xyz = xyz_of(v2)

    dims = xyz.max(axis=0) - xyz.min(axis=0)
    s = float(target_size) / float(np.max(dims))

    xyz_new = xyz * s
    xyz_new += anchor.astype(np.float32)
    xyz_new[:, 2] += float(lift)

    set_xyz(v2, xyz_new)
    add_log_scale(v2, s, extra_sharp=extra_sharp)
    boost_opacity(v2, opacity_bias)
    return v2, {"scale": s, "dims": dims.tolist(), "anchor": anchor.tolist(), "lift": lift, "rot_mode": rot_mode}

def extract_and_fix_A(bg_v, a_raw_v, a_cand_v):
    """
    A 候选 PLY 是 [background + A] 拼出来的，所以直接取最后 len(A_raw) 个点。
    然后以当前 A 的底部中心为支点做放大，并整体抬高。
    """
    nA = len(a_raw_v)
    if len(a_cand_v) < len(bg_v) + nA:
        raise RuntimeError("A candidate point count seems wrong.")

    a = np.array(a_cand_v[-nA:], dtype=a_cand_v.dtype)
    xyz = xyz_of(a)

    # robust core
    med = np.median(xyz, axis=0)
    dist = np.linalg.norm(xyz - med, axis=1)
    mask = dist < np.quantile(dist, 0.995)
    core = xyz[mask] if mask.sum() > 200 else xyz

    mn = core.min(axis=0)
    mx = core.max(axis=0)
    pivot = np.array([
        0.5 * (mn[0] + mx[0]),
        0.5 * (mn[1] + mx[1]),
        np.quantile(core[:, 2], 0.02)
    ], dtype=np.float32)

    xyz_new = (xyz - pivot) * A_SCALE_UP + pivot
    xyz_new[:, 0] += A_EXTRA_XY[0]
    xyz_new[:, 1] += A_EXTRA_XY[1]
    xyz_new[:, 2] += A_EXTRA_LIFT

    set_xyz(a, xyz_new)
    add_log_scale(a, A_SCALE_UP, extra_sharp=-0.20)
    boost_opacity(a, 0.15)

    return a, {
        "source": str(a_cand),
        "scale_up": A_SCALE_UP,
        "extra_lift": A_EXTRA_LIFT,
        "extra_xy": A_EXTRA_XY.tolist(),
        "pivot": pivot.tolist(),
    }

def main():
    print("[INFO] load BG:", bg_ply)
    bg_v = read_v(bg_ply)
    print("[INFO] BG points:", len(bg_v))

    print("[INFO] load A raw:", a_raw)
    a_raw_v = read_v(a_raw)
    print("[INFO] A raw points:", len(a_raw_v))

    print("[INFO] load A candidate:", a_cand)
    a_cand_v = read_v(a_cand)
    print("[INFO] A cand points:", len(a_cand_v))

    print("[INFO] load B:", b_ply)
    b_v = read_v(b_ply)
    print("[INFO] B points:", len(b_v))

    print("[INFO] load C:", c_ply)
    c_v = read_v(c_ply)
    print("[INFO] C points:", len(c_v))

    meta_all = []

    for tag, c_rot in C_ROT_VARIANTS.items():
        model_name = f"repair_v2_{tag}"
        model_dir = out_root / model_name
        out_ply = model_dir / "point_cloud/iteration_30000/point_cloud.ply"

        if model_dir.exists():
            shutil.rmtree(model_dir)

        subprocess.run(
            ["rsync", "-a", "--exclude=point_cloud", str(bg_model) + "/", str(model_dir) + "/"],
            check=True
        )
        out_ply.parent.mkdir(parents=True, exist_ok=True)

        a_fixed, a_meta = extract_and_fix_A(bg_v, a_raw_v, a_cand_v)

        b_fixed, b_meta = place_asset(
            b_v,
            anchor=B_ANCHOR,
            target_size=B_TARGET_SIZE,
            lift=B_LIFT,
            rot_mode="none",
            opacity_bias=0.25,
            extra_sharp=-0.45,
        )

        c_fixed, c_meta = place_asset(
            c_v,
            anchor=C_ANCHOR,
            target_size=C_TARGET_SIZE,
            lift=C_LIFT,
            rot_mode=c_rot,
            opacity_bias=0.25,
            extra_sharp=-0.45,
        )

        assert bg_v.dtype == a_fixed.dtype == b_fixed.dtype == c_fixed.dtype

        merged = np.concatenate([bg_v, a_fixed, b_fixed, c_fixed], axis=0)
        PlyData([PlyElement.describe(merged, "vertex")], text=False).write(str(out_ply))

        meta = {
            "model": model_name,
            "point_cloud": str(out_ply),
            "points": int(len(merged)),
            "A": a_meta,
            "B": b_meta,
            "C": c_meta,
        }
        meta_all.append(meta)

        print("[DONE]", model_name, "points:", len(merged), "C_rot:", c_rot)

    preview_dir.mkdir(parents=True, exist_ok=True)
    with open(preview_dir / "repair_v2_meta.json", "w") as f:
        json.dump(meta_all, f, indent=2)

    print("[DONE] meta:", preview_dir / "repair_v2_meta.json")

if __name__ == "__main__":
    main()
PY

# 渲染 4 个候选，只渲染 test，节省时间
cd "$GS_ROOT"

for MODEL_DIR in "$OUT_ROOT"/*; do
  [ -d "$MODEL_DIR" ] || continue
  TAG=$(basename "$MODEL_DIR")

  echo
  echo "========== RENDER $TAG =========="
  python render.py \
    -m "$MODEL_DIR" \
    --iteration 30000 \
    --skip_train \
    2>&1 | tee "$LOG_DIR/render_${TAG}.log"
done

# 生成总览图
cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

project = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
out_root = project / "outputs/fusion_scene/final_ABC_repair_v2"
preview_dir = project / "final_assets/fusion_repair_v2"
preview_dir.mkdir(parents=True, exist_ok=True)

models = sorted([p for p in out_root.iterdir() if p.is_dir()])

# 每个候选取 9 张 test 图
for model in models:
    render_dir = model / "test/ours_30000/renders"
    imgs = sorted(render_dir.glob("*.png"))
    if not imgs:
        continue

    idxs = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    picks = [imgs[i] for i in idxs if i < len(imgs)]

    tiles = []
    for p in picks:
        im = Image.open(p).convert("RGB")
        im = ImageOps.contain(im, (320, 220))
        canvas = Image.new("RGB", (340, 260), "white")
        canvas.paste(im, ((340 - im.width)//2, 10))
        d = ImageDraw.Draw(canvas)
        d.text((10, 230), p.name, fill=(0, 0, 0))
        tiles.append(canvas)

    cols = 3
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 340, rows * 260), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 340, (i // cols) * 260))

    out = preview_dir / f"{model.name}_multiview.jpg"
    sheet.save(out, quality=95)
    print("[DONE]", out)

# 固定 00012 对比四个候选
tiles = []
for model in models:
    render_dir = model / "test/ours_30000/renders"
    p = render_dir / "00012.png"
    if not p.exists():
        imgs = sorted(render_dir.glob("*.png"))
        if not imgs:
            continue
        p = imgs[len(imgs)//2]

    im = Image.open(p).convert("RGB")
    im = ImageOps.contain(im, (360, 250))
    canvas = Image.new("RGB", (380, 300), "white")
    canvas.paste(im, ((380 - im.width)//2, 10))
    d = ImageDraw.Draw(canvas)
    d.text((10, 262), model.name, fill=(0, 0, 0))
    d.text((10, 280), p.name, fill=(0, 0, 0))
    tiles.append(canvas)

cols = 2
rows = math.ceil(len(tiles) / cols)
sheet = Image.new("RGB", (cols * 380, rows * 300), "white")
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 380, (i // cols) * 300))

out = preview_dir / "repair_v2_fixed00012_overview.jpg"
sheet.save(out, quality=95)
print("[DONE]", out)
PY

echo
echo "=================================================="
echo "[DONE] repair v2 finished."
echo "Open:"
echo "$PREVIEW_DIR/repair_v2_fixed00012_overview.jpg"
echo "$PREVIEW_DIR/repair_v2_c_none_multiview.jpg"
echo "$PREVIEW_DIR/repair_v2_c_rx90_multiview.jpg"
echo "$PREVIEW_DIR/repair_v2_c_rxm90_multiview.jpg"
echo "$PREVIEW_DIR/repair_v2_c_flipz_multiview.jpg"
echo "=================================================="
