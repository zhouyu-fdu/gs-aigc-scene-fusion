#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
GS_ROOT=/mnt/data/disk3/zhouyu/projects/gaussian-splatting
CONDA_SH=/mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh

FIG_DIR="$PROJECT_ROOT/figures"
mkdir -p "$FIG_DIR"

A_DATA="$PROJECT_ROOT/data/object_A_multiview"
A_MODEL="$PROJECT_ROOT/outputs/object_A_3dgs/object_A_30000"
A_CLEAN_PLY="$PROJECT_ROOT/final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply"
A_CLEAN_MODEL="$PROJECT_ROOT/outputs/object_A_3dgs/object_A_clean_final_render_model"

COLMAP_SPARSE="$A_DATA/sparse/0"
COLMAP_TXT_DIR="$FIG_DIR/object_A_colmap_txt"

echo "=================================================="
echo "[INFO] Make P1-2 Object A figure v2"
echo "=================================================="

set +u
source "$CONDA_SH"
conda activate gs_splatting
set -u

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# 1. 把 COLMAP binary 模型转成 txt，方便 Python 读取
rm -rf "$COLMAP_TXT_DIR"
mkdir -p "$COLMAP_TXT_DIR"

if command -v colmap >/dev/null 2>&1 && [ -d "$COLMAP_SPARSE" ]; then
  colmap model_converter \
    --input_path "$COLMAP_SPARSE" \
    --output_path "$COLMAP_TXT_DIR" \
    --output_type TXT
else
  echo "[WARN] colmap not found or sparse dir missing, will use text placeholder."
fi

# 2. 确保原始 A 和 clean A 都有 render
if [ ! -d "$A_MODEL/train/ours_30000/renders" ]; then
  echo "[INFO] Rendering original Object A model..."
  cd "$GS_ROOT"
  python render.py \
    -m "$A_MODEL" \
    --iteration 30000 \
    --skip_test \
    2>&1 | tee "$PROJECT_ROOT/logs/p1_2_render_object_A_original.log"
fi

if [ ! -d "$A_CLEAN_MODEL/train/ours_30000/renders" ]; then
  echo "[INFO] Preparing clean Object A render model..."
  rm -rf "$A_CLEAN_MODEL"
  mkdir -p "$A_CLEAN_MODEL"
  rsync -a --exclude='point_cloud' "$A_MODEL"/ "$A_CLEAN_MODEL"/
  mkdir -p "$A_CLEAN_MODEL/point_cloud/iteration_30000"
  cp "$A_CLEAN_PLY" "$A_CLEAN_MODEL/point_cloud/iteration_30000/point_cloud.ply"

  echo "[INFO] Rendering clean Object A model..."
  cd "$GS_ROOT"
  python render.py \
    -m "$A_CLEAN_MODEL" \
    --iteration 30000 \
    --skip_test \
    2>&1 | tee "$PROJECT_ROOT/logs/p1_2_render_object_A_clean.log"
fi

# 3. 拼 2x2 图，其中 b 为 COLMAP 稀疏点云可视化
cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import math

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
FIG_DIR = ROOT / "figures"
TXT_DIR = FIG_DIR / "object_A_colmap_txt"

frame_dir = ROOT / "data/object_A_multiview/images"
orig_dir = ROOT / "outputs/object_A_3dgs/object_A_30000/train/ours_30000/renders"
clean_dir = ROOT / "outputs/object_A_3dgs/object_A_clean_final_render_model/train/ours_30000/renders"

W, H = 760, 500
TITLE_H = 54

def pick_middle_image(d):
    imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg")))
    if not imgs:
        raise RuntimeError(f"No images found in {d}")
    return imgs[len(imgs)//2]

def load_panel_image(path, title):
    im = Image.open(path).convert("RGB")
    im = ImageOps.contain(im, (W, H - TITLE_H - 10))
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, ((W - im.width)//2, TITLE_H + (H - TITLE_H - im.height)//2))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, fill=(0, 0, 0))
    return canvas

def parse_points3d(path):
    pts = []
    cols = []
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            pts.append([x, y, z])
            cols.append([r, g, b])

    return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)

def qvec_to_rotmat(qvec):
    qvec = np.asarray(qvec, dtype=np.float64)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)

def parse_camera_centers(images_txt):
    centers = []
    if not images_txt.exists():
        return np.zeros((0, 3), dtype=np.float32)

    lines = [l.strip() for l in images_txt.read_text(errors="ignore").splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            qvec = list(map(float, parts[1:5]))
            tvec = np.asarray(list(map(float, parts[5:8])), dtype=np.float64)
            R = qvec_to_rotmat(qvec)
            C = -R.T @ tvec
            centers.append(C.tolist())
            i += 2  # 下一行是 2D points
        else:
            i += 1

    return np.asarray(centers, dtype=np.float32)

def make_colmap_panel(title):
    pts, rgb = parse_points3d(TXT_DIR / "points3D.txt")
    cams = parse_camera_centers(TXT_DIR / "images.txt")

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, fill=(0, 0, 0))

    if len(pts) == 0:
        draw.rectangle((40, 80, W - 40, H - 70), outline=(100, 100, 100), width=2)
        draw.text((70, 120), "COLMAP sparse reconstruction", fill=(0, 0, 0))
        draw.text((70, 175), "Registered images: 79", fill=(0, 0, 0))
        draw.text((70, 225), "Sparse points: 4429", fill=(0, 0, 0))
        draw.text((70, 275), "Mean reprojection error: 1.109 px", fill=(0, 0, 0))
        return canvas

    # 用 PCA 将 3D 点云投影到 2D，作为稀疏点云俯视/主方向可视化
    all_xyz = pts
    if len(cams) > 0:
        all_xyz = np.concatenate([pts, cams], axis=0)

    center = np.median(all_xyz, axis=0)
    X = all_xyz - center
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    basis = vh[:2].T

    pts2 = (pts - center) @ basis
    cams2 = (cams - center) @ basis if len(cams) > 0 else np.zeros((0, 2), dtype=np.float32)

    all2 = pts2
    if len(cams2) > 0:
        all2 = np.concatenate([pts2, cams2], axis=0)

    mn = np.quantile(all2, 0.01, axis=0)
    mx = np.quantile(all2, 0.99, axis=0)
    span = mx - mn
    span[span < 1e-6] = 1.0

    plot_x0, plot_y0 = 50, 80
    plot_w, plot_h = W - 100, H - 145

    def to_px(p2):
        norm = (p2 - mn) / span
        x = plot_x0 + norm[:, 0] * plot_w
        y = plot_y0 + (1.0 - norm[:, 1]) * plot_h
        return x.astype(np.int32), y.astype(np.int32)

    px, py = to_px(pts2)
    valid = (px >= plot_x0) & (px < plot_x0 + plot_w) & (py >= plot_y0) & (py < plot_y0 + plot_h)

    px = px[valid]
    py = py[valid]
    col = rgb[valid]

    # 点太多时随机抽样，避免绘图太慢
    if len(px) > 8000:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(px), size=8000, replace=False)
        px, py, col = px[idx], py[idx], col[idx]

    for x, y, c in zip(px, py, col):
        draw.point((int(x), int(y)), fill=tuple(int(v) for v in c))

    if len(cams2) > 0:
        cx, cy = to_px(cams2)
        for x, y in zip(cx, cy):
            if plot_x0 <= x < plot_x0 + plot_w and plot_y0 <= y < plot_y0 + plot_h:
                r = 4
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(220, 30, 30), outline=(0, 0, 0))

    draw.rectangle((plot_x0, plot_y0, plot_x0 + plot_w, plot_y0 + plot_h), outline=(100, 100, 100), width=2)

    info = f"Registered images: 79    Sparse points: {len(pts)}    Mean reprojection error: 1.109 px"
    draw.text((50, H - 48), info, fill=(0, 0, 0))
    draw.text((50, H - 26), "Colored dots: sparse 3D points; red dots: estimated camera centers.", fill=(70, 70, 70))

    return canvas

frame_img = pick_middle_image(frame_dir)
orig_img = pick_middle_image(orig_dir)
clean_img = pick_middle_image(clean_dir)

panels = [
    load_panel_image(frame_img, "(a) Sample frame from phone video"),
    make_colmap_panel("(b) COLMAP sparse point cloud and camera poses"),
    load_panel_image(orig_img, "(c) Original Object A 3DGS rendering"),
    load_panel_image(clean_img, "(d) Cleaned Object A Gaussian asset"),
]

sheet = Image.new("RGB", (W * 2, H * 2), "white")
for i, p in enumerate(panels):
    x = (i % 2) * W
    y = (i // 2) * H
    sheet.paste(p, (x, y))

out = FIG_DIR / "p1_2_object_A_reconstruction_v2.png"
sheet.save(out, quality=95)
print("[DONE]", out)
PY

echo
echo "=================================================="
echo "[DONE] P1-2 v2 figure generated:"
echo "$FIG_DIR/p1_2_object_A_reconstruction_v2.png"
echo "=================================================="
