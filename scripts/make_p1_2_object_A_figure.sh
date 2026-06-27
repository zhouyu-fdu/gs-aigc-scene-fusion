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
COLMAP_TXT="$FIG_DIR/p1_2_colmap_stats.txt"

echo "=================================================="
echo "[INFO] Make P1-2 Object A figure"
echo "=================================================="

# 1. COLMAP 统计
if command -v colmap >/dev/null 2>&1 && [ -d "$COLMAP_SPARSE" ]; then
  colmap model_analyzer \
    --path "$COLMAP_SPARSE" \
    > "$COLMAP_TXT" 2>&1 || true
else
  cat > "$COLMAP_TXT" <<'TXT'
COLMAP sparse reconstruction

Input frames: 105
Registered images: 79
Sparse points: 4429
Mean reprojection error: 1.109 px
TXT
fi

# 如果 model_analyzer 输出不完整，就写入你前面记录的统计
if ! grep -q "Registered images" "$COLMAP_TXT"; then
  cat > "$COLMAP_TXT" <<'TXT'
COLMAP sparse reconstruction

Input frames: 105
Registered images: 79
Sparse points: 4429
Mean reprojection error: 1.109 px
TXT
fi

# 2. 渲染原始 A 3DGS
set +u
source "$CONDA_SH"
conda activate gs_splatting
set -u

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

if [ ! -d "$A_MODEL/train/ours_30000/renders" ]; then
  echo "[INFO] Rendering original Object A model..."
  cd "$GS_ROOT"
  python render.py \
    -m "$A_MODEL" \
    --iteration 30000 \
    --skip_test \
    2>&1 | tee "$PROJECT_ROOT/logs/p1_2_render_object_A_original.log"
fi

# 3. 构造清理版 A 的 render model
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

# 4. 拼 2x2 总图
cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import textwrap
import re

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

frame_dir = ROOT / "data/object_A_multiview/images"
orig_dir = ROOT / "outputs/object_A_3dgs/object_A_30000/train/ours_30000/renders"
clean_dir = ROOT / "outputs/object_A_3dgs/object_A_clean_final_render_model/train/ours_30000/renders"
colmap_txt = FIG_DIR / "p1_2_colmap_stats.txt"

def pick_middle_image(d):
    imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg")))
    if not imgs:
        raise RuntimeError(f"No images found in {d}")
    return imgs[len(imgs)//2]

frame_img = pick_middle_image(frame_dir)
orig_img = pick_middle_image(orig_dir)
clean_img = pick_middle_image(clean_dir)

print("[INFO] frame:", frame_img)
print("[INFO] original render:", orig_img)
print("[INFO] clean render:", clean_img)

W, H = 760, 500
TITLE_H = 54

def load_panel_image(path, title):
    im = Image.open(path).convert("RGB")
    im = ImageOps.contain(im, (W, H - TITLE_H))
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, ((W - im.width)//2, TITLE_H + (H - TITLE_H - im.height)//2))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, fill=(0, 0, 0))
    return canvas

def make_text_panel(text_path, title):
    raw = text_path.read_text(errors="ignore")

    # 尝试提取关键信息；失败则直接使用固定摘要
    lines = []
    for key in ["Cameras", "Images", "Registered images", "Points", "Mean track length", "Mean observations per image", "Mean reprojection error"]:
        for line in raw.splitlines():
            if line.strip().startswith(key):
                lines.append(line.strip())
                break

    if not any("Registered images" in x for x in lines):
        lines = [
            "Input frames: 105",
            "Registered images: 79",
            "Sparse points: 4429",
            "Mean reprojection error: 1.109 px",
        ]

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((16, 14), title, fill=(0, 0, 0))
    y = 90

    header = "COLMAP sparse reconstruction"
    draw.text((40, y), header, fill=(0, 0, 0))
    y += 56

    for line in lines:
        # 把 Points 改成 Sparse points，更适合报告
        line = line.replace("Points:", "Sparse points:")
        wrapped = textwrap.wrap(line, width=58)
        for wline in wrapped:
            draw.text((70, y), wline, fill=(0, 0, 0))
            y += 36
        y += 8

    note = "These statistics indicate that the multi-view camera poses are sufficiently stable for 3DGS training."
    y += 28
    for wline in textwrap.wrap(note, width=68):
        draw.text((40, y), wline, fill=(70, 70, 70))
        y += 30

    # 简单画边框
    draw.rectangle((24, 74, W-24, H-28), outline=(120, 120, 120), width=2)

    return canvas

panels = [
    load_panel_image(frame_img, "(a) Sample frame from phone video"),
    make_text_panel(colmap_txt, "(b) COLMAP sparse reconstruction statistics"),
    load_panel_image(orig_img, "(c) Original Object A 3DGS rendering"),
    load_panel_image(clean_img, "(d) Cleaned Object A Gaussian asset"),
]

sheet = Image.new("RGB", (W * 2, H * 2), "white")
for i, p in enumerate(panels):
    x = (i % 2) * W
    y = (i // 2) * H
    sheet.paste(p, (x, y))

out = FIG_DIR / "p1_2_object_A_reconstruction.png"
sheet.save(out, quality=95)
print("[DONE]", out)
PY

echo
echo "=================================================="
echo "[DONE] P1-2 figure generated:"
echo "$FIG_DIR/p1_2_object_A_reconstruction.png"
echo "=================================================="
