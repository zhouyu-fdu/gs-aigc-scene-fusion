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

set +u
source "$CONDA_SH"
conda activate gs_splatting
set -u

# 1) 如果原始 A 还没渲染，就补一遍
if [ ! -d "$A_MODEL/train/ours_30000/renders" ]; then
  echo "[INFO] Rendering original Object A ..."
  cd "$GS_ROOT"
  python render.py \
    -m "$A_MODEL" \
    --iteration 30000 \
    --skip_test \
    2>&1 | tee "$PROJECT_ROOT/logs/p1_2_render_object_A_original.log"
fi

# 2) 如果 clean A 还没渲染，就补一遍
if [ ! -d "$A_CLEAN_MODEL/train/ours_30000/renders" ]; then
  echo "[INFO] Preparing clean Object A render model..."
  rm -rf "$A_CLEAN_MODEL"
  mkdir -p "$A_CLEAN_MODEL"
  rsync -a --exclude='point_cloud' "$A_MODEL"/ "$A_CLEAN_MODEL"/
  mkdir -p "$A_CLEAN_MODEL/point_cloud/iteration_30000"
  cp "$A_CLEAN_PLY" "$A_CLEAN_MODEL/point_cloud/iteration_30000/point_cloud.ply"

  echo "[INFO] Rendering clean Object A ..."
  cd "$GS_ROOT"
  python render.py \
    -m "$A_CLEAN_MODEL" \
    --iteration 30000 \
    --skip_test \
    2>&1 | tee "$PROJECT_ROOT/logs/p1_2_render_object_A_clean.log"
fi

# 3) 生成三图并排大图
cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
FIG_DIR = ROOT / "figures"

frame_dir = ROOT / "data/object_A_multiview/images"
orig_dir  = ROOT / "outputs/object_A_3dgs/object_A_30000/train/ours_30000/renders"
clean_dir = ROOT / "outputs/object_A_3dgs/object_A_clean_final_render_model/train/ours_30000/renders"

W, H = 560, 520
TITLE_H = 54

def pick_middle_image(d):
    imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg")))
    if not imgs:
        raise RuntimeError(f"No images found in {d}")
    return imgs[len(imgs)//2]

def make_panel(img_path, title):
    im = Image.open(img_path).convert("RGB")
    im = ImageOps.contain(im, (W, H - TITLE_H - 12))
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, ((W - im.width)//2, TITLE_H + (H - TITLE_H - im.height)//2))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, fill=(0, 0, 0))
    return canvas

frame_img = pick_middle_image(frame_dir)
orig_img  = pick_middle_image(orig_dir)
clean_img = pick_middle_image(clean_dir)

panels = [
    make_panel(frame_img, "(a) Sample frame from phone video"),
    make_panel(orig_img,  "(b) Original Object A 3DGS rendering"),
    make_panel(clean_img, "(c) Cleaned Object A Gaussian asset"),
]

sheet = Image.new("RGB", (W * 3, H), "white")
for i, p in enumerate(panels):
    sheet.paste(p, (i * W, 0))

out = FIG_DIR / "p1_2_object_A_three_panel.png"
sheet.save(out, quality=95)
print("[DONE]", out)
PY

echo
echo "[DONE] Figure saved to:"
echo "$FIG_DIR/p1_2_object_A_three_panel.png"
