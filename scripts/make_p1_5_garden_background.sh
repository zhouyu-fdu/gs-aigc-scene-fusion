#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
GS_ROOT=/mnt/data/disk3/zhouyu/projects/gaussian-splatting
CONDA_SH=/mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh

FIG_DIR="$PROJECT_ROOT/figures"
mkdir -p "$FIG_DIR"

BG_MODEL="$PROJECT_ROOT/outputs/background_3dgs/garden_30000"
RENDER_DIR="$BG_MODEL/test/ours_30000/renders"

echo "=================================================="
echo "[INFO] Make P1-5 garden background 3DGS figure"
echo "=================================================="

# 如果还没有 background test renders，则补渲染一遍
if [ ! -d "$RENDER_DIR" ] || [ "$(find "$RENDER_DIR" -maxdepth 1 -name '*.png' | wc -l)" -lt 4 ]; then
  echo "[INFO] Background renders not found. Rendering garden background..."
  set +u
  source "$CONDA_SH"
  conda activate gs_splatting
  set -u

  cd "$GS_ROOT"
  python render.py \
    -m "$BG_MODEL" \
    --iteration 30000 \
    --skip_train \
    2>&1 | tee "$PROJECT_ROOT/logs/p1_5_render_garden_background.log"
fi

cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

render_dir = ROOT / "outputs/background_3dgs/garden_30000/test/ours_30000/renders"
imgs = sorted(render_dir.glob("*.png"))

print("[INFO] render dir:", render_dir)
print("[INFO] number of renders:", len(imgs))

if len(imgs) < 4:
    raise RuntimeError(f"Need at least 4 rendered images, found {len(imgs)}")

# 优先选比较分散的 4 个视角
# test 一般 24 张，所以这些 index 能覆盖不同角度
candidate_indices = [0, 6, 12, 18]
picked = []
for idx in candidate_indices:
    if idx < len(imgs):
        picked.append(imgs[idx])

# 如果图片数量不足，则均匀采样
if len(picked) < 4:
    picked = [imgs[int(i * (len(imgs)-1) / 3)] for i in range(4)]

titles = [
    "(a) Garden view 1",
    "(b) Garden view 2",
    "(c) Garden view 3",
    "(d) Garden view 4",
]

cell_w = 640
cell_h = 430
title_h = 42

panels = []
for p, title in zip(picked, titles):
    im = Image.open(p).convert("RGB")
    im = ImageOps.contain(im, (cell_w, cell_h - title_h - 8))

    canvas = Image.new("RGB", (cell_w, cell_h), "white")
    canvas.paste(im, ((cell_w - im.width)//2, title_h + (cell_h - title_h - im.height)//2))

    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), f"{title} | {p.name}", fill=(0, 0, 0))
    panels.append(canvas)

sheet = Image.new("RGB", (cell_w * 2, cell_h * 2 + 56), "white")

for i, panel in enumerate(panels):
    x = (i % 2) * cell_w
    y = (i // 2) * cell_h
    sheet.paste(panel, (x, y))

draw = ImageDraw.Draw(sheet)
metric_text = "Background 3DGS metrics: PSNR = 27.34 dB, SSIM = 0.8568, LPIPS = 0.1219"
draw.text((20, cell_h * 2 + 18), metric_text, fill=(0, 0, 0))

out = FIG_DIR / "p1_5_garden_background_3dgs.png"
sheet.save(out, quality=95)

print("[DONE]", out)
print("[INFO] picked images:")
for p in picked:
    print("  ", p)
PY

echo
echo "[DONE] P1-5 figure saved to:"
echo "$FIG_DIR/p1_5_garden_background_3dgs.png"
