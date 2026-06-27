#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
FIG_DIR="$PROJECT_ROOT/figures"
mkdir -p "$FIG_DIR"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import cv2

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    ("6000 steps",  ROOT / "outputs/object_B_text3d/dreamfusion_sd/red_apple_sd_seed42_6000_gpu0@20260621-040102/save/it6000-test.mp4"),
    ("10000 steps", ROOT / "outputs/object_B_text3d/dreamfusion_sd/red_apple_sd_seed42_6000_gpu0@20260621-050420/save/it10000-test.mp4"),
    ("15000 steps", ROOT / "outputs/object_B_text3d/dreamfusion_sd/red_apple_sd_seed42_6000_gpu0@20260621-060225/save/it15000-test.mp4"),
]

OUT_DIR = FIG_DIR / "p1_3_object_B_3x3_tmp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_middle_frame(video_path: Path, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = max(0, int(n * 0.35))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame).save(out_path)
    return out_path

# 先抽帧
frame_paths = []
for step, video in VIDEOS:
    out = OUT_DIR / f"{step.replace(' ', '_')}.png"
    extract_middle_frame(video, out)
    frame_paths.append((step, out))

# 从每一帧裁成 3 块
# 默认按宽度三等分；如果边界稍有偏差，后面可以再手动改
cropped = []
for step, path in frame_paths:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    w1 = w // 3
    parts = [
        im.crop((0, 0, w1, h)),
        im.crop((w1, 0, 2 * w1, h)),
        im.crop((2 * w1, 0, w, h)),
    ]
    cropped.append((step, parts))

# 画 3x3 大图
cell_w = 340
cell_h = 250
left_margin = 140
top_margin = 70
gap_x = 24
gap_y = 28

col_titles = ["RGB view", "Normal view", "Opacity / mask"]

canvas_w = left_margin + 3 * cell_w + 2 * gap_x + 40
canvas_h = top_margin + 3 * cell_h + 2 * gap_y + 60
canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

# 列标题
for j, title in enumerate(col_titles):
    x = left_margin + j * (cell_w + gap_x) + 10
    draw.text((x, 20), title, fill=(0, 0, 0))

# 行标题 + 图片
for i, (step, parts) in enumerate(cropped):
    y0 = top_margin + i * (cell_h + gap_y)
    draw.text((20, y0 + cell_h // 2 - 8), step, fill=(0, 0, 0))
    for j, part in enumerate(parts):
        panel = ImageOps.contain(part, (cell_w, cell_h))
        x0 = left_margin + j * (cell_w + gap_x)
        y_img = y0 + (cell_h - panel.height) // 2
        x_img = x0 + (cell_w - panel.width) // 2
        canvas.paste(panel, (x_img, y_img))

out_path = FIG_DIR / "p1_3_object_B_red_apple_steps_3x3.png"
canvas.save(out_path, quality=95)
print("[DONE]", out_path)
PY

echo "[DONE] saved to: $FIG_DIR/p1_3_object_B_red_apple_steps_3x3.png"
