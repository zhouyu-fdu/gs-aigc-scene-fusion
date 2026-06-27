#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
FIG_DIR="$PROJECT_ROOT/figures"
mkdir -p "$FIG_DIR"

cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import cv2
import numpy as np

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
FIG_DIR = ROOT / "figures"
TMP_DIR = FIG_DIR / "p1_4_object_C_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

INPUT_IMG = ROOT / "data/object_C_image/processed/object_C_cup_rgba_512.png"

VIDEOS = [
    {
        "label": "3000 steps",
        "path": ROOT / "final_assets/object_C_image3d/object_C_cup_bear_a6000_3000_128_final.mp4",
    },
    {
        "label": "5000 steps",
        "path": ROOT / "final_assets/object_C_image3d/object_C_cup_bear_a6000_5000_128_final.mp4",
    },
]

# 如果 final_assets 里的 mp4 不存在，则回退到 outputs 目录
FALLBACKS = {
    "3000 steps": ROOT / "outputs/object_C_image3d/stable_zero123/cup_bear_a6000_3000_128_slurm@20260622-032042/save/it3000-test.mp4",
    "5000 steps": ROOT / "outputs/object_C_image3d/stable_zero123/cup_bear_a6000_continue_5000_128@20260622-043602/save/it5000-test.mp4",
}

def resolve_video(label, path):
    if path.exists():
        return path
    fb = FALLBACKS[label]
    if fb.exists():
        return fb
    raise FileNotFoundError(f"Cannot find video for {label}: {path} or {fb}")

def extract_frame(video_path: Path, out_path: Path, ratio=0.35):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = max(0, int(n * ratio))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    im.save(out_path)
    return out_path

def crop_columns(im: Image.Image, ncols=4):
    """
    stable-Zero123 / threestudio 的 test video 通常会把多个结果横向拼在一帧里。
    这里按宽度等分为 4 列。如果实际只有 3 列，也能通过后面的 contain 缩放显示。
    """
    w, h = im.size

    # 有些视频是 3 列，有些是 4 列。这里根据宽高比自动猜一下。
    aspect = w / max(h, 1)
    if aspect > 3.2:
        ncols = 4
    else:
        ncols = 3

    parts = []
    for i in range(ncols):
        x0 = int(i * w / ncols)
        x1 = int((i + 1) * w / ncols)
        parts.append(im.crop((x0, 0, x1, h)))
    return parts

def make_image_panel(img, size, title):
    W, H = size
    title_h = 34
    canvas = Image.new("RGB", (W, H), "white")

    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    im = ImageOps.contain(img, (W - 10, H - title_h - 10))
    canvas.paste(im, ((W - im.width)//2, title_h + (H - title_h - im.height)//2))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0))
    return canvas

# -----------------------------
# 读取输入图
# -----------------------------
if not INPUT_IMG.exists():
    raise FileNotFoundError(f"Missing input image: {INPUT_IMG}")

input_img = Image.open(INPUT_IMG)

# -----------------------------
# 抽取 3000 / 5000 step 视频帧
# -----------------------------
step_rows = []
max_cols = 0

for item in VIDEOS:
    label = item["label"]
    video_path = resolve_video(label, item["path"])
    frame_path = TMP_DIR / f"object_C_{label.replace(' ', '_')}.png"

    print("[INFO] extract:", label, video_path)
    extract_frame(video_path, frame_path, ratio=0.35)

    frame = Image.open(frame_path).convert("RGB")
    parts = crop_columns(frame)
    step_rows.append((label, parts))
    max_cols = max(max_cols, len(parts))

# -----------------------------
# 生成最终图
# -----------------------------
# 左侧输入图占两行高度；右侧为 3000/5000 两行，每行 3~4 个输出
input_w = 300
cell_w = 270
cell_h = 210
row_label_w = 110
gap = 18
top_h = 46

col_titles_3 = ["RGB view", "Normal view", "Mask / depth"]
col_titles_4 = ["RGB view", "Normal view", "Depth view", "Mask view"]

col_titles = col_titles_4 if max_cols >= 4 else col_titles_3

canvas_w = input_w + gap + row_label_w + max_cols * cell_w + (max_cols - 1) * gap + 40
canvas_h = top_h + 2 * cell_h + gap + 50

canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

# 总标题
draw.text((20, 12), "Object C: single-image-to-3D generation using stable-Zero123", fill=(0, 0, 0))

# 左侧输入图
input_panel = make_image_panel(input_img, (input_w, 2 * cell_h + gap), "Input RGBA image")
canvas.paste(input_panel, (20, top_h))

# 右侧列标题
x_start = 20 + input_w + gap + row_label_w
for j in range(max_cols):
    title = col_titles[j] if j < len(col_titles) else f"View {j+1}"
    x = x_start + j * (cell_w + gap) + 8
    draw.text((x, top_h - 26), title, fill=(0, 0, 0))

# 右侧两行
for i, (label, parts) in enumerate(step_rows):
    y = top_h + i * (cell_h + gap)

    # 行标签
    draw.text((20 + input_w + gap + 8, y + cell_h // 2 - 8), label, fill=(0, 0, 0))

    for j in range(max_cols):
        x = x_start + j * (cell_w + gap)

        if j < len(parts):
            panel = make_image_panel(parts[j], (cell_w, cell_h), "")
        else:
            panel = Image.new("RGB", (cell_w, cell_h), "white")

        # 给每个 panel 加细边框
        pd = ImageDraw.Draw(panel)
        pd.rectangle((0, 0, cell_w - 1, cell_h - 1), outline=(180, 180, 180), width=1)

        canvas.paste(panel, (x, y))

out = FIG_DIR / "p1_4_object_C_zero123_result.png"
canvas.save(out, quality=95)
print("[DONE]", out)
PY

echo
echo "[DONE] P1-4 figure saved to:"
echo "$FIG_DIR/p1_4_object_C_zero123_result.png"
