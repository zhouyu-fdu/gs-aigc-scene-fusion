#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT=/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion
FIG_DIR="$PROJECT_ROOT/figures"
mkdir -p "$FIG_DIR"

cd "$PROJECT_ROOT"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    {
        "step": "6000 steps",
        "path": ROOT / "outputs/object_B_text3d/dreamfusion_sd/red_apple_sd_seed42_6000_gpu0@20260621-040102/save/it6000-test.mp4",
    },
    {
        "step": "10000 steps",
        "path": ROOT / "outputs/object_B_text3d/dreamfusion_sd/red_apple_sd_seed42_6000_gpu0@20260621-050420/save/it10000-test.mp4",
    },
    {
        "step": "15000 steps",
        "path": ROOT / "outputs/object_B_text3d/dreamfusion_sd/red_apple_sd_seed42_6000_gpu0@20260621-060225/save/it15000-test.mp4",
    },
]

OUT_FRAMES = FIG_DIR / "p1_3_object_B_frames"
OUT_FRAMES.mkdir(parents=True, exist_ok=True)

def extract_middle_frame(video_path: Path, out_path: Path):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(
            "当前 Python 环境没有 cv2。请先尝试：conda activate gs_splatting；"
            "如果仍没有，则安装 opencv-python-headless。"
        ) from e

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        frame_id = 0
    else:
        # 取中后段，比第一帧更容易看到完整物体
        frame_id = int(n * 0.35)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()

    # 如果指定帧失败，退回逐帧读
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    im.save(out_path)
    return out_path

W, H = 520, 420
TITLE_H = 54

def make_panel(image_path: Path, title: str):
    im = Image.open(image_path).convert("RGB")
    im = ImageOps.contain(im, (W, H - TITLE_H - 10))

    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, ((W - im.width) // 2, TITLE_H + (H - TITLE_H - im.height) // 2))

    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, fill=(0, 0, 0))
    return canvas

panels = []
for item in VIDEOS:
    video = item["path"]
    step = item["step"]
    frame_out = OUT_FRAMES / f"red_apple_{step.replace(' ', '_')}.png"

    print("[INFO] extracting:", video)
    extract_middle_frame(video, frame_out)
    panels.append(make_panel(frame_out, f"Red apple | {step}"))

sheet = Image.new("RGB", (W * 3, H), "white")
for i, panel in enumerate(panels):
    sheet.paste(panel, (i * W, 0))

out = FIG_DIR / "p1_3_object_B_red_apple_steps.png"
sheet.save(out, quality=95)

print("[DONE]", out)
PY

echo
echo "[DONE] P1-3 figure saved to:"
echo "$FIG_DIR/p1_3_object_B_red_apple_steps.png"
