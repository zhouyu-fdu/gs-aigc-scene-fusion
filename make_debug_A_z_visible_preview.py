from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

root = Path("outputs/fusion_scene/debug_A_z_visible_sweep")
out_path = Path("final_assets/fusion_preview/debug_A_z_visible_sweep_preview.jpg")
out_path.parent.mkdir(parents=True, exist_ok=True)

models = sorted([p for p in root.glob("z*") if p.is_dir()])
thumbs = []

def red_score(im):
    arr = np.asarray(im).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    score = np.maximum(r - 1.25 * g - 1.25 * b, 0)
    return float(score.mean())

for m in models:
    candidates = [
        m / "test/ours_30000/renders",
        m / "train/ours_30000/renders",
    ]

    render_dir = None
    for d in candidates:
        if d.exists() and len(list(d.glob("*.png"))) > 0:
            render_dir = d
            break

    if render_dir is None:
        print("[WARN] no renders:", m)
        continue

    imgs = sorted(render_dir.glob("*.png"))

    best_p = None
    best_im = None
    best_s = -1.0

    for p in imgs:
        im = Image.open(p).convert("RGB")
        s = red_score(im)
        if s > best_s:
            best_s = s
            best_p = p
            best_im = im

    best_im.thumbnail((420, 280))
    canvas = Image.new("RGB", (420, 315), "white")
    canvas.paste(best_im, ((420 - best_im.width) // 2, 0))

    d = ImageDraw.Draw(canvas)
    d.text((8, 288), f"{m.name} | {best_p.name} | red={best_s:.2f}", fill=(0, 0, 0))
    thumbs.append(canvas)

cols = 2
rows = (len(thumbs) + cols - 1) // cols

sheet = Image.new("RGB", (cols * 420, rows * 315), "white")
for i, im in enumerate(thumbs):
    sheet.paste(im, ((i % cols) * 420, (i // cols) * 315))

sheet.save(out_path)
print("[DONE]", out_path)
