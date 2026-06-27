from pathlib import Path
from PIL import Image, ImageDraw
import math

base = Path("outputs/object_A_3dgs/object_A_r003_color_strong_render_test")
candidates = [
    base / "test/ours_30000/renders",
    base / "train/ours_30000/renders",
]

render_dir = None
for d in candidates:
    if d.exists() and len(list(d.glob("*.png"))) > 0:
        render_dir = d
        break

if render_dir is None:
    raise RuntimeError("No render images found.")

imgs = sorted(render_dir.glob("*.png"))
sel = imgs[::max(1, len(imgs)//20)][:20]

thumbs = []
for p in sel:
    im = Image.open(p).convert("RGB")
    im.thumbnail((240, 180))
    canvas = Image.new("RGB", (240, 205), "white")
    canvas.paste(im, ((240 - im.width) // 2, 0))
    d = ImageDraw.Draw(canvas)
    d.text((5, 184), p.name, fill=(0, 0, 0))
    thumbs.append(canvas)

cols = 5
rows = math.ceil(len(thumbs) / cols)
out = Image.new("RGB", (cols * 240, rows * 205), "white")

for i, im in enumerate(thumbs):
    out.paste(im, ((i % cols) * 240, (i // cols) * 205))

out_path = Path("final_assets/object_A_clean/object_A_r003_color_strong_render_preview.jpg")
out_path.parent.mkdir(parents=True, exist_ok=True)
out.save(out_path)

print("[DONE]", out_path)
print("[SOURCE]", render_dir)
