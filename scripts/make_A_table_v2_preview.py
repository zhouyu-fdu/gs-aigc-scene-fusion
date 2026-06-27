from pathlib import Path
from PIL import Image, ImageDraw
import math

base = Path("outputs/fusion_scene/garden_with_A_on_table_v2")
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
    im.thumbnail((320, 220))
    canvas = Image.new("RGB", (320, 245), "white")
    canvas.paste(im, ((320 - im.width)//2, 0))
    d = ImageDraw.Draw(canvas)
    d.text((5, 224), p.name, fill=(0,0,0))
    thumbs.append(canvas)

cols = 4
rows = math.ceil(len(thumbs) / cols)
out = Image.new("RGB", (cols * 320, rows * 245), "white")

for i, im in enumerate(thumbs):
    out.paste(im, ((i % cols) * 320, (i // cols) * 245))

out_path = Path("final_assets/fusion_preview/garden_with_A_on_table_v2_preview.jpg")
out_path.parent.mkdir(parents=True, exist_ok=True)
out.save(out_path)

print("[DONE]", out_path)
print("[SOURCE]", render_dir)
