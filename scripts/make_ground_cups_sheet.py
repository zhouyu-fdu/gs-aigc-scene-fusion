from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
render_dir = ROOT / "outputs/fusion_scene/garden_with_ABC_ground_cups/test/ours_30000/renders"

imgs = sorted(render_dir.glob("*.png"))
print("[INFO] renders:", len(imgs))
if not imgs:
    raise RuntimeError(f"No renders found in {render_dir}")

idxs = [0, 2, 4, 6, 8, 10, 12, 14, 16]
picks = [imgs[i] for i in idxs if i < len(imgs)]

tiles = []
for p in picks:
    im = Image.open(p).convert("RGB")
    im = ImageOps.contain(im, (360, 245))
    canvas = Image.new("RGB", (380, 295), "white")
    canvas.paste(im, ((380 - im.width)//2, 10))
    d = ImageDraw.Draw(canvas)
    d.text((10, 258), p.name, fill=(0, 0, 0))
    tiles.append(canvas)

cols = 3
rows = math.ceil(len(tiles) / cols)
sheet = Image.new("RGB", (cols * 380, rows * 295), "white")

for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * 380, (i // cols) * 295))

out = ROOT / "final_assets/fusion_debug/garden_with_ABC_ground_cups_sheet.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=95)
print("[DONE]", out)
