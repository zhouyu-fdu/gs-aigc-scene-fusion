from pathlib import Path
from PIL import Image, ImageDraw

root = Path("outputs/fusion_scene/z_sweep_A")
out_path = Path("final_assets/fusion_preview/A_z_sweep_preview.jpg")
out_path.parent.mkdir(parents=True, exist_ok=True)

models = sorted([p for p in root.glob("z*") if p.is_dir()])

thumbs = []

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

    # 选一个中间视角作为对比
    p = imgs[len(imgs) // 2]

    im = Image.open(p).convert("RGB")
    im.thumbnail((420, 280))

    canvas = Image.new("RGB", (420, 315), "white")
    canvas.paste(im, ((420 - im.width) // 2, 0))

    d = ImageDraw.Draw(canvas)
    d.text((8, 288), f"{m.name} | {p.name}", fill=(0, 0, 0))
    thumbs.append(canvas)

cols = 2
rows = (len(thumbs) + cols - 1) // cols

sheet = Image.new("RGB", (cols * 420, rows * 315), "white")
for i, im in enumerate(thumbs):
    sheet.paste(im, ((i % cols) * 420, (i // cols) * 315))

sheet.save(out_path)
print("[DONE]", out_path)
