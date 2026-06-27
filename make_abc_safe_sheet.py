from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
names = [
    "garden_with_ABC_safe1",
    "garden_with_ABC_safe2",
    "garden_with_ABC_safe3",
]

panels = []
for name in names:
    render_dir = ROOT / "outputs/fusion_scene" / name / "test/ours_30000/renders"
    imgs = sorted(render_dir.glob("*.png"))
    if not imgs:
        print("[WARN] no renders:", render_dir)
        continue
    pick = imgs[len(imgs)//2]
    im = Image.open(pick).convert("RGB")
    im = ImageOps.contain(im, (760, 520))

    canvas = Image.new("RGB", (780, 570), "white")
    canvas.paste(im, ((780 - im.width)//2, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), f"{name} | {pick.name}", fill=(0, 0, 0))
    panels.append(canvas)

if not panels:
    raise RuntimeError("No rendered candidate images found.")

sheet = Image.new("RGB", (780, 570 * len(panels)), "white")
for i, p in enumerate(panels):
    sheet.paste(p, (0, i * 570))

out = ROOT / "final_assets/fusion_debug/abc_safe_candidates_sheet.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=95)
print("[DONE]", out)
