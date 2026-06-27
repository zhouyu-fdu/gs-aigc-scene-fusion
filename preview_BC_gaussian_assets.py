from pathlib import Path
import numpy as np
from plyfile import PlyData
from PIL import Image, ImageDraw

C0 = 0.28209479177387814

ITEMS = [
    ("Object B: red apple", Path("final_assets/for_fusion/object_B/object_B_gaussian_asset.ply")),
    ("Object C: cup bear", Path("final_assets/for_fusion/object_C/object_C_gaussian_asset.ply")),
]

OUT_DIR = Path("final_assets/for_fusion/previews")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_rgb(v):
    names = v.dtype.names
    if all(k in names for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        sh = np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
            axis=1
        ).astype(np.float32)
        rgb = sh * C0 + 0.5
        return np.clip(rgb, 0.0, 1.0)
    return np.ones((len(v), 3), dtype=np.float32) * 0.7

def render_view(xyz, rgb, axes=(0, 1), size=(420, 320), title=""):
    W, H = size
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    pts = xyz[:, list(axes)]
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    span[span < 1e-6] = 1.0

    norm = (pts - mn) / span
    px = (norm[:, 0] * (W - 40) + 20).astype(np.int32)
    py = ((1.0 - norm[:, 1]) * (H - 60) + 40).astype(np.int32)

    colors = (rgb * 255).clip(0, 255).astype(np.uint8)

    valid = (px >= 1) & (px < W - 1) & (py >= 1) & (py < H - 1)
    px = px[valid]
    py = py[valid]
    colors = colors[valid]

    # 简单 2x2 splat，让点云更容易看见
    canvas[py, px] = colors
    canvas[py + 1, px] = colors
    canvas[py, px + 1] = colors
    canvas[py + 1, px + 1] = colors

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((12, 10), title, fill=(0, 0, 0))
    return img

def main():
    all_rows = []

    for name, ply_path in ITEMS:
        print("=" * 80)
        print("[ITEM]", name)
        print("[PLY ]", ply_path)

        ply = PlyData.read(str(ply_path))
        v = ply["vertex"].data

        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        rgb = get_rgb(v)

        print("[INFO] points:", len(xyz))
        print("[INFO] bounds min:", xyz.min(axis=0).tolist())
        print("[INFO] bounds max:", xyz.max(axis=0).tolist())
        print("[INFO] rgb mean  :", rgb.mean(axis=0).tolist())

        if len(xyz) > 80000:
            rng = np.random.default_rng(123)
            idx = rng.choice(len(xyz), size=80000, replace=False)
            xyz_show = xyz[idx]
            rgb_show = rgb[idx]
        else:
            xyz_show = xyz
            rgb_show = rgb

        views = [
            ((0, 1), "XY view"),
            ((0, 2), "XZ view"),
            ((1, 2), "YZ view"),
        ]

        row_imgs = []
        for axes, title in views:
            row_imgs.append(render_view(xyz_show, rgb_show, axes=axes, title=title))

        W = sum(im.width for im in row_imgs)
        H = max(im.height for im in row_imgs) + 40
        row = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(row)
        draw.text((12, 8), f"{name} | points={len(xyz)}", fill=(0, 0, 0))

        x = 0
        for im in row_imgs:
            row.paste(im, (x, 40))
            x += im.width

        out = OUT_DIR / (name.split(":")[0].replace(" ", "_") + "_gaussian_preview.jpg")
        row.save(out, quality=95)
        print("[DONE]", out)

        all_rows.append(row)

    W = max(im.width for im in all_rows)
    H = sum(im.height for im in all_rows)
    sheet = Image.new("RGB", (W, H), "white")

    y = 0
    for im in all_rows:
        sheet.paste(im, (0, y))
        y += im.height

    out_all = OUT_DIR / "BC_gaussian_assets_preview.jpg"
    sheet.save(out_all, quality=95)
    print("[ALL DONE]", out_all)

if __name__ == "__main__":
    main()
