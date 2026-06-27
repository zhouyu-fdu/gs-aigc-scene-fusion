from pathlib import Path
import numpy as np
from plyfile import PlyData

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

C0 = 0.28209479177387814

cand_dir = Path("final_assets/object_A_clean/r003_color_clean_candidates")
out_dir = Path("final_assets/object_A_clean/r003_color_clean_preview")
out_dir.mkdir(parents=True, exist_ok=True)

ply_files = sorted(cand_dir.glob("*.ply"))
if len(ply_files) == 0:
    raise RuntimeError(f"No candidates found in {cand_dir}")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_rgb(v):
    names = v.dtype.names
    if all(k in names for k in ["red", "green", "blue"]):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0
        return np.clip(rgb, 0, 1)

    if all(k in names for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        sh = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
        rgb = sh * C0 + 0.5
        return np.clip(rgb, 0, 1)

    return np.ones((len(v), 3), dtype=np.float32) * 0.7

preview_imgs = []

for ply_path in ply_files:
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = get_rgb(v)

    if "opacity" in v.dtype.names:
        alpha = sigmoid(v["opacity"].astype(np.float32))
    else:
        alpha = np.ones(len(xyz), dtype=np.float32)

    keep = alpha > 0.02
    xyz2 = xyz[keep]
    rgb2 = rgb[keep]

    if len(xyz2) > 80000:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(xyz2), size=80000, replace=False)
        xyz2 = xyz2[idx]
        rgb2 = rgb2[idx]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    views = [
        ("XY top", 0, 1),
        ("XZ front", 0, 2),
        ("YZ side", 1, 2),
    ]

    for ax, (title, a, b) in zip(axes, views):
        ax.scatter(xyz2[:, a], xyz2[:, b], s=0.35, c=rgb2)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    fig.suptitle(f"{ply_path.name} | points {len(xyz2)}", fontsize=10)
    out_png = out_dir / f"{ply_path.stem}_preview.png"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

    preview_imgs.append(out_png)
    print("[DONE]", out_png)

thumbs = []
for p in preview_imgs:
    im = Image.open(p).convert("RGB")
    im.thumbnail((900, 300))
    canvas = Image.new("RGB", (900, 330), "white")
    canvas.paste(im, ((900 - im.width) // 2, 0))
    thumbs.append(canvas)

sheet = Image.new("RGB", (900, 330 * len(thumbs)), "white")
for i, im in enumerate(thumbs):
    sheet.paste(im, (0, i * 330))

sheet_path = out_dir / "all_r003_color_clean_preview.jpg"
sheet.save(sheet_path)

print("[ALL PREVIEW]", sheet_path)
