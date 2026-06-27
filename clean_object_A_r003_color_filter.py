from pathlib import Path
from plyfile import PlyData, PlyElement
import numpy as np

C0 = 0.28209479177387814

in_ply = Path("final_assets/object_A_clean/seed_neighborhood_candidates/object_A_m120_seed_r003.ply")
out_dir = Path("final_assets/object_A_clean/r003_color_clean_candidates")
out_dir.mkdir(parents=True, exist_ok=True)

if not in_ply.exists():
    raise FileNotFoundError(in_ply)

ply = PlyData.read(in_ply)
v = ply["vertex"].data
names = v.dtype.names

xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

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

    return np.ones((len(v), 3), dtype=np.float32) * 0.5

rgb = get_rgb(v)
R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
brightness = (R + G + B) / 3.0

if "opacity" in names:
    alpha = sigmoid(v["opacity"].astype(np.float32))
else:
    alpha = np.ones(len(v), dtype=np.float32)

# 核心保留区域：杯子主体、蓝色勺子、深色细节
cup_brown = (
    (alpha > 0.015) &
    (brightness < 0.62) &
    (R > B + 0.02) &
    (R > 0.08)
)

blue_spoon = (
    (alpha > 0.010) &
    (B > R + 0.04) &
    (B > 0.16)
)

dark_detail = (
    (alpha > 0.015) &
    (brightness < 0.40)
)

core_keep = cup_brown | blue_spoon | dark_detail

# 桌面/背景颜色：浅、低饱和、偏木色/灰白
maxc = np.maximum.reduce([R, G, B])
minc = np.minimum.reduce([R, G, B])
sat = maxc - minc

wood_like = (
    (brightness > 0.48) &
    (sat < 0.28) &
    (R > B - 0.04) &
    (G > B - 0.08)
)

pale_like = (
    (brightness > 0.62) &
    (sat < 0.22)
)

cyan_noise = (
    (B > R + 0.07) &
    (G > R + 0.03) &
    (brightness > 0.48)
)

# 不同强度版本
settings = {
    "soft":   {"wood": 0.58, "pale": 0.72, "alpha": 0.006},
    "mid":    {"wood": 0.52, "pale": 0.66, "alpha": 0.008},
    "strong": {"wood": 0.48, "pale": 0.62, "alpha": 0.010},
    "xstrong":{"wood": 0.44, "pale": 0.58, "alpha": 0.012},
}

print("[INFO] total:", len(v))
print("[INFO] core_keep:", int(core_keep.sum()))
print("[INFO] cup_brown:", int(cup_brown.sum()))
print("[INFO] blue_spoon:", int(blue_spoon.sum()))
print("[INFO] dark_detail:", int(dark_detail.sum()))

for name, cfg in settings.items():
    wood_remove = (
        (brightness > cfg["wood"]) &
        (sat < 0.30) &
        (R > B - 0.05) &
        (G > B - 0.10)
    )

    pale_remove = (
        (brightness > cfg["pale"]) &
        (sat < 0.25)
    )

    remove = (wood_remove | pale_remove | cyan_noise | (alpha < cfg["alpha"])) & (~core_keep)

    keep = ~remove

    cropped = v[keep]
    out_ply = out_dir / f"object_A_r003_color_{name}.ply"
    PlyData([PlyElement.describe(cropped, "vertex")], text=False).write(out_ply)

    print()
    print(name)
    print("  kept:", int(keep.sum()), "/", len(v))
    print("  removed:", int(remove.sum()))
    print("  out:", out_ply)

print("[DONE] color clean candidates generated.")
