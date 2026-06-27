from pathlib import Path
import json
import os
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

MODEL = Path("outputs/background_3dgs/garden_30000")
PLY = MODEL / "point_cloud/iteration_30000/point_cloud.ply"
CAM_JSON = MODEL / "cameras.json"

IMG = Path(os.environ["IMG"])
PIXEL_X = float(os.environ["PIXEL_X"])
PIXEL_Y = float(os.environ["PIXEL_Y"])

img = Image.open(IMG).convert("RGB")
W, H = img.size
img_stem = IMG.stem

cams = json.loads(CAM_JSON.read_text())
cam = None
for c in cams:
    name = str(c.get("img_name", ""))
    if Path(name).stem == img_stem or name == img_stem:
        cam = c
        break

if cam is None:
    cam = cams[len(cams)//2]
    print("[WARN] no exact camera match, fallback to middle camera")

fx = float(cam["fx"])
fy = float(cam["fy"])
cx = W / 2.0
cy = H / 2.0

C = np.array(cam["position"], dtype=np.float32)
R = np.array(cam["rotation"], dtype=np.float32)

ply = PlyData.read(PLY)
v = ply["vertex"].data
xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

best = None

for variant in [0, 1]:
    if variant == 0:
        cam_xyz = (xyz - C) @ R
    else:
        cam_xyz = (xyz - C) @ R.T

    z = cam_xyz[:, 2]
    valid = z > 1e-6

    u = fx * (cam_xyz[:, 0] / z) + cx
    vv = fy * (cam_xyz[:, 1] / z) + cy
    valid &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)

    if valid.sum() < 1000:
        continue

    du = u - PIXEL_X
    dv = vv - PIXEL_Y
    pix_dist = np.sqrt(du * du + dv * dv)

    near = valid & (pix_dist < 8.0)
    if near.sum() < 5:
        near = valid & (pix_dist < 15.0)
    if near.sum() < 5:
        continue

    idxs = np.where(near)[0]
    depths = cam_xyz[idxs, 2]
    best_local = idxs[np.argmin(depths)]

    cand = {
        "variant": variant,
        "idx": int(best_local),
        "xyz": xyz[best_local],
        "u": float(u[best_local]),
        "v": float(vv[best_local]),
        "depth": float(cam_xyz[best_local, 2]),
        "near_count": int(near.sum()),
        "pix_dist": float(pix_dist[best_local]),
    }

    if best is None or cand["pix_dist"] < best["pix_dist"]:
        best = cand

if best is None:
    raise RuntimeError("No 3D point found near the selected pixel, try another tabletop pixel.")

out_dir = Path("final_assets/fusion_debug")
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "table_pick_xyz.npy", best["xyz"])

draw = ImageDraw.Draw(img)
x, y = PIXEL_X, PIXEL_Y
draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 0, 0), width=3)
draw.line((x - 15, y, x + 15, y), fill=(255, 0, 0), width=2)
draw.line((x, y - 15, x, y + 15), fill=(255, 0, 0), width=2)

u, vv = best["u"], best["v"]
draw.ellipse((u - 5, vv - 5, u + 5, vv + 5), outline=(0, 255, 0), width=3)

debug_img = out_dir / "table_pick_pixel_debug.jpg"
img.save(debug_img)

print("[RESULT] table xyz:", best["xyz"].tolist())
print("[RESULT] projected uv:", best["u"], best["v"])
print("[RESULT] pixel distance:", best["pix_dist"])
print("[RESULT] saved:", out_dir / "table_pick_xyz.npy")
print("[RESULT] debug image:", debug_img)
