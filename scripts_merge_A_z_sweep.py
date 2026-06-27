from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import shutil

BG_MODEL = Path("outputs/background_3dgs/garden_30000")
BG_PLY = BG_MODEL / "point_cloud/iteration_30000/point_cloud.ply"
A_PLY = Path("final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply")

OUT_ROOT = Path("outputs/fusion_scene/z_sweep_A")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 0.18

# 重点扫这里。0.38 已经偏低，所以从 0.6 往上试
Z_OFFSETS = [0.55, 0.70, 0.85, 1.00, 1.15, 1.30]

XY_OFFSET = np.array([0.0, 0.0], dtype=np.float32)

def read_vertex(path):
    ply = PlyData.read(path)
    return ply["vertex"].data

def write_vertex(path, vertex):
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)

print("[INFO] reading background...")
bg_v = read_vertex(BG_PLY)

print("[INFO] reading object A...")
a_v = read_vertex(A_PLY)

if bg_v.dtype != a_v.dtype:
    raise RuntimeError("Background and A Gaussian fields are inconsistent.")

bg_xyz = np.stack([bg_v["x"], bg_v["y"], bg_v["z"]], axis=1).astype(np.float32)
a_xyz = np.stack([a_v["x"], a_v["y"], a_v["z"]], axis=1).astype(np.float32)

bg_center = np.median(bg_xyz, axis=0)

a_center = np.median(a_xyz, axis=0)
a_xyz_centered = a_xyz - a_center
a_span = np.quantile(a_xyz_centered, 0.98, axis=0) - np.quantile(a_xyz_centered, 0.02, axis=0)
a_long = float(np.max(a_span))
scale = TARGET_SIZE / max(a_long, 1e-6)

print("[INFO] bg_center:", bg_center)
print("[INFO] a_span:", a_span)
print("[INFO] scale:", scale)

for z in Z_OFFSETS:
    tag = f"z{z:.2f}".replace(".", "p")
    out_model = OUT_ROOT / tag
    out_ply = out_model / "point_cloud/iteration_30000/point_cloud.ply"

    if out_model.exists():
        shutil.rmtree(out_model)

    shutil.copytree(
        BG_MODEL,
        out_model,
        ignore=shutil.ignore_patterns("point_cloud")
    )

    out_ply.parent.mkdir(parents=True, exist_ok=True)

    place = bg_center + np.array([XY_OFFSET[0], XY_OFFSET[1], z], dtype=np.float32)

    a_new = a_v.copy()
    a_new_xyz = a_xyz_centered * scale + place

    a_new["x"] = a_new_xyz[:, 0]
    a_new["y"] = a_new_xyz[:, 1]
    a_new["z"] = a_new_xyz[:, 2]

    log_s = np.log(scale)
    for k in ["scale_0", "scale_1", "scale_2"]:
        if k in a_new.dtype.names:
            a_new[k] = a_new[k] + log_s

    merged = np.empty(len(bg_v) + len(a_new), dtype=bg_v.dtype)
    merged[:len(bg_v)] = bg_v
    merged[len(bg_v):] = a_new

    write_vertex(out_ply, merged)

    print("[DONE]", tag, "OFFSET z =", z, "->", out_model)
