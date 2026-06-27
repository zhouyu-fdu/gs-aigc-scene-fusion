from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import shutil

C0 = 0.28209479177387814

BG_MODEL = Path("outputs/background_3dgs/garden_30000")
BG_PLY = BG_MODEL / "point_cloud/iteration_30000/point_cloud.ply"
A_PLY = Path("final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply")

OUT_ROOT = Path("outputs/fusion_scene/debug_A_z_visible_sweep")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# debug 时故意放大，方便看见
TARGET_SIZE = 0.65

# 先固定当前 x/y，只扫描 z
XY_OFFSET = np.array([0.0, 0.0], dtype=np.float32)

# z 从低到高大范围扫
Z_OFFSETS = [-0.2, 0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6]

def read_vertex(path):
    return PlyData.read(path)["vertex"].data

def write_vertex(path, vertex):
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)

print("[INFO] reading background:", BG_PLY)
bg_v = read_vertex(BG_PLY)

print("[INFO] reading object A:", A_PLY)
a_v = read_vertex(A_PLY)

if bg_v.dtype != a_v.dtype:
    raise RuntimeError("Background and object A Gaussian fields are inconsistent.")

bg_xyz = np.stack([bg_v["x"], bg_v["y"], bg_v["z"]], axis=1).astype(np.float32)
a_xyz = np.stack([a_v["x"], a_v["y"], a_v["z"]], axis=1).astype(np.float32)

bg_center = np.median(bg_xyz, axis=0)

a_center = np.median(a_xyz, axis=0)
a_xyz_centered = a_xyz - a_center
a_span = np.quantile(a_xyz_centered, 0.98, axis=0) - np.quantile(a_xyz_centered, 0.02, axis=0)
a_long = float(np.max(a_span))
scale = TARGET_SIZE / max(a_long, 1e-6)

print("[INFO] bg_center:", bg_center)
print("[INFO] a_center:", a_center)
print("[INFO] a_span:", a_span)
print("[INFO] target_size:", TARGET_SIZE)
print("[INFO] scale:", scale)

for z in Z_OFFSETS:
    tag = f"z{z:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
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

    # 高斯尺度同步缩放
    log_s = np.log(scale)
    for k in ["scale_0", "scale_1", "scale_2"]:
        if k in a_new.dtype.names:
            a_new[k] = a_new[k] + log_s

    # 染成红色，方便肉眼确认
    if all(k in a_new.dtype.names for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        rgb = np.array([1.0, 0.02, 0.02], dtype=np.float32)
        sh = (rgb - 0.5) / C0
        a_new["f_dc_0"] = sh[0]
        a_new["f_dc_1"] = sh[1]
        a_new["f_dc_2"] = sh[2]

    # 提高 opacity，debug 更容易看见
    if "opacity" in a_new.dtype.names:
        a_new["opacity"] = np.maximum(a_new["opacity"], 3.0)

    merged = np.empty(len(bg_v) + len(a_new), dtype=bg_v.dtype)
    merged[:len(bg_v)] = bg_v
    merged[len(bg_v):] = a_new

    write_vertex(out_ply, merged)

    print("[DONE]", tag, "z =", z, "place =", place, "->", out_model)
