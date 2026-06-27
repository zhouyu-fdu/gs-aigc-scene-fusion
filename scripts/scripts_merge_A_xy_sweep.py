from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import shutil

BG_MODEL = Path("outputs/background_3dgs/garden_30000")
BG_PLY = BG_MODEL / "point_cloud/iteration_30000/point_cloud.ply"
A_PLY = Path("final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply")

OUT_ROOT = Path("outputs/fusion_scene/xy_sweep_A")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 为了 debug 先稍微放大一点，方便看见
TARGET_SIZE = 0.25

# 你已经确认 z 变大杯子会更高，先固定一个中间高度
Z_OFFSET = 1.00

# 扫水平位置。先不要太多，避免生成太多 1GB 级别的融合 PLY
XY_OFFSETS = {
    "center":      (0.00,  0.00),
    "x_p040":      (0.40,  0.00),
    "x_m040":      (-0.40, 0.00),
    "y_p040":      (0.00,  0.40),
    "y_m040":      (0.00, -0.40),
    "xp040_yp040": (0.40,  0.40),
    "xp040_ym040": (0.40, -0.40),
    "xm040_yp040": (-0.40, 0.40),
    "xm040_ym040": (-0.40,-0.40),
}

def read_vertex(path):
    return PlyData.read(path)["vertex"].data

def write_vertex(path, vertex):
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)

bg_v = read_vertex(BG_PLY)
a_v = read_vertex(A_PLY)

if bg_v.dtype != a_v.dtype:
    raise RuntimeError("Background and object A Gaussian fields are inconsistent.")

bg_xyz = np.stack([bg_v["x"], bg_v["y"], bg_v["z"]], axis=1).astype(np.float32)
a_xyz = np.stack([a_v["x"], a_v["y"], a_v["z"]], axis=1).astype(np.float32)

# 用 median 比 mean 稳一点
bg_center = np.median(bg_xyz, axis=0)

a_center = np.median(a_xyz, axis=0)
a_xyz_centered = a_xyz - a_center
a_span = np.quantile(a_xyz_centered, 0.98, axis=0) - np.quantile(a_xyz_centered, 0.02, axis=0)
a_long = float(np.max(a_span))
scale = TARGET_SIZE / max(a_long, 1e-6)

print("[INFO] bg_center:", bg_center)
print("[INFO] a_span:", a_span)
print("[INFO] scale:", scale)
print("[INFO] fixed Z_OFFSET:", Z_OFFSET)

for tag, (xo, yo) in XY_OFFSETS.items():
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

    place = bg_center + np.array([xo, yo, Z_OFFSET], dtype=np.float32)

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

    print("[DONE]", tag, "xy=", (xo, yo), "z=", Z_OFFSET, "->", out_model)
