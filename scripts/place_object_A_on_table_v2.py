from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import shutil

BG_MODEL = Path("outputs/background_3dgs/garden_30000")
BG_PLY = BG_MODEL / "point_cloud/iteration_30000/point_cloud.ply"

A_PLY = Path("final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply")
TABLE_XYZ = np.load("final_assets/fusion_debug/table_pick_xyz.npy").astype(np.float32)

OUT_MODEL = Path("outputs/fusion_scene/garden_with_A_on_table_v2")
OUT_PLY = OUT_MODEL / "point_cloud/iteration_30000/point_cloud.ply"

# 关键参数：先用小一点，避免杯子巨大
TARGET_SIZE = 0.12

# 让杯底略高于桌面，避免被桌面吞掉
LIFT = 0.02

def read_vertex(path):
    return PlyData.read(path)["vertex"].data

def write_vertex(path, vertex):
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)

bg_v = read_vertex(BG_PLY)
a_v = read_vertex(A_PLY)

if bg_v.dtype != a_v.dtype:
    raise RuntimeError("Background and object A Gaussian fields are inconsistent.")

a_xyz = np.stack([a_v["x"], a_v["y"], a_v["z"]], axis=1).astype(np.float32)

# 以中位数为中心，更稳
a_center = np.median(a_xyz, axis=0)
a_centered = a_xyz - a_center

# 用 2%-98% bbox 估计尺寸，避免极端点影响
a_lo = np.quantile(a_centered, 0.02, axis=0)
a_hi = np.quantile(a_centered, 0.98, axis=0)
a_span = a_hi - a_lo
a_long = float(np.max(a_span))

scale = TARGET_SIZE / max(a_long, 1e-6)
a_scaled = a_centered * scale

# 以缩放后 A 的 1% z 分位数作为底部
z_bottom = np.quantile(a_scaled[:, 2], 0.01)

# table_xyz 是桌面上的点，所以让 A 的底部 z 对齐到 table z + LIFT
place = TABLE_XYZ.copy()
place[2] = TABLE_XYZ[2] - z_bottom + LIFT

a_new_xyz = a_scaled + place

a_new = a_v.copy()
a_new["x"] = a_new_xyz[:, 0]
a_new["y"] = a_new_xyz[:, 1]
a_new["z"] = a_new_xyz[:, 2]

# 非常关键：Gaussian 自身尺度也必须同步缩放
# 3DGS 里的 scale_0/1/2 是 log-space，所以要加 log(scale)
log_s = np.log(scale)
for k in ["scale_0", "scale_1", "scale_2"]:
    if k in a_new.dtype.names:
        a_new[k] = a_new[k] + log_s

merged = np.empty(len(bg_v) + len(a_new), dtype=bg_v.dtype)
merged[:len(bg_v)] = bg_v
merged[len(bg_v):] = a_new

if OUT_MODEL.exists():
    shutil.rmtree(OUT_MODEL)

shutil.copytree(
    BG_MODEL,
    OUT_MODEL,
    ignore=shutil.ignore_patterns("point_cloud")
)

OUT_PLY.parent.mkdir(parents=True, exist_ok=True)
write_vertex(OUT_PLY, merged)

print("[DONE]", OUT_MODEL)
print("[INFO] table_xyz:", TABLE_XYZ.tolist())
print("[INFO] target_size:", TARGET_SIZE)
print("[INFO] A span:", a_span.tolist())
print("[INFO] scale:", float(scale))
print("[INFO] log_s:", float(log_s))
print("[INFO] z_bottom_scaled:", float(z_bottom))
print("[INFO] place:", place.tolist())
print("[INFO] bg points:", len(bg_v))
print("[INFO] A points:", len(a_new))
print("[INFO] merged:", len(merged))
