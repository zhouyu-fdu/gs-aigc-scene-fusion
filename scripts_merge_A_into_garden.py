from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import shutil

BG_MODEL = Path("outputs/background_3dgs/garden_30000")
BG_PLY = BG_MODEL / "point_cloud/iteration_30000/point_cloud.ply"

A_PLY = Path("final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply")

OUT_MODEL = Path("outputs/fusion_scene/garden_with_A_test")
OUT_PLY = OUT_MODEL / "point_cloud/iteration_30000/point_cloud.ply"

# =========================
# 这些参数后面可以调
# =========================
# target_size：A 在 garden 坐标里的整体尺寸，太大就减小，太小就增大
TARGET_SIZE = 0.25

# 位置偏移。第一次先放在背景点云中心附近。
# 如果渲染里看不到 A，或者位置不合适，后面主要改这三个数。
OFFSET = np.array([0.0, 0.0, 0.45], dtype=np.float32)

# 是否只做平移缩放。先不旋转，后面如有需要再调。
# =========================

print("[INFO] BG_PLY:", BG_PLY)
print("[INFO] A_PLY :", A_PLY)

if not BG_PLY.exists():
    raise FileNotFoundError(BG_PLY)
if not A_PLY.exists():
    raise FileNotFoundError(A_PLY)

# 复制背景模型目录结构，但不复制原 point_cloud
if OUT_MODEL.exists():
    shutil.rmtree(OUT_MODEL)

shutil.copytree(
    BG_MODEL,
    OUT_MODEL,
    ignore=shutil.ignore_patterns("point_cloud")
)

OUT_PLY.parent.mkdir(parents=True, exist_ok=True)

print("[INFO] Reading background...")
bg_ply = PlyData.read(BG_PLY)
bg_v = bg_ply["vertex"].data

print("[INFO] Reading object A...")
a_ply = PlyData.read(A_PLY)
a_v = a_ply["vertex"].data

if bg_v.dtype != a_v.dtype:
    print("[WARN] dtype differs")
    print("BG dtype:", bg_v.dtype.names)
    print("A dtype :", a_v.dtype.names)
    raise RuntimeError("Background and A Gaussian fields are inconsistent.")

bg_xyz = np.stack([bg_v["x"], bg_v["y"], bg_v["z"]], axis=1).astype(np.float32)
a_xyz = np.stack([a_v["x"], a_v["y"], a_v["z"]], axis=1).astype(np.float32)

# 背景中心，用分位数中位数更稳
bg_center = np.median(bg_xyz, axis=0)

# A 自身中心与尺度
a_center = np.median(a_xyz, axis=0)
a_xyz_centered = a_xyz - a_center
a_span = np.quantile(a_xyz_centered, 0.98, axis=0) - np.quantile(a_xyz_centered, 0.02, axis=0)
a_long = float(np.max(a_span))
scale = TARGET_SIZE / max(a_long, 1e-6)

place = bg_center + OFFSET

print("[INFO] bg_center:", bg_center.tolist())
print("[INFO] a_center :", a_center.tolist())
print("[INFO] a_span   :", a_span.tolist())
print("[INFO] scale    :", scale)
print("[INFO] place    :", place.tolist())

# 变换 A 的 xyz
a_new = a_v.copy()
a_new_xyz = a_xyz_centered * scale + place
a_new["x"] = a_new_xyz[:, 0]
a_new["y"] = a_new_xyz[:, 1]
a_new["z"] = a_new_xyz[:, 2]

# 3DGS 的 scale_0/1/2 是 log scale，所以 uniform 缩放要加 log(scale)
log_s = np.log(scale)
for k in ["scale_0", "scale_1", "scale_2"]:
    if k in a_new.dtype.names:
        a_new[k] = a_new[k] + log_s

# 合并
merged = np.empty(len(bg_v) + len(a_new), dtype=bg_v.dtype)
merged[:len(bg_v)] = bg_v
merged[len(bg_v):] = a_new

print("[INFO] background points:", len(bg_v))
print("[INFO] object A points  :", len(a_new))
print("[INFO] merged points    :", len(merged))

PlyData([PlyElement.describe(merged, "vertex")], text=False).write(OUT_PLY)

print("[DONE] merged model:", OUT_MODEL)
print("[DONE] merged ply  :", OUT_PLY)
