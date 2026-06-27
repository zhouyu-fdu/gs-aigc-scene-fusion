from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

# =========================
# paths
# =========================
bg_model = Path("outputs/background_3dgs/garden_30000")
bg_ply = bg_model / "point_cloud/iteration_30000/point_cloud.ply"

obj_ply = Path("final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply")
table_xyz_path = Path("final_assets/fusion_debug/table_pick_xyz.npy")

out_model = Path("outputs/fusion_scene/garden_with_A_on_table")
out_ply = out_model / "point_cloud/iteration_30000/point_cloud.ply"

# =========================
# load table point
# =========================
table_xyz = np.load(table_xyz_path).astype(np.float32)
print("[INFO] table_xyz =", table_xyz.tolist())

# =========================
# read bg
# =========================
bg = PlyData.read(str(bg_ply))
bg_v = bg["vertex"].data

# =========================
# read object A
# =========================
obj = PlyData.read(str(obj_ply))
obj_v = obj["vertex"].data

names = obj_v.dtype.names

x = np.asarray(obj_v["x"], dtype=np.float32)
y = np.asarray(obj_v["y"], dtype=np.float32)
z = np.asarray(obj_v["z"], dtype=np.float32)
xyz = np.stack([x, y, z], axis=1)

# =========================
# estimate object bottom center
# =========================
z_min = np.percentile(z, 1.0)          # 更稳一点，不直接用绝对最小值
bottom_mask = z <= (z_min + 0.01)      # 取底部一薄层
if bottom_mask.sum() < 20:
    bottom_mask = z <= np.percentile(z, 5.0)

bottom_pts = xyz[bottom_mask]
bottom_center = bottom_pts.mean(axis=0)

print("[INFO] object bottom center =", bottom_center.tolist())
print("[INFO] object z_min =", float(z.min()))
print("[INFO] bottom pts =", int(bottom_mask.sum()))

# =========================
# optional scale
# =========================
# 如果你觉得杯子太大/太小，可以改这个 SCALE
SCALE = 0.35

obj_xyz_center = xyz.mean(axis=0)
xyz_scaled = (xyz - obj_xyz_center) * SCALE + obj_xyz_center

# 重新计算缩放后的底部中心
z2 = xyz_scaled[:, 2]
z2_min = np.percentile(z2, 1.0)
bottom_mask2 = z2 <= (z2_min + 0.01)
if bottom_mask2.sum() < 20:
    bottom_mask2 = z2 <= np.percentile(z2, 5.0)

bottom_center2 = xyz_scaled[bottom_mask2].mean(axis=0)

# =========================
# translation: move object bottom center to table point
# =========================
T = table_xyz - bottom_center2

# 可选微调：让它稍微高一点，避免陷入桌面
T[2] += 0.01

xyz_final = xyz_scaled + T

print("[INFO] SCALE =", SCALE)
print("[INFO] translation =", T.tolist())

# =========================
# rebuild object vertex array
# =========================
obj_new = np.empty(len(obj_v), dtype=obj_v.dtype)
for n in names:
    obj_new[n] = obj_v[n]

obj_new["x"] = xyz_final[:, 0]
obj_new["y"] = xyz_final[:, 1]
obj_new["z"] = xyz_final[:, 2]

# =========================
# merge bg + object
# =========================
merged = np.concatenate([bg_v, obj_new])

out_model.mkdir(parents=True, exist_ok=True)
out_ply.parent.mkdir(parents=True, exist_ok=True)

PlyData([PlyElement.describe(merged, "vertex")], text=False).write(str(out_ply))

print("[DONE] saved merged ply to:", out_ply)
print("[INFO] bg points :", len(bg_v))
print("[INFO] obj points:", len(obj_new))
print("[INFO] merged    :", len(merged))
