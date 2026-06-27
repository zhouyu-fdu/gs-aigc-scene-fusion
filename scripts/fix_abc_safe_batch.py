from pathlib import Path
import shutil
import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")
OUT_ROOT = ROOT / "outputs/fusion_scene"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BG_MODEL = ROOT / "outputs/background_3dgs/garden_30000"
BG_PLY = BG_MODEL / "point_cloud/iteration_30000/point_cloud.ply"

# 你之前从桌面上点出来的 world 坐标
TABLE_CENTER = np.array(
    [-0.17198236286640167, 1.461328148841858, 1.1010363101959229],
    dtype=np.float32
)

def find_first(patterns):
    for pat in patterns:
        xs = sorted(ROOT.glob(pat))
        if xs:
            return xs[0]
    raise FileNotFoundError(f"Cannot find file with patterns: {patterns}")

A_PLY = find_first([
    "final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply",
    "final_assets/object_A_clean/final/object_A_clean_final.ply",
])

B_PLY = find_first([
    "final_assets/for_fusion/object_B/gaussian/*.ply",
    "final_assets/for_fusion/object_B/*gaussian*.ply",
    "final_assets/for_fusion/object_B/**/*.ply",
])

C_PLY = find_first([
    "final_assets/for_fusion/object_C/gaussian/*.ply",
    "final_assets/for_fusion/object_C/*gaussian*.ply",
    "final_assets/for_fusion/object_C/**/*.ply",
])

print("[INFO] BG =", BG_PLY)
print("[INFO] A  =", A_PLY)
print("[INFO] B  =", B_PLY)
print("[INFO] C  =", C_PLY)

def load_vertex(path):
    return PlyData.read(str(path))["vertex"].data.copy()

def save_vertex(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(str(path))

def euler_to_matrix(rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ], dtype=np.float32)

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float32)

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    return Rz @ Ry @ Rx

def transform_object(rec, anchor_xyz, scale=0.3, rot_deg=(0,0,0), lift_y=0.10):
    out = rec.copy()

    xyz = np.stack([
        out["x"].astype(np.float32),
        out["y"].astype(np.float32),
        out["z"].astype(np.float32),
    ], axis=1)

    # 先居中
    xyz = xyz - np.median(xyz, axis=0, keepdims=True)

    # 旋转
    R = euler_to_matrix(*rot_deg)
    xyz = xyz @ R.T

    # 缩放
    xyz = xyz * scale

    # 再对齐：底部落在桌面上方，x/z 对齐到目标位置
    xyz[:, 0] -= np.median(xyz[:, 0])
    xyz[:, 2] -= np.median(xyz[:, 2])
    xyz[:, 1] -= np.min(xyz[:, 1])   # bottom -> 0

    xyz[:, 0] += anchor_xyz[0]
    xyz[:, 1] += anchor_xyz[1] + lift_y
    xyz[:, 2] += anchor_xyz[2]

    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]

    # Gaussian 尺度也跟着放大/缩小
    log_s = float(np.log(scale))
    for k in ["scale_0", "scale_1", "scale_2"]:
        if k in out.dtype.names:
            out[k] = out[k].astype(np.float32) + log_s

    return out

def merge_scene(scene_name, A_cfg, B_cfg, C_cfg):
    dst_model = OUT_ROOT / scene_name
    if dst_model.exists():
        shutil.rmtree(dst_model)

    shutil.copytree(
        BG_MODEL,
        dst_model,
        ignore=shutil.ignore_patterns("point_cloud"),
        dirs_exist_ok=True
    )

    bg = load_vertex(BG_PLY)
    a0 = load_vertex(A_PLY)
    b0 = load_vertex(B_PLY)
    c0 = load_vertex(C_PLY)

    a = transform_object(a0, **A_cfg)
    b = transform_object(b0, **B_cfg)
    c = transform_object(c0, **C_cfg)

    merged = np.concatenate([bg, a, b, c], axis=0)

    out_ply = dst_model / "point_cloud/iteration_30000/point_cloud.ply"
    save_vertex(out_ply, merged)

    print(f"[DONE] {scene_name}")
    print("       saved:", out_ply)
    print("       points:", len(merged))

# -------------------------
# 固定桌面布局（保底可见版）
# -------------------------
# A: 左侧杯子
A_anchor = TABLE_CENTER + np.array([-0.065, 0.0, -0.005], dtype=np.float32)

# B: 前方苹果
B_anchor = TABLE_CENTER + np.array([ 0.020, 0.0, -0.070], dtype=np.float32)

# C: 右侧杯子
C_anchor = TABLE_CENTER + np.array([ 0.085, 0.0,  0.015], dtype=np.float32)

SCENES = [
    {
        "name": "garden_with_ABC_safe1",
        "A": dict(anchor_xyz=A_anchor, scale=0.34, rot_deg=(0, 0, 0),    lift_y=0.115),
        "B": dict(anchor_xyz=B_anchor, scale=0.18, rot_deg=(0, 0, 0),    lift_y=0.080),
        "C": dict(anchor_xyz=C_anchor, scale=0.34, rot_deg=(180, 0, 0),  lift_y=0.125),
    },
    {
        "name": "garden_with_ABC_safe2",
        "A": dict(anchor_xyz=A_anchor, scale=0.36, rot_deg=(0, 0, 0),    lift_y=0.125),
        "B": dict(anchor_xyz=B_anchor, scale=0.20, rot_deg=(0, 0, 0),    lift_y=0.085),
        "C": dict(anchor_xyz=C_anchor, scale=0.36, rot_deg=(90, 0, 0),   lift_y=0.135),
    },
    {
        "name": "garden_with_ABC_safe3",
        "A": dict(anchor_xyz=A_anchor, scale=0.38, rot_deg=(0, 0, 0),    lift_y=0.135),
        "B": dict(anchor_xyz=B_anchor, scale=0.20, rot_deg=(0, 0, 0),    lift_y=0.090),
        "C": dict(anchor_xyz=C_anchor, scale=0.38, rot_deg=(-90, 0, 0),  lift_y=0.145),
    },
]

for s in SCENES:
    merge_scene(s["name"], s["A"], s["B"], s["C"])

print()
print("[ALL DONE] Candidates:")
for s in SCENES:
    print("  ", OUT_ROOT / s["name"])
