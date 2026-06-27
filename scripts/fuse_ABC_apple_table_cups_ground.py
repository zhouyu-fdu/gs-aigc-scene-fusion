from pathlib import Path
import shutil
import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")

BG_MODEL = ROOT / "outputs/background_3dgs/garden_30000"
BG_PLY = BG_MODEL / "point_cloud/iteration_30000/point_cloud.ply"

A_PLY = ROOT / "final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply"
B_PLY = ROOT / "final_assets/for_fusion/object_B/object_B_gaussian_asset.ply"
C_PLY = ROOT / "final_assets/for_fusion/object_C/object_C_gaussian_asset.ply"

OUT_MODEL = ROOT / "outputs/fusion_scene/garden_with_ABC_ground_cups"
OUT_PLY = OUT_MODEL / "point_cloud/iteration_30000/point_cloud.ply"

# 你前面已经验证过的桌面点，z 约等于桌面高度
TABLE_XYZ = np.array([-0.17198236, 1.46132815, 1.10103631], dtype=np.float32)

# x/y 是水平平面，z 是高度。
# 苹果放桌面；两个杯子放桌子左右两边的地面。
APPLE_ANCHOR_XY = TABLE_XYZ[:2] + np.array([0.06, -0.10], dtype=np.float32)
LEFT_CUP_XY    = TABLE_XYZ[:2] + np.array([-0.78, -0.20], dtype=np.float32)
RIGHT_CUP_XY   = TABLE_XYZ[:2] + np.array([ 0.78,  0.10], dtype=np.float32)

def load_v(path):
    return PlyData.read(str(path))["vertex"].data.copy()

def save_v(path, v):
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))

def xyz_of(v):
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

def set_xyz(v, xyz):
    v["x"] = xyz[:, 0]
    v["y"] = xyz[:, 1]
    v["z"] = xyz[:, 2]

def estimate_surface_z(bg_xyz, xy, radius=0.20, q=0.08, fallback=None):
    """
    在背景点云中，找 xy 附近的点，取较低分位数作为地面高度。
    q 取低分位数，是为了尽量避开植物/桌腿/杂点，贴近地面。
    """
    d = np.linalg.norm(bg_xyz[:, :2] - xy[None, :], axis=1)
    mask = d < radius

    if mask.sum() < 200:
        mask = d < radius * 1.8

    if mask.sum() < 50:
        if fallback is None:
            return float(np.quantile(bg_xyz[:, 2], 0.05))
        return float(fallback)

    z = bg_xyz[mask, 2]
    return float(np.quantile(z, q))

def euler_to_matrix(rx_deg=0, ry_deg=0, rz_deg=0):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    return Rz @ Ry @ Rx

def normalize_bottom(v, rot_deg=(0, 0, 0)):
    """
    把物体变成：
    - x/y 中心在 0
    - z 底部在 0
    这样 anchor_z + lift 就是真正的放置高度。
    """
    out = v.copy()
    xyz = xyz_of(out)

    # 先居中再旋转
    xyz = xyz - np.median(xyz, axis=0, keepdims=True)
    R = euler_to_matrix(*rot_deg)
    xyz = xyz @ R.T

    # 去掉少量离群点，避免底部被飞点影响
    med = np.median(xyz, axis=0)
    dist = np.linalg.norm(xyz - med, axis=1)
    mask = dist < np.quantile(dist, 0.995)
    core = xyz[mask] if mask.sum() > 300 else xyz

    xy_center = np.median(core[:, :2], axis=0)
    z_bottom = np.quantile(core[:, 2], 0.02)

    xyz[:, 0] -= xy_center[0]
    xyz[:, 1] -= xy_center[1]
    xyz[:, 2] -= z_bottom

    set_xyz(out, xyz)
    return out

def place_object(v, anchor_xyz, target_size, lift=0.05, rot_deg=(0,0,0),
                 opacity_bias=0.20, sharp_bias=-0.45):
    out = normalize_bottom(v, rot_deg=rot_deg)
    xyz = xyz_of(out)

    dims = xyz.max(axis=0) - xyz.min(axis=0)
    max_dim = float(np.max(dims))
    if max_dim < 1e-8:
        max_dim = 1.0

    scale = float(target_size) / max_dim

    xyz = xyz * scale
    xyz[:, 0] += anchor_xyz[0]
    xyz[:, 1] += anchor_xyz[1]
    xyz[:, 2] += anchor_xyz[2] + lift

    set_xyz(out, xyz)

    # 3DGS scale 是 log-space
    if all(k in out.dtype.names for k in ["scale_0", "scale_1", "scale_2"]):
        delta = np.float32(np.log(scale) + sharp_bias)
        out["scale_0"] = out["scale_0"].astype(np.float32) + delta
        out["scale_1"] = out["scale_1"].astype(np.float32) + delta
        out["scale_2"] = out["scale_2"].astype(np.float32) + delta

    if "opacity" in out.dtype.names:
        out["opacity"] = np.clip(out["opacity"].astype(np.float32) + opacity_bias, -8.0, 8.0)

    return out

def copy_bg_model():
    if OUT_MODEL.exists():
        shutil.rmtree(str(OUT_MODEL))
    shutil.copytree(
        str(BG_MODEL),
        str(OUT_MODEL),
        ignore=shutil.ignore_patterns("point_cloud")
    )

def main():
    print("[INFO] BG:", BG_PLY)
    print("[INFO] A :", A_PLY)
    print("[INFO] B :", B_PLY)
    print("[INFO] C :", C_PLY)

    bg_v = load_v(BG_PLY)
    a_v = load_v(A_PLY)
    b_v = load_v(B_PLY)
    c_v = load_v(C_PLY)

    bg_xyz = xyz_of(bg_v)

    # 桌面苹果：直接用已知桌面 z
    apple_z = float(TABLE_XYZ[2])
    apple_anchor = np.array([APPLE_ANCHOR_XY[0], APPLE_ANCHOR_XY[1], apple_z], dtype=np.float32)

    # 地面杯子：从背景点云估计左右地面 z
    # 如果估计失败，fallback 用桌面高度减 0.65，大致落到桌子下方地面区域。
    fallback_ground_z = float(TABLE_XYZ[2] - 0.65)
    left_ground_z = estimate_surface_z(bg_xyz, LEFT_CUP_XY, radius=0.25, q=0.08, fallback=fallback_ground_z)
    right_ground_z = estimate_surface_z(bg_xyz, RIGHT_CUP_XY, radius=0.25, q=0.08, fallback=fallback_ground_z)

    left_anchor = np.array([LEFT_CUP_XY[0], LEFT_CUP_XY[1], left_ground_z], dtype=np.float32)
    right_anchor = np.array([RIGHT_CUP_XY[0], RIGHT_CUP_XY[1], right_ground_z], dtype=np.float32)

    print("[INFO] apple anchor:", apple_anchor.tolist())
    print("[INFO] left cup anchor:", left_anchor.tolist())
    print("[INFO] right cup anchor:", right_anchor.tolist())

    # 尺寸故意放大，先保证能看见
    # A/C 放地上，所以可以比桌面版更大一点。
    A_PLACED = place_object(
        a_v,
        anchor_xyz=left_anchor,
        target_size=0.52,
        lift=0.10,
        rot_deg=(0, 0, 0),
        opacity_bias=0.25,
        sharp_bias=-0.35,
    )

    B_PLACED = place_object(
        b_v,
        anchor_xyz=apple_anchor,
        target_size=0.20,
        lift=0.06,
        rot_deg=(0, 0, 0),
        opacity_bias=0.35,
        sharp_bias=-0.55,
    )

    # C 如果仍然方向不佳，至少在右侧地面上会完整可见。
    # 这里先使用 none，不再做复杂旋转，避免又倒着插进桌子。
    C_PLACED = place_object(
        c_v,
        anchor_xyz=right_anchor,
        target_size=0.55,
        lift=0.12,
        rot_deg=(0, 0, 0),
        opacity_bias=0.30,
        sharp_bias=-0.45,
    )

    assert bg_v.dtype == A_PLACED.dtype == B_PLACED.dtype == C_PLACED.dtype

    merged = np.concatenate([bg_v, A_PLACED, B_PLACED, C_PLACED], axis=0)

    copy_bg_model()
    save_v(OUT_PLY, merged)

    print("[DONE] saved:", OUT_PLY)
    print("[INFO] total points:", len(merged))

if __name__ == "__main__":
    main()
