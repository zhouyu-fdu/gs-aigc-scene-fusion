from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

BG_PLY = Path("outputs/background_3dgs/garden_30000/point_cloud/iteration_30000/point_cloud.ply")
B_PLY  = Path("final_assets/for_fusion/object_B/object_B_gaussian_asset.ply")
C_PLY  = Path("final_assets/for_fusion/object_C/object_C_gaussian_asset.ply")

OUT_DIR = Path("outputs/fusion_scene/garden_with_BC_test")
OUT_PLY = OUT_DIR / "point_cloud/iteration_30000/point_cloud.ply"

# 你前面手点桌面得到的大致桌面点
TABLE_XYZ = np.array([-0.17198236, 1.46132815, 1.10103631], dtype=np.float32)

# 先给一个“能看见、便于初步检查”的版本：
# B = 苹果，放在桌面右侧一点
# C = 杯子，放在桌面左侧一点
# 注意：这里不是最终精调版，只是先跑一版看初步结果

B_SCALE = 0.18
C_SCALE = 0.22

B_OFFSET = TABLE_XYZ + np.array([ 0.16, -0.02, 0.03], dtype=np.float32)
C_OFFSET = TABLE_XYZ + np.array([-0.10,  0.03, 0.03], dtype=np.float32)

def load_vertex_array(path):
    ply = PlyData.read(str(path))
    return ply["vertex"].data

def copy_vertex_array(v):
    return np.array(v, dtype=v.dtype)

def get_xyz(v):
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

def set_xyz(v, xyz):
    v["x"] = xyz[:, 0]
    v["y"] = xyz[:, 1]
    v["z"] = xyz[:, 2]

def recenter_bottom(v):
    out = copy_vertex_array(v)
    xyz = get_xyz(out)

    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)

    bottom_center = np.array([
        0.5 * (xyz_min[0] + xyz_max[0]),
        0.5 * (xyz_min[1] + xyz_max[1]),
        xyz_min[2]
    ], dtype=np.float32)

    xyz = xyz - bottom_center
    set_xyz(out, xyz)
    return out

def transform_gaussian(v, scale, offset):
    out = copy_vertex_array(v)
    xyz = get_xyz(out)
    xyz = xyz * scale + offset
    set_xyz(out, xyz)

    # 若有 scale_* 字段，需要同步缩放高斯大小
    names = out.dtype.names
    if all(k in names for k in ["scale_0", "scale_1", "scale_2"]):
        log_s = np.log(scale).astype(np.float32)
        out["scale_0"] = out["scale_0"] + log_s
        out["scale_1"] = out["scale_1"] + log_s
        out["scale_2"] = out["scale_2"] + log_s

    return out

def main():
    bg = load_vertex_array(BG_PLY)
    b  = load_vertex_array(B_PLY)
    c  = load_vertex_array(C_PLY)

    print("[INFO] background points:", len(bg))
    print("[INFO] B points         :", len(b))
    print("[INFO] C points         :", len(c))

    b = recenter_bottom(b)
    c = recenter_bottom(c)

    b = transform_gaussian(b, B_SCALE, B_OFFSET)
    c = transform_gaussian(c, C_SCALE, C_OFFSET)

    # dtype 必须一致
    assert bg.dtype == b.dtype == c.dtype, "PLY dtype mismatch among bg / B / C"

    merged = np.concatenate([bg, b, c], axis=0)

    OUT_PLY.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(str(OUT_PLY))

    print("[DONE] saved merged ply:", OUT_PLY)
    print("[INFO] merged points    :", len(merged))
    print("[INFO] B scale/offset   :", B_SCALE, B_OFFSET.tolist())
    print("[INFO] C scale/offset   :", C_SCALE, C_OFFSET.tolist())

if __name__ == "__main__":
    main()
