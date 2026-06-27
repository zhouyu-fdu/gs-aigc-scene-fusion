from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

# =========================================================
# 路径
# =========================================================
ROOT = Path("/mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion")

BG_PLY = ROOT / "outputs/background_3dgs/garden_30000/point_cloud/iteration_30000/point_cloud.ply"
A_PLY  = ROOT / "final_assets/for_fusion/object_A/object_A_3dgs_clean_final.ply"

# 自动搜索 B / C 的 gaussian ply
def find_first_ply(search_dir: Path):
    cands = sorted(search_dir.rglob("*.ply"))
    # 优先找名字里包含 gaussian 的
    cands2 = [p for p in cands if "gaussian" in str(p).lower()]
    if len(cands2) > 0:
        return cands2[0]
    if len(cands) > 0:
        return cands[0]
    raise FileNotFoundError(f"No .ply found under {search_dir}")

B_PLY = find_first_ply(ROOT / "final_assets/for_fusion/object_B")
C_PLY = find_first_ply(ROOT / "final_assets/for_fusion/object_C")

OUT_DIR = ROOT / "final_assets/fusion_scene/quick_fix_ABC"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PLY = OUT_DIR / "merged_point_cloud.ply"

print("[INFO] BG :", BG_PLY)
print("[INFO] A  :", A_PLY)
print("[INFO] B  :", B_PLY)
print("[INFO] C  :", C_PLY)

# =========================================================
# 桌面锚点（你前面已经点出来的桌面点）
# =========================================================
TABLE_XYZ = np.array([
    -0.17198236286640167,
     1.461328148841858,
     1.1010363101959229
], dtype=np.float32)

# =========================================================
# 读写 PLY
# =========================================================
def read_ply(path: Path):
    ply = PlyData.read(str(path))
    return ply, ply["vertex"].data

def write_ply_like(ref_ply: PlyData, vertex_array, out_path: Path):
    el = PlyElement.describe(vertex_array, "vertex")
    PlyData([el], text=False).write(str(out_path))

# =========================================================
# 工具函数
# =========================================================
def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))

def get_xyz(arr):
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)

def robust_core_mask(arr):
    xyz = get_xyz(arr)
    mask = np.ones(len(arr), dtype=bool)
    if "opacity" in arr.dtype.names:
        alpha = sigmoid(arr["opacity"].astype(np.float32))
        mask = alpha > 0.03
        if mask.sum() < 300:
            mask = np.ones(len(arr), dtype=bool)
    # 再做一个简单的 robust 半径过滤，避免极端离群点影响底部估计
    core = xyz[mask]
    center = np.median(core, axis=0)
    dist = np.linalg.norm(core - center, axis=1)
    thr = np.quantile(dist, 0.995)
    keep_core = dist <= thr
    core_idx = np.where(mask)[0]
    final_mask = np.zeros(len(arr), dtype=bool)
    final_mask[core_idx[keep_core]] = True
    if final_mask.sum() < 300:
        return mask
    return final_mask

def normalize_object(arr):
    """
    把物体归一到：
    - x/y 中心在 0 附近
    - z 的“底部”在 0 附近
    这样放置时更稳，不容易只露个盖子或埋进桌面
    """
    a = arr.copy()
    xyz = get_xyz(a)
    mask = robust_core_mask(a)
    core = xyz[mask]

    cx = np.median(core[:, 0])
    cy = np.median(core[:, 1])

    # 用低分位数估计底部，而不是最小值，避免个别飞点把底部拉得太低
    z0 = np.quantile(core[:, 2], 0.02)

    a["x"] = a["x"] - cx
    a["y"] = a["y"] - cy
    a["z"] = a["z"] - z0
    return a

def transform_object(arr, scale, dx, dy, lift, opacity_bias=0.0):
    a = arr.copy()
    # 先缩放，再放到桌面锚点附近
    a["x"] = a["x"] * scale + TABLE_XYZ[0] + dx
    a["y"] = a["y"] * scale + TABLE_XYZ[1] + dy
    a["z"] = a["z"] * scale + TABLE_XYZ[2] + lift

    # Gaussian 的尺度字段同步加 log(scale)
    for k in ["scale_0", "scale_1", "scale_2"]:
        if k in a.dtype.names:
            a[k] = a[k] + np.log(scale)

    # 稍微提一点 opacity，让物体更容易看清
    if "opacity" in a.dtype.names and abs(opacity_bias) > 1e-8:
        a["opacity"] = np.clip(a["opacity"] + opacity_bias, -8.0, 8.0)

    return a

# =========================================================
# 参数：这是一版“快速修复 + 稍微放大 + 稍微抬高”的保守参数
# =========================================================
# 注意：
# - dx/dy：桌面上的位置
# - lift：整体抬高，防止埋进桌面
# - scale：稍微放大一点，方便看清
SPECS = {
    "A": dict(scale=0.30, dx=-0.14, dy= 0.03, lift=0.085, opacity_bias=0.10),
    "B": dict(scale=0.42, dx= 0.15, dy= 0.02, lift=0.060, opacity_bias=0.18),
    "C": dict(scale=0.36, dx= 0.02, dy=-0.16, lift=0.080, opacity_bias=0.15),
}

# =========================================================
# 执行融合
# =========================================================
bg_ply, bg_arr = read_ply(BG_PLY)
_, a_arr0 = read_ply(A_PLY)
_, b_arr0 = read_ply(B_PLY)
_, c_arr0 = read_ply(C_PLY)

print("[INFO] background points:", len(bg_arr))
print("[INFO] A raw points     :", len(a_arr0))
print("[INFO] B raw points     :", len(b_arr0))
print("[INFO] C raw points     :", len(c_arr0))

a_arr = transform_object(normalize_object(a_arr0), **SPECS["A"])
b_arr = transform_object(normalize_object(b_arr0), **SPECS["B"])
c_arr = transform_object(normalize_object(c_arr0), **SPECS["C"])

merged = np.concatenate([bg_arr, a_arr, b_arr, c_arr], axis=0)

print("[INFO] merged points    :", len(merged))
write_ply_like(bg_ply, merged, OUT_PLY)
print("[DONE] saved to:", OUT_PLY)
