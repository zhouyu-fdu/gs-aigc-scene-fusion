from pathlib import Path
import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814

BG_PLY = Path("outputs/background_3dgs/garden_30000/point_cloud/iteration_30000/point_cloud.ply")

TASKS = [
    {
        "name": "object_B",
        "obj": Path("final_assets/for_fusion/object_B/mesh/object_B_red_apple.obj"),
        "out": Path("final_assets/for_fusion/object_B/object_B_gaussian_asset.ply"),
        "num_points": 50000,
        "rgb": np.array([0.90, 0.04, 0.03], dtype=np.float32),
    },
    {
        "name": "object_C",
        "obj": Path("final_assets/for_fusion/object_C/mesh/object_C_cup_bear.obj"),
        "out": Path("final_assets/for_fusion/object_C/object_C_gaussian_asset.ply"),
        "num_points": 50000,
        "rgb": np.array([0.55, 0.35, 0.22], dtype=np.float32),
    },
]

def load_mesh(path: Path):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Cannot load mesh: {path}")
    if len(mesh.faces) == 0:
        raise RuntimeError(f"No faces in mesh: {path}")
    return mesh

def sample_mesh(mesh, n: int):
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    pts = pts.astype(np.float32)

    rgb = None

    # 尝试读取 face / vertex color
    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc is not None and len(vc) == len(mesh.vertices):
            vc = vc[:, :3].astype(np.float32) / 255.0
            faces = np.asarray(mesh.faces)
            rgb = vc[faces[face_idx]].mean(axis=1).astype(np.float32)
    except Exception:
        rgb = None

    return pts, rgb

def make_gaussian(points, rgb, ref_dtype, out_path):
    n = len(points)
    data = np.empty(n, dtype=ref_dtype)

    for name in ref_dtype.names:
        data[name] = 0

    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]

    if all(k in ref_dtype.names for k in ["nx", "ny", "nz"]):
        data["nx"] = 0.0
        data["ny"] = 0.0
        data["nz"] = 0.0

    if all(k in ref_dtype.names for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        sh = (rgb - 0.5) / C0
        data["f_dc_0"] = sh[:, 0]
        data["f_dc_1"] = sh[:, 1]
        data["f_dc_2"] = sh[:, 2]

    # 让 pseudo Gaussian 较清晰，不要太糊
    if "opacity" in ref_dtype.names:
        data["opacity"] = 3.0

    for k in ["scale_0", "scale_1", "scale_2"]:
        if k in ref_dtype.names:
            data[k] = -7.2

    if all(k in ref_dtype.names for k in ["rot_0", "rot_1", "rot_2", "rot_3"]):
        data["rot_0"] = 1.0
        data["rot_1"] = 0.0
        data["rot_2"] = 0.0
        data["rot_3"] = 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(out_path))

def main():
    ref_v = PlyData.read(str(BG_PLY))["vertex"].data
    ref_dtype = ref_v.dtype

    for task in TASKS:
        print("=" * 80)
        print("[TASK]", task["name"])
        print("[OBJ ]", task["obj"])

        mesh = load_mesh(task["obj"])
        print("[INFO] vertices:", len(mesh.vertices))
        print("[INFO] faces   :", len(mesh.faces))
        print("[INFO] bounds  :", mesh.bounds.tolist())

        pts, rgb = sample_mesh(mesh, task["num_points"])

        if rgb is None:
            rgb = np.tile(task["rgb"][None, :], (len(pts), 1)).astype(np.float32)

        rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)

        print("[INFO] sampled points:", pts.shape)
        print("[INFO] rgb mean:", rgb.mean(axis=0).tolist())

        make_gaussian(pts, rgb, ref_dtype, task["out"])
        print("[DONE]", task["out"])

if __name__ == "__main__":
    main()
