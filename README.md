# GS-AIGC Scene Fusion

Course project for **Deep Learning and Spatial Intelligence**.



This project implements a multi-source 3D asset generation and scene fusion pipeline. It combines:

- **3D Gaussian Splatting (3DGS)** for real-scene and real-object reconstruction
- **COLMAP** for camera pose estimation from phone video frames
- **threestudio / DreamFusion / SDS** for text-to-3D generation
- **stable-Zero123** for single-image-to-3D generation
- **Gaussian-level scene fusion** for inserting generated assets into a real 3DGS background

The main task is to generate three different 3D assets from different sources and insert them into the Mip-NeRF 360 `garden` scene.

---

## 1. Repository Structure

The repository only contains source code, configuration files, report files, and selected figures. Large datasets, checkpoints, Gaussian PLY assets, rendered videos, and intermediate outputs are intentionally excluded from GitHub due to file-size limits.

Recommended uploaded structure:

```text
gs-aigc-scene-fusion/
├── README.md
├── .gitignore
├── report/
│   ├── 空间智能_期末报告.pdf
│   └── 空间智能_期末报告.tex
├── figures/
│   ├── p1_2_object_A_three_panel.png
│   ├── p1_3_object_B_red_apple_steps_3x3.png
│   ├── p1_4_object_C_zero123_result.png
│   ├── p1_5_garden_background_3dgs.png
│   └── p1_6_fusion_coordinate_and_rendering.jpg
├── scripts/
│   ├── make_p1_2_object_A_three_panel.sh
│   ├── make_p1_3_object_B_3x3.sh
│   ├── make_p1_4_object_C_figure.sh
│   ├── make_p1_5_garden_background.sh
│   ├── convert_obj_to_gaussian_ply.py
│   ├── fuse_ABC_apple_table_cups_ground.py
│   └── make_ground_cups_sheet.py
├── configs/
│   └── example_config_notes.md
└── docs/
    └── asset_paths.md
```

The following local directories were used during experiments but are **not uploaded** to GitHub:

```text
data/
outputs/
final_assets/
logs/
*.ckpt
*.ply
*.obj
*.glb
*.mp4
```

These files are excluded because they are either too large or generated automatically.

---

## 2. Project Overview

The project contains four main parts.

### 2.1 Object A: Real Multi-view Reconstruction

Object A is a real cup captured by a phone video. The pipeline is:

```text
phone video
    -> frame extraction
    -> COLMAP camera pose estimation
    -> 3DGS training
    -> Gaussian point-cloud cleanup
    -> cleaned Object A Gaussian asset
```

Main results:

- Extracted video frames: 105
- Registered COLMAP images: 79
- Sparse points: 4429
- Mean reprojection error: 1.109 px
- 3DGS training iterations: 30000
- Original training PSNR: 36.49 dB
- Cleaned asset: `object_A_clean_final.ply`

The cleaned Gaussian asset removes most table and room-background residuals while preserving the main cup body, handle, top decoration, and spoon.

### 2.2 Object B: Text-to-3D Generation

Object B is generated from a text prompt using threestudio and SDS/DreamFusion.

Prompt:

```text
a red apple, single object, centered, realistic, isolated on a white background
```

The experiment compared:

- 6000 steps
- 10000 steps
- 15000 steps

The final red apple result was exported from threestudio, converted to a Gaussian asset, and used for scene fusion.

### 2.3 Object C: Single-image-to-3D Generation

Object C is generated from one RGBA input image using stable-Zero123.

Pipeline:

```text
single RGBA cup image
    -> stable-Zero123 generation
    -> 3000 / 5000 step comparison
    -> mesh export
    -> Gaussian conversion
```

The final 5000-step result preserves the main cup shape, bear decoration, handle, and spoon, but still contains geometric noise and view-dependent ambiguity because only one input image is provided.

### 2.4 Background Scene: Garden 3DGS

The background scene is the Mip-NeRF 360 `garden` scene, reconstructed using 3DGS.

Main metrics:

- PSNR: 27.34 dB
- SSIM: 0.8568
- LPIPS: 0.1219
- Training iterations: 30000

This background is used as the unified 3D environment for inserting Object A, Object B, and Object C.

---

## 3. Scene Fusion Method

The final fusion uses a Gaussian-level representation.

For each asset, a similarity transform is applied:

```text
x' = s R (x - c) + t
```

where:

- `s` is the scale factor
- `R` is the rotation matrix
- `c` is the object center
- `t` is the target position in the garden scene

For Gaussian assets, the scale fields are stored in log-space. Therefore, when uniformly scaling an asset by `s`, the Gaussian scale parameters are updated as:

```text
scale_0 += log(s)
scale_1 += log(s)
scale_2 += log(s)
```

Finally, the background Gaussian PLY and transformed object Gaussian PLY files are concatenated and rendered using the original 3DGS renderer.

---

## 4. Main Scripts

The following scripts are used to generate report figures and fusion results.

### Figure generation

```text
make_p1_2_object_A_three_panel.sh
```

Generates the Object A reconstruction figure, including:

- sample phone-video frame
- original Object A 3DGS rendering
- cleaned Object A Gaussian asset

```text
make_p1_3_object_B_3x3.sh
```

Generates the Object B text-to-3D comparison figure with 6000 / 10000 / 15000 steps.

```text
make_p1_4_object_C_figure.sh
```

Generates the Object C single-image-to-3D comparison figure with input RGBA image and stable-Zero123 results.

```text
make_p1_5_garden_background.sh
```

Generates the garden background 3DGS reconstruction figure.

### Asset conversion and fusion

```text
convert_obj_to_gaussian_ply.py
```

Converts exported mesh assets from Object B and Object C into Gaussian-compatible PLY assets.

```text
fuse_ABC_apple_table_cups_ground.py
```

Creates the final fusion scene where:

- the red apple is placed on the table
- the two cup assets are placed near the table / ground area to avoid severe table intersection
- all assets are merged into the garden Gaussian scene

```text
make_ground_cups_sheet.py
```

Generates a multi-view preview sheet for the final fusion result.

---

## 5. Example Commands

### 5.1 Generate report figures

```bash
cd /mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion

bash scripts/make_p1_2_object_A_three_panel.sh
bash scripts/make_p1_3_object_B_3x3.sh
bash scripts/make_p1_4_object_C_figure.sh
bash scripts/make_p1_5_garden_background.sh
```

### 5.2 Convert mesh assets to Gaussian assets

```bash
cd /mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion

source /mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh
conda activate gs_splatting

python scripts/convert_obj_to_gaussian_ply.py
```

### 5.3 Fuse A/B/C assets into the garden scene

```bash
cd /mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion

source /mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh
conda activate gs_splatting

python scripts/fuse_ABC_apple_table_cups_ground.py
```

### 5.4 Render the final fused scene

```bash
cd /mnt/data/disk3/zhouyu/projects/gaussian-splatting

source /mnt/data/disk3/zhouyu/miniforge3/etc/profile.d/conda.sh
conda activate gs_splatting

python render.py \
  -m /mnt/data/disk3/zhouyu/projects/gs_aigc_scene_fusion/outputs/fusion_scene/garden_with_ABC_ground_cups \
  --iteration 30000 \
  --skip_train
```

---


## 6. Known Limitations

The final fusion result demonstrates the feasibility of inserting multi-source assets into a 3DGS background, but it still has limitations:

1. The red apple generated from text is over-saturated and does not fully match the lighting and material style of the real garden scene.
2. Some generated cup assets are blurry after mesh-to-Gaussian conversion.
3. Contact between inserted objects and the table or ground is not physically optimized.
4. The current fusion uses manual placement and scale adjustment.
5. A more robust future solution should estimate table or ground planes automatically and optimize object placement using contact and visibility constraints.

---

## 7. Future Work

Future improvements include:

- automatic table / ground plane estimation
- more accurate mesh-to-Gaussian conversion
- better texture sampling from exported threestudio meshes
- hybrid rendering of 3DGS background and mesh assets
- automatic scale and placement search
- re-optimization of AIGC assets into native 3DGS representation

---
