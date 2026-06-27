from pathlib import Path
from PIL import Image
import cv2
import numpy as np

base = Path("outputs/object_A_3dgs/object_A_clean_final_render_test")
candidates = [
    base / "test/ours_30000/renders",
    base / "train/ours_30000/renders",
]

render_dir = None
for d in candidates:
    if d.exists() and len(list(d.glob("*.png"))) > 0:
        render_dir = d
        break

if render_dir is None:
    raise RuntimeError("No render images found.")

imgs = sorted(render_dir.glob("*.png"))
first = Image.open(imgs[0]).convert("RGB")
w, h = first.size

out_path = Path("final_assets/object_A_clean/final/object_A_clean_final_render_video.mp4")
out_path.parent.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, 20, (w, h))

for p in imgs:
    im = Image.open(p).convert("RGB")
    if im.size != (w, h):
        im = im.resize((w, h))
    frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    writer.write(frame)

writer.release()

print("[DONE]", out_path)
print("[FRAMES]", len(imgs))
print("[SOURCE]", render_dir)
