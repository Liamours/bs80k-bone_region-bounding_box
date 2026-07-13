# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "matplotlib"]
# ///
"""Preview: 5 whole body images with the recovered whole body bounding box drawn on top, same
style and color as src/matching/preview_bounding_boxes.py."""
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matching"))
from core import background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "figures" / "wholebody_bbox_preview.png"
VIEW = "ANT"
N = 5
ALPHA = 1.0
CYAN = np.array([0, 255, 255], dtype=np.float64)
PAD = 5
THRESHOLD = 2

all_ids = sorted({int(p.stem) for p in (RAW / f"wholeBody{VIEW}").glob("*.jpg")})
ids = random.Random(0).sample(all_ids, 20)[:N]

fig, axes = plt.subplots(1, N, figsize=(2.5 * N, 10))
whole_dir = RAW / f"wholeBody{VIEW}"
for ax, i in zip(axes, ids):
    whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))
    base_rgb = np.stack([whole] * 3, axis=-1).astype(np.float64)

    mask = background_mask(whole, threshold=THRESHOLD) > 0
    ys, xs = np.where(mask)
    H, W = whole.shape
    x0, y0 = max(0, int(xs.min()) - PAD), max(0, int(ys.min()) - PAD)
    x1, y1 = min(W - 1, int(xs.max()) + PAD), min(H - 1, int(ys.max()) + PAD)

    line_mask = np.zeros(whole.shape, dtype=np.uint8)
    cv2.rectangle(line_mask, (x0, y0), (x1, y1), color=255, thickness=1)

    outlined = base_rgb.copy()
    on = line_mask > 0
    outlined[on] = (1 - ALPHA) * base_rgb[on] + ALPHA * CYAN

    ax.imshow(outlined.astype(np.uint8))
    ax.axis("off")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.02)
print(f"saved {OUT}")
