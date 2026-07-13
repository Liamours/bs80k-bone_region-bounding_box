# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "matplotlib"]
# ///
"""Visual check: is the naive threshold bbox tight around the body, or blown wide by noise?"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matching"))
from core import background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "figures" / "naive_threshold_preview.png"
VIEW = "ANT"
ids = [1731, 1821, 2168, 413, 1267]

fig, axes = plt.subplots(1, len(ids), figsize=(2.5 * len(ids), 10))
for ax, i in zip(axes, ids):
    img = np.asarray(Image.open(RAW / f"wholeBody{VIEW}" / f"{i}.jpg"))
    mask = background_mask(img, threshold=2) > 0
    ys, xs = np.where(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest = sizes.max() if len(sizes) else 0
    n_specks = (sizes < 5).sum()

    rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)
    cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 0, 255), 1)
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(f"id {i}\n{n_components - 1} components, largest {largest}px\n{n_specks} specks <5px", fontsize=7)

fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0.05)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"saved {OUT}")
