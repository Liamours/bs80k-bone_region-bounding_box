# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "matplotlib"]
# ///
"""Preview: whole body images with every predicted region bounding box drawn on top."""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core import locate, background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "figures" / "bounding_box_preview.png"
VIEW = "ANT"
N = 5

REGIONS = [
    "ankleL", "ankleR", "chestL", "chestR", "elbowL", "elbowR",
    "head", "kneeL", "kneeR", "pelvis", "shoL", "shoR", "vertbra",
]

shared_ids = sorted({int(p.stem) for p in (RAW / f"head{VIEW}").glob("*.jpg")})
ids = random.Random(0).sample(shared_ids, 20)[:N]

fig, axes = plt.subplots(1, N, figsize=(2.5 * N, 10))
whole_dir = RAW / f"wholeBody{VIEW}"
for ax, i in zip(axes, ids):
    whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))
    ax.imshow(whole, cmap="gray")
    for region in REGIONS:
        crop = np.asarray(Image.open(RAW / f"{region}{VIEW}" / f"{i}.jpg"))
        mask = background_mask(crop)
        m = locate(crop, whole, mask=mask)
        ax.add_patch(plt.Rectangle((m["x"], m["y"]), m["w"], m["h"], facecolor="blue", edgecolor="none", alpha=0.5))
    ax.axis("off")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.02)
print(f"saved {OUT}")
