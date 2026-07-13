# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "matplotlib"]
# ///
"""Preview: whole body images with every predicted region bounding box drawn on top."""
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core import locate, background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "figures" / "bounding_box_preview.png"
VIEW = "ANT"
N = 5
ALPHA = 1.0
CYAN = np.array([0, 255, 255], dtype=np.float64)

PLAIN_REGIONS = [
    "ankleL", "ankleR", "chestL", "chestR", "elbowL", "elbowR",
    "head", "kneeL", "kneeR", "pelvis", "vertbra",
]
MARGIN_ABOVE = 100
MARGIN_BELOW = 250

shared_ids = sorted({int(p.stem) for p in (RAW / f"head{VIEW}").glob("*.jpg")})
ids = random.Random(0).sample(shared_ids, 20)[:N]

fig, axes = plt.subplots(1, N, figsize=(2.5 * N, 10))
whole_dir = RAW / f"wholeBody{VIEW}"
for ax, i in zip(axes, ids):
    whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))
    base_rgb = np.stack([whole] * 3, axis=-1).astype(np.float64)

    line_mask = np.zeros(whole.shape, dtype=np.uint8)
    for region in PLAIN_REGIONS:
        crop = np.asarray(Image.open(RAW / f"{region}{VIEW}" / f"{i}.jpg"))
        mask = background_mask(crop)
        m = locate(crop, whole, mask=mask)
        cv2.rectangle(line_mask, (m["x"], m["y"]), (m["x"] + m["w"], m["y"] + m["h"]), color=255, thickness=1)

    vert_crop = np.asarray(Image.open(RAW / f"vertbra{VIEW}" / f"{i}.jpg"))
    vert_m = locate(vert_crop, whole, mask=background_mask(vert_crop))
    band_top = max(0, vert_m["y"] - MARGIN_ABOVE)
    band_bottom = min(whole.shape[0], vert_m["y"] + MARGIN_BELOW)
    band = whole[band_top:band_bottom]
    for part in ["shoL", "shoR"]:
        crop = np.asarray(Image.open(RAW / f"{part}{VIEW}" / f"{i}.jpg"))
        mask = background_mask(crop)
        banded = band.shape[0] >= crop.shape[0]
        m = locate(crop, band, y_offset=band_top, mask=mask) if banded else locate(crop, whole, mask=mask)
        cv2.rectangle(line_mask, (m["x"], m["y"]), (m["x"] + m["w"], m["y"] + m["h"]), color=255, thickness=1)

    outlined = base_rgb.copy()
    on = line_mask > 0
    outlined[on] = (1 - ALPHA) * base_rgb[on] + ALPHA * CYAN

    ax.imshow(outlined.astype(np.uint8))
    ax.axis("off")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.02)
print(f"saved {OUT}")
