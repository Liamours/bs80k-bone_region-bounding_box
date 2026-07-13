# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "pandas"]
# ///
"""Plain area filtering (component_filter_sweep.py) made containment much worse, 15/20 down
to 3/20, because sparse low-count signal at the body's true extremities (fingertips, toes,
top of the skull) forms small, disconnected components too, the same way real signal does,
size alone cannot tell real sparse anatomy apart from a stray noise pixel.

Try morphological closing first, bridging small real gaps so genuine extremities merge into
a larger component, before the same area filter, which should then mostly catch true isolated
noise instead.
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matching"))
from core import background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
BB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb\bounding_boxes.csv")
VIEW = "ANT"
N = 20
KERNELS = [3, 5, 7, 9, 11]
MIN_AREA = 5

df = pd.read_csv(BB_CSV)
df = df[df["component"].str.endswith(VIEW)]
region_union = df.groupby("id").apply(
    lambda g: (g["x"].min(), g["y"].min(), (g["x"] + g["width"]).max(), (g["y"] + g["height"]).max()),
    include_groups=False,
).to_dict()

ids = random.Random(0).sample(sorted(region_union), N)
images = {i: np.asarray(Image.open(RAW / f"wholeBody{VIEW}" / f"{i}.jpg")) for i in ids}
masks = {i: background_mask(images[i], threshold=2) > 0 for i in ids}

print(f"{'kernel':>7} {'n_contains':>11} {'median_coverage':>16} {'median_width_frac':>18}")
for k in KERNELS:
    kernel = np.ones((k, k), np.uint8)
    n_contains = 0
    coverages, width_fracs = [], []
    for i in ids:
        closed = cv2.morphologyEx(masks[i].astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        keep = np.zeros(stats.shape[0], dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA
        kept = keep[labels]
        ys, xs = np.where(kept)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        rx0, ry0, rx1, ry1 = region_union[i]
        contains = x0 <= rx0 and y0 <= ry0 and x1 >= rx1 and y1 >= ry1
        n_contains += contains
        coverages.append(kept.mean())
        width_fracs.append((x1 - x0) / images[i].shape[1])

    print(f"{k:>7} {n_contains:>11}/{N} {np.median(coverages):>15.1%} {np.median(width_fracs):>17.1%}")
