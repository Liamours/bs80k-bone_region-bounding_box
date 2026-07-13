# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "pandas", "scipy"]
# ///
"""Neither plain area filtering nor morphological closing cleanly separated real sparse
extremity signal (fingertips, toes) from actual noise specks, closing had to grow so large to
recover containment that it stopped being tighter than the raw box (component_filter_sweep.py,
closing_sweep.py).

Try distance from the main body blob instead of size: real anatomy sits close to the main
mass (a wrist to a hand is a short gap), true noise is scattered arbitrarily and is not
reliably close to anything. Keep the largest component plus any other component within a
given distance of it, everything else is dropped regardless of its own size.
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
DISTANCES = [5, 10, 15, 20, 25, 30]

df = pd.read_csv(BB_CSV)
df = df[df["component"].str.endswith(VIEW)]
region_union = df.groupby("id").apply(
    lambda g: (g["x"].min(), g["y"].min(), (g["x"] + g["width"]).max(), (g["y"] + g["height"]).max()),
    include_groups=False,
).to_dict()

ids = random.Random(0).sample(sorted(region_union), N)
images = {i: np.asarray(Image.open(RAW / f"wholeBody{VIEW}" / f"{i}.jpg")) for i in ids}
masks = {i: background_mask(images[i], threshold=2) > 0 for i in ids}

precomputed = {}
for i in ids:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(masks[i].astype(np.uint8), connectivity=8)
    main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = (labels == main_label).astype(np.uint8)
    dist = cv2.distanceTransform(1 - main_mask, cv2.DIST_L2, 5)
    precomputed[i] = (labels, stats, main_label, dist)

print(f"{'max_dist':>9} {'n_contains':>11} {'median_coverage':>16} {'median_width_frac':>18}")
for max_dist in DISTANCES:
    n_contains = 0
    coverages, width_fracs = [], []
    for i in ids:
        labels, stats, main_label, dist = precomputed[i]
        keep = np.zeros(stats.shape[0], dtype=bool)
        keep[main_label] = True
        for lbl in range(1, stats.shape[0]):
            if lbl == main_label:
                continue
            comp_pixels = labels == lbl
            if dist[comp_pixels].min() <= max_dist:
                keep[lbl] = True
        kept = keep[labels]
        ys, xs = np.where(kept)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        rx0, ry0, rx1, ry1 = region_union[i]
        contains = x0 <= rx0 and y0 <= ry0 and x1 >= rx1 and y1 >= ry1
        n_contains += contains
        coverages.append(kept.mean())
        width_fracs.append((x1 - x0) / images[i].shape[1])

    print(f"{max_dist:>9} {n_contains:>11}/{N} {np.median(coverages):>15.1%} {np.median(width_fracs):>17.1%}")
