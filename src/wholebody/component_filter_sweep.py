# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "pandas"]
# ///
"""Sweep a minimum connected-component area, dropping specks before computing the whole body
bbox, since a plain pixel threshold lets single-pixel noise stretch the bbox to nearly the
full scanner canvas (src/wholebody/naive_threshold_experiment.py found this directly: 3 of 5
inspected ids had their bbox's own left/right edge set by a 1px speck).

Same real check as the naive experiment: does the resulting bbox still contain the union of
this project's own already-recovered region boxes.
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
MIN_AREAS = [0, 5, 10, 20, 50, 100]

df = pd.read_csv(BB_CSV)
df = df[df["component"].str.endswith(VIEW)]
region_union = df.groupby("id").apply(
    lambda g: (g["x"].min(), g["y"].min(), (g["x"] + g["width"]).max(), (g["y"] + g["height"]).max()),
    include_groups=False,
).to_dict()

ids = random.Random(0).sample(sorted(region_union), N)
images = {i: np.asarray(Image.open(RAW / f"wholeBody{VIEW}" / f"{i}.jpg")) for i in ids}
masks = {i: background_mask(images[i], threshold=2) > 0 for i in ids}
components = {}
for i in ids:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(masks[i].astype(np.uint8), connectivity=8)
    components[i] = (labels, stats)

print(f"{'min_area':>9} {'n_contains':>11} {'median_coverage':>16} {'median_width_frac':>18}")
for min_area in MIN_AREAS:
    n_contains = 0
    coverages, width_fracs = [], []
    for i in ids:
        labels, stats = components[i]
        keep = np.zeros(stats.shape[0], dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area  # label 0 is background
        kept_mask = keep[labels]
        ys, xs = np.where(kept_mask)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        rx0, ry0, rx1, ry1 = region_union[i]
        contains = x0 <= rx0 and y0 <= ry0 and x1 >= rx1 and y1 >= ry1
        n_contains += contains
        coverages.append(kept_mask.mean())
        width_fracs.append((x1 - x0) / images[i].shape[1])

    print(f"{min_area:>9} {n_contains:>11}/{N} {np.median(coverages):>15.1%} {np.median(width_fracs):>17.1%}")
