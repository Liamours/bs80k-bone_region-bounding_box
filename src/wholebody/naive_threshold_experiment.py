# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "pandas", "matplotlib"]
# ///
"""First pass: does a single fixed threshold (this project's own background_mask, already used
for region crops) recover a sane whole body bounding box inside the fixed scanner canvas?

The real check available here, no manual annotation needed: this project's own already
recovered region boxes (bounding_boxes.csv) are parts of the body, so the true whole body box
must contain the union of all of them, for a given id and view. A threshold bbox that fails to
contain that union has cut off a real body part, not just been imprecise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matching"))
from core import background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
BB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb\bounding_boxes.csv")
VIEW = "ANT"
N = 20

df = pd.read_csv(BB_CSV)
df = df[df["component"].str.endswith(VIEW)]
region_union = df.groupby("id").apply(
    lambda g: (g["x"].min(), g["y"].min(), (g["x"] + g["width"]).max(), (g["y"] + g["height"]).max()),
    include_groups=False,
).to_dict()

import random
ids = random.Random(0).sample(sorted(region_union), N)

print(f"{'id':>6} {'thresh_bbox':>22} {'region_union':>22} {'contains?':>10} {'coverage':>9}")
n_contains = 0
for i in ids:
    img = np.asarray(Image.open(RAW / f"wholeBody{VIEW}" / f"{i}.jpg"))
    mask = background_mask(img, threshold=2) > 0
    ys, xs = np.where(mask)
    tx0, ty0, tx1, ty1 = xs.min(), ys.min(), xs.max(), ys.max()

    rx0, ry0, rx1, ry1 = region_union[i]
    contains = tx0 <= rx0 and ty0 <= ry0 and tx1 >= rx1 and ty1 >= ry1
    n_contains += contains
    coverage = mask.mean()
    print(f"{i:>6} {f'[{tx0},{ty0},{tx1},{ty1}]':>22} {f'[{rx0},{ry0},{rx1},{ry1}]':>22} {str(contains):>10} {coverage:>9.1%}")

print(f"\n{n_contains}/{N} threshold bboxes contain the full region-box union")
