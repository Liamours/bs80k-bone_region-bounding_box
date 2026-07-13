# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "matplotlib", "pandas"]
# ///
"""Visual check on the IsolationForest-flagged outliers: genuinely smaller/differently posed
patient, or a thresholding failure."""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-wholebody-bb\wholebody_ANT_sample.csv")
OUT = Path(__file__).resolve().parents[2] / "result" / "figures" / "wholebody_outlier_preview.png"
VIEW = "ANT"

df = pd.read_csv(CSV)
outliers = df.sort_values("anomaly_score", ascending=False).head(4)
normal = df[~df["outlier"]].sample(1, random_state=0)
picked = pd.concat([outliers, normal])

fig, axes = plt.subplots(1, len(picked), figsize=(2.5 * len(picked), 10))
for ax, (_, r) in zip(axes, picked.iterrows()):
    img = np.asarray(Image.open(RAW / f"wholeBody{VIEW}" / f"{int(r['id'])}.jpg"))
    rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)
    x, y, w, h = int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])
    cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 1)
    ax.imshow(rgb)
    ax.axis("off")
    tag = "OUTLIER" if r["outlier"] else "normal"
    ax.set_title(f"id {int(r['id'])} ({tag})\nheight_frac {r['height_frac']:.2f}\ncenter_x_off {r['center_x_offset']:.1f}", fontsize=7)

fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0.05)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"saved {OUT}")
