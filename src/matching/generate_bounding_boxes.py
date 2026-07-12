# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "scikit-image", "pandas"]
# ///
"""Predict a bounding box for every region crop in the full dataset, masked search and
masked evaluation throughout (context/method.md), the fair, corrected approach for 24 of
26 region types, shoulder is the one folder still known to be unreliable.

Output is the project's deliverable, not an internal analysis table, see intent.md and
context/dataset.md's Locations section, so it goes to bs80k-bone_region-bb, not result/.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from crop_match_metrics import compare
from core import locate, background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT_DIR = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb")

REGIONS = [
    "ankleL", "ankleR", "chestL", "chestR", "elbowL", "elbowR",
    "head", "kneeL", "kneeR", "pelvis", "shoL", "shoR", "vertbra",
]

shared_ids = sorted({int(p.stem) for p in (RAW / "headANT").glob("*.jpg")})

t0 = time.time()
rows = []
for view in ["ANT", "POST"]:
    whole_dir = RAW / f"wholeBody{view}"
    for n, i in enumerate(shared_ids):
        whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))
        for region in REGIONS:
            name = f"{region}{view}"
            crop = np.asarray(Image.open(RAW / name / f"{i}.jpg"))
            mask = background_mask(crop)

            m = locate(crop, whole, mask=mask)
            window = whole[m["y"]:m["y"] + m["h"], m["x"]:m["x"] + m["w"]]
            metrics = compare(crop, window, mask=mask)

            rows.append({
                "component": name, "id": i,
                "x": m["x"], "y": m["y"], "width": m["w"], "height": m["h"],
                "match_score": m["score"], "peak_margin": m["peak_margin"],
                **metrics,
            })
        if (n + 1) % 500 == 0:
            print(f"{view}: {n + 1}/{len(shared_ids)} ids, {time.time() - t0:.0f}s elapsed", flush=True)

df = pd.DataFrame(rows)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "bounding_boxes.csv"
df.to_csv(out_path, index=False)

summary = df.groupby("component").agg(
    n=("id", "count"),
    match_score_mean=("match_score", "mean"),
    peak_margin_mean=("peak_margin", "mean"),
    near_exact_fraction_mean=("near_exact_fraction", "mean"),
    ssim_mean=("ssim", "mean"),
).reset_index()
summary_path = OUT_DIR / "bounding_boxes_summary.csv"
summary.to_csv(summary_path, index=False)

print(f"saved {out_path}, {len(df)} rows, {time.time() - t0:.0f}s total")
print(summary.to_string(index=False))
