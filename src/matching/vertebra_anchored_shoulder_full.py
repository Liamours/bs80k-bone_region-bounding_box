# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "scikit-image", "pandas", "openpyxl"]
# ///
"""Full dataset run of the vertebra anchored shoulder search, same approach as
vertebra_anchored_shoulder.py's 20 id sample, checking whether the positional outlier rate
for shoulder actually drops at full scale, not just the aggregate quality means.
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
OUT = Path(__file__).resolve().parents[2] / "result" / "tables" / "vertebra_anchored_shoulder_full.xlsx"
MARGIN_ABOVE = 100
MARGIN_BELOW = 250
Z_THRESHOLD = 3.5


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


shared_ids = sorted({int(p.stem) for p in (RAW / "headANT").glob("*.jpg")})

t0 = time.time()
rows = []
for view in ["ANT", "POST"]:
    whole_dir = RAW / f"wholeBody{view}"
    for n, i in enumerate(shared_ids):
        whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))

        vert = np.asarray(Image.open(RAW / f"vertbra{view}" / f"{i}.jpg"))
        vert_mask = background_mask(vert)
        vert_m = locate(vert, whole, mask=vert_mask)

        band_top = max(0, vert_m["y"] - MARGIN_ABOVE)
        band_bottom = min(whole.shape[0], vert_m["y"] + MARGIN_BELOW)
        band = whole[band_top:band_bottom]

        for part in ["shoL", "shoR"]:
            name = f"{part}{view}"
            crop = np.asarray(Image.open(RAW / name / f"{i}.jpg"))
            mask = background_mask(crop)
            h = crop.shape[0]

            banded = band.shape[0] >= h
            m = locate(crop, band, y_offset=band_top, mask=mask) if banded else locate(crop, whole, mask=mask)

            window = whole[m["y"]:m["y"] + m["h"], m["x"]:m["x"] + m["w"]]
            metrics = compare(crop, window, mask=mask)

            rows.append({
                "component": name, "id": i, "x": m["x"], "y": m["y"], "width": m["w"], "height": m["h"],
                "banded": banded, "vert_y": vert_m["y"],
                "match_score": m["score"], "peak_margin": m["peak_margin"],
                **metrics,
            })
        if (n + 1) % 500 == 0:
            print(f"{view}: {n + 1}/{len(shared_ids)} ids, {time.time() - t0:.0f}s elapsed", flush=True)

df = pd.DataFrame(rows)

flagged = []
for name, g in df.groupby("component"):
    dist = np.sqrt((g["x"] - g["x"].median()) ** 2 + (g["y"] - g["y"].median()) ** 2).to_numpy()
    z = robust_z(dist)
    part = g.copy()
    part["distance_from_median"] = dist
    part["robust_z"] = z
    part["outlier"] = z > Z_THRESHOLD
    flagged.append(part)
df = pd.concat(flagged, ignore_index=True)

summary = df.groupby("component").agg(
    n=("id", "count"),
    outliers=("outlier", "sum"),
    match_score_mean=("match_score", "mean"),
    peak_margin_mean=("peak_margin", "mean"),
    near_exact_fraction_mean=("near_exact_fraction", "mean"),
    ssim_mean=("ssim", "mean"),
).reset_index()
summary["outlier_fraction"] = summary["outliers"] / summary["n"]

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as writer:
    df.to_excel(writer, sheet_name="per_sample", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"saved {OUT}, {len(df)} rows, {time.time() - t0:.0f}s total")
print(summary.to_string(index=False))
