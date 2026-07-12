# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "scikit-image", "pandas", "openpyxl"]
# ///
"""Anchor the shoulder search on vertebra's own top edge, not head's bottom edge.

The head to pelvis band tried earlier (context/method.md) made shoulder worse, likely
cutting off the true location. Checked the real relationship first, among shoulder matches
not already flagged as positional outliers, shoulder's y sits a median 13-15 pixels below
vertebra's own top edge, std 18-29 pixels, a tighter relationship than the earlier band
assumed. Vertebra is one of the 24 solved regions, so its own match is a reliable anchor.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from crop_match_metrics import compare
from core import locate, background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "tables" / "vertebra_anchored_shoulder.xlsx"
N = 20
MARGIN_ABOVE = 100
MARGIN_BELOW = 250

shared_ids = sorted({int(p.stem) for p in (RAW / "headANT").glob("*.jpg")})
ids = random.Random(0).sample(shared_ids, N)

rows = []
for view in ["ANT", "POST"]:
    whole_dir = RAW / f"wholeBody{view}"
    for i in ids:
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
                "component": name, "id": i, "x": m["x"], "y": m["y"], "banded": banded,
                "vert_y": vert_m["y"],
                "match_score": m["score"], "peak_margin": m["peak_margin"],
                **metrics,
            })

df = pd.DataFrame(rows)
summary = df.groupby("component").agg(
    n=("id", "count"),
    banded_count=("banded", "sum"),
    match_score_mean=("match_score", "mean"),
    peak_margin_mean=("peak_margin", "mean"),
    near_exact_fraction_mean=("near_exact_fraction", "mean"),
    ssim_mean=("ssim", "mean"),
).reset_index()

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as writer:
    df.to_excel(writer, sheet_name="per_sample", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"saved {OUT}, {len(df)} matches")
print(summary.to_string(index=False))
