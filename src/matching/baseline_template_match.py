# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "scikit-image", "pandas", "openpyxl"]
# ///
"""Baseline: locate each region crop inside its whole body source via template matching."""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from crop_match_metrics import compare
from core import locate

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "tables" / "baseline_template_match.xlsx"
N = 20

FOLDERS = [
    "ankleLANT", "ankleLPOST", "ankleRANT", "ankleRPOST",
    "chestLANT", "chestLPOST", "chestRANT", "chestRPOST",
    "elbowLANT", "elbowLPOST", "elbowRANT", "elbowRPOST",
    "headANT", "headPOST",
    "kneeLANT", "kneeLPOST", "kneeRANT", "kneeRPOST",
    "pelvisANT", "pelvisPOST",
    "shoLANT", "shoLPOST", "shoRANT", "shoRPOST",
    "vertbraANT", "vertbraPOST",
]

shared_ids = sorted({int(p.stem) for p in (RAW / FOLDERS[0]).glob("*.jpg")})
ids = random.Random(0).sample(shared_ids, N)

rows = []
for name in FOLDERS:
    view = "ANT" if name.endswith("ANT") else "POST"
    whole_dir = RAW / f"wholeBody{view}"
    for i in ids:
        crop = np.asarray(Image.open(RAW / name / f"{i}.jpg"))
        whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))

        m = locate(crop, whole)
        window = whole[m["y"]:m["y"] + m["h"], m["x"]:m["x"] + m["w"]]
        metrics = compare(crop, window)

        rows.append({
            "component": name, "id": i, "x": m["x"], "y": m["y"],
            "match_score": m["score"], "peak_margin": m["peak_margin"],
            **metrics,
        })

df = pd.DataFrame(rows)

summary = df.groupby("component").agg(
    n=("id", "count"),
    match_score_mean=("match_score", "mean"),
    peak_margin_mean=("peak_margin", "mean"),
    near_exact_fraction_mean=("near_exact_fraction", "mean"),
    ssim_mean=("ssim", "mean"),
    pearson_corr_mean=("pearson_corr", "mean"),
).reset_index()

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as writer:
    df.to_excel(writer, sheet_name="per_sample", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"saved {OUT}, {len(df)} matches")
print(summary.to_string(index=False))
