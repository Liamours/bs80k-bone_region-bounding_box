# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""Per component quality metrics table for the recovered bounding boxes."""
from pathlib import Path

import numpy as np
import pandas as pd

BB_DIR = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb")
OUT = Path(__file__).resolve().parents[2] / "result" / "figures" / "bounding_box_quality.md"

df = pd.read_csv(BB_DIR / "bounding_boxes.csv")
df["match_score"] = df["match_score"].replace([np.inf, -np.inf], np.nan)

cols = ["near_exact_fraction", "ssim", "mae", "rmse", "pearson_corr", "hist_intersection", "match_score", "peak_margin"]
summary = df.groupby("component")[cols].mean()

header = "| component | near exact | ssim | mae | rmse | pearson r | hist intersect | match score | peak margin |"
sep = "|---|---|---|---|---|---|---|---|---|"
lines = [header, sep]
for name, row in summary.iterrows():
    lines.append(
        f"| {name} | {row.near_exact_fraction:.4f} | {row.ssim:.4f} | {row.mae:.3f} | {row.rmse:.3f} "
        f"| {row.pearson_corr:.4f} | {row.hist_intersection:.4f} | {row.match_score:.4f} | {row.peak_margin:.4f} |"
    )
table = "\n".join(lines)
print(table)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    "# Bounding box quality metrics, per component\n\n"
    "Mean of each per row pixel metric, 2925 rows per component (`bounding_boxes.csv`). "
    "`match_score` excludes 2 known non finite rows (near blank crops, see context/method.md).\n\n"
    + table + "\n",
    encoding="utf-8",
)
print(f"\nsaved {OUT}")
