# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "openpyxl", "pillow", "numpy"]
# ///
"""Per image size and pixel stats across the 26 bone region folders."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "analysis" / "image_stats.xlsx"

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

rows = []
for name in FOLDERS:
    for path in (RAW / name).glob("*.jpg"):
        arr = np.asarray(Image.open(path))
        h, w = arr.shape[:2]
        rows.append({
            "id": int(path.stem),
            "component": name,
            "height": h,
            "width": w,
            "area": h * w,
            "pixel_min": int(arr.min()),
            "pixel_max": int(arr.max()),
            "pixel_mean": float(arr.mean()),
            "pixel_median": float(np.median(arr)),
            "pixel_std": float(arr.std()),
        })

df = pd.DataFrame(rows).sort_values(["component", "id"])

summary = df.groupby("component").agg(
    n=("id", "count"),
    height_mean=("height", "mean"),
    width_mean=("width", "mean"),
    area_mean=("area", "mean"),
    pixel_mean=("pixel_mean", "mean"),
    pixel_std_mean=("pixel_std", "mean"),
).reset_index()

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as writer:
    df.to_excel(writer, sheet_name="per_image", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"saved {OUT}, {len(df)} rows")
