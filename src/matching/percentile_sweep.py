# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "scikit-image", "pandas", "openpyxl"]
# ///
"""Sweep a per-crop relative mask threshold for chest, shoulder, vertebra, one clean control.

Fixed absolute thresholds either diluted the mask with noise floor or, raised high enough
to fix that, left some crops with no pixels above the cutoff at all (context/method.md).
A percentile of each crop's own pixels keeps a fixed fraction regardless of how bright or
dim that one crop happens to be, so it cannot produce an empty mask the same way.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from crop_match_metrics import compare
from core import locate, percentile_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "tables" / "percentile_sweep.xlsx"
N = 20
PERCENTILES = [50, 70, 80, 90, 95]

FOLDERS = [
    "chestLANT", "chestLPOST", "chestRANT", "chestRPOST",
    "shoLANT", "shoLPOST", "shoRANT", "shoRPOST",
    "vertbraANT", "vertbraPOST",
    "kneeLANT",
]

shared_ids = sorted({int(p.stem) for p in (RAW / "headANT").glob("*.jpg")})
ids = random.Random(0).sample(shared_ids, N)

rows = []
for name in FOLDERS:
    view = "ANT" if name.endswith("ANT") else "POST"
    whole_dir = RAW / f"wholeBody{view}"
    for i in ids:
        crop = np.asarray(Image.open(RAW / name / f"{i}.jpg"))
        whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))
        for p in PERCENTILES:
            mask = percentile_mask(crop, p)
            bg_frac = float((mask == 0).mean())
            m = locate(crop, whole, mask=mask)
            window = whole[m["y"]:m["y"] + m["h"], m["x"]:m["x"] + m["w"]]
            metrics = compare(crop, window)

            rows.append({
                "component": name, "id": i, "percentile": p, "background_fraction": bg_frac,
                "match_score": m["score"], "peak_margin": m["peak_margin"],
                **metrics,
            })

df = pd.DataFrame(rows)

summary = df.groupby(["component", "percentile"]).agg(
    n=("id", "count"),
    background_fraction_mean=("background_fraction", "mean"),
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
