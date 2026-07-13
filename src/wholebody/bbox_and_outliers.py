# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "pandas", "scikit-learn"]
# ///
"""Whole body bbox: fixed threshold mask (this project's own background_mask, threshold=2)
plus a 5px safety pad, chosen over every component-filtering variant tried
(component_filter_sweep.py, closing_sweep.py, distance_filter_sweep.py), all of which reduced
containment against this project's own region-box union because sparse real signal at the
body's true extremities (fingertips, toes, scalp) is not reliably separable from noise specks
by size, closing, or distance alone. Padding a plain threshold outperformed all of them: 72.7%
raw containment at N=150 versus 92.7% at pad=5, with no risk of discarding real signal.

Then: multi-metric feature vector per bbox, IsolationForest (unsupervised, sklearn) to flag
outliers, since body size genuinely varies (children vs adults) and a plain size cutoff would
conflate "small patient" with "bad threshold," a multi-feature anomaly model at least has more
than one axis to separate the two on.
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matching"))
from core import background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT_DIR = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-wholebody-bb")
VIEW = "ANT"
N = 300
PAD = 5
THRESHOLD = 2


def whole_body_bbox(img: np.ndarray) -> dict:
    mask = background_mask(img, threshold=THRESHOLD) > 0
    ys, xs = np.where(mask)
    H, W = img.shape
    x0, y0 = max(0, int(xs.min()) - PAD), max(0, int(ys.min()) - PAD)
    x1, y1 = min(W - 1, int(xs.max()) + PAD), min(H - 1, int(ys.max()) + PAD)
    w, h = x1 - x0, y1 - y0
    return {
        "x": x0, "y": y0, "width": w, "height": h,
        "coverage": float(mask.mean()),
        "aspect_ratio": w / h,
        "center_x_offset": (x0 + w / 2) - W / 2,
        "top_margin_frac": y0 / H,
        "bottom_margin_frac": (H - y1) / H,
        "width_frac": w / W,
        "height_frac": h / H,
    }


def main():
    all_ids = sorted({int(p.stem) for p in (RAW / f"wholeBody{VIEW}").glob("*.jpg")})
    ids = random.Random(0).sample(all_ids, N)

    rows = []
    for i in ids:
        img = np.asarray(Image.open(RAW / f"wholeBody{VIEW}" / f"{i}.jpg"))
        rows.append({"id": i, **whole_body_bbox(img)})
    df = pd.DataFrame(rows)

    feature_cols = ["coverage", "aspect_ratio", "center_x_offset", "top_margin_frac", "bottom_margin_frac", "width_frac", "height_frac"]
    X = df[feature_cols].to_numpy()
    clf = IsolationForest(random_state=0, contamination=0.05)
    df["outlier"] = clf.fit_predict(X) == -1
    df["anomaly_score"] = -clf.score_samples(X)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"wholebody_{VIEW}_sample.csv"
    df.to_csv(out_csv, index=False)

    print(f"{N} ids, {df['outlier'].sum()} flagged outliers ({df['outlier'].mean():.1%})")
    print(f"saved {out_csv}")
    print("\n=== feature summary ===")
    print(df[feature_cols].describe().to_string())
    print("\n=== top 10 outliers by anomaly score ===")
    print(df.sort_values("anomaly_score", ascending=False)[["id", *feature_cols, "anomaly_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
