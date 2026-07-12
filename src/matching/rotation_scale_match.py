# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "scikit-image", "pandas", "openpyxl"]
# ///
"""Try rotation and scale invariant matching for shoulder, the one region masking never fixed.

Plain matchTemplate only searches translation. If shoulder's true content is rotated or
scaled relative to a plain axis aligned crop, no amount of masking would help, this is the
one major remaining explanation left in context/method.md. Tests it directly: try the crop
at several rotations and scales, keep whichever setting and location scores best.
"""
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from crop_match_metrics import compare
from core import locate, background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "tables" / "rotation_scale_match.xlsx"
N = 20
ANGLES = [-15, -10, -5, 0, 5, 10, 15]
SCALES = [0.9, 0.95, 1.0, 1.05, 1.1]

FOLDERS = ["shoLANT", "shoLPOST", "shoRANT", "shoRPOST"]

shared_ids = sorted({int(p.stem) for p in (RAW / "headANT").glob("*.jpg")})
ids = random.Random(0).sample(shared_ids, N)


def transform(img: np.ndarray, mask: np.ndarray, angle: float, scale: float):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img_t = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask_t = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return img_t, mask_t


t0 = time.time()
rows = []
for name in FOLDERS:
    view = "ANT" if name.endswith("ANT") else "POST"
    whole_dir = RAW / f"wholeBody{view}"
    for i in ids:
        crop = np.asarray(Image.open(RAW / name / f"{i}.jpg"))
        whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))
        base_mask = background_mask(crop)

        best = None
        for angle in ANGLES:
            for scale in SCALES:
                crop_t, mask_t = transform(crop, base_mask, angle, scale)
                if mask_t.sum() == 0:
                    continue
                m = locate(crop_t, whole, mask=mask_t)
                if best is None or m["score"] > best["score"]:
                    best = {**m, "angle": angle, "scale": scale, "crop_t": crop_t, "mask_t": mask_t}

        window = whole[best["y"]:best["y"] + best["h"], best["x"]:best["x"] + best["w"]]
        metrics = compare(best["crop_t"], window, mask=best["mask_t"])

        rows.append({
            "component": name, "id": i, "x": best["x"], "y": best["y"],
            "angle": best["angle"], "scale": best["scale"],
            "match_score": best["score"], "peak_margin": best["peak_margin"],
            **metrics,
        })

df = pd.DataFrame(rows)

summary = df.groupby("component").agg(
    n=("id", "count"),
    match_score_mean=("match_score", "mean"),
    peak_margin_mean=("peak_margin", "mean"),
    near_exact_fraction_mean=("near_exact_fraction", "mean"),
    ssim_mean=("ssim", "mean"),
    angle_mode=("angle", lambda s: s.mode().iloc[0]),
    scale_mode=("scale", lambda s: s.mode().iloc[0]),
).reset_index()

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as writer:
    df.to_excel(writer, sheet_name="per_sample", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"saved {OUT}, {len(df)} matches, {time.time() - t0:.0f}s")
print(summary.to_string(index=False))
print()
print("angle distribution:")
print(df["angle"].value_counts().sort_index().to_string())
print()
print("scale distribution:")
print(df["scale"].value_counts().sort_index().to_string())
