# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pillow", "matplotlib", "openpyxl", "opencv-python-headless"]
# ///
"""Flag predictions positioned wrong relative to that patient's own body, not the image frame.

position_outliers.py compares a prediction's raw (x, y) against the median (x, y) for that
component across all patients. This instead centers each patient's own predictions on that
patient's own body centroid first, so a patient scanned slightly higher, lower, or off to one
side in the frame does not get compared against a typical position measured in raw image
coordinates. The centroid also gives, per component, a single typical offset from the body's
own center, the "where does the skull/spine/shoulder/ankle/etc usually sit relative to the
body" table asked for directly.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from core import background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
BB_DIR = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb")
OUT_FIGURE = Path(__file__).resolve().parents[2] / "result" / "figures" / "centroid_outliers.png"
OUT_TABLE = Path(__file__).resolve().parents[2] / "result" / "tables" / "centroid_outliers.xlsx"
Z_THRESHOLD = 3.5  # standard robust outlier cutoff (Iglewicz and Hoaglin)

plt.rcParams["font.family"] = "Times New Roman"

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


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def body_centroid(whole: np.ndarray) -> tuple[float, float]:
    """Center of mass of the body silhouette, not the geometric center of the image frame."""
    mask = background_mask(whole)
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


df = pd.read_csv(BB_DIR / "bounding_boxes.csv")
df["view"] = df["component"].apply(lambda c: "ANT" if c.endswith("ANT") else "POST")
df["region_x"] = df["x"] + df["width"] / 2
df["region_y"] = df["y"] + df["height"] / 2

centroids = {}
for view in ["ANT", "POST"]:
    whole_dir = RAW / f"wholeBody{view}"
    for i in df.loc[df["view"] == view, "id"].unique():
        whole = np.asarray(Image.open(whole_dir / f"{i}.jpg"))
        centroids[(view, i)] = body_centroid(whole)

df["centroid_x"] = [centroids[(v, i)][0] for v, i in zip(df["view"], df["id"])]
df["centroid_y"] = [centroids[(v, i)][1] for v, i in zip(df["view"], df["id"])]
df["offset_x"] = df["region_x"] - df["centroid_x"]
df["offset_y"] = df["region_y"] - df["centroid_y"]

flagged = []
for name, g in df.groupby("component"):
    med_dx, med_dy = g["offset_x"].median(), g["offset_y"].median()
    dist = np.sqrt((g["offset_x"] - med_dx) ** 2 + (g["offset_y"] - med_dy) ** 2).to_numpy()
    z = robust_z(dist)
    part = g[["component", "id", "x", "y", "offset_x", "offset_y", "match_score", "near_exact_fraction", "ssim"]].copy()
    part["distance_from_typical_offset"] = dist
    part["robust_z"] = z
    part["outlier"] = z > Z_THRESHOLD
    flagged.append(part)

result = pd.concat(flagged, ignore_index=True)
outliers = result[result["outlier"]].sort_values(["component", "robust_z"], ascending=[True, False])

typical_offset = df.groupby("component").agg(
    n=("id", "count"),
    typical_offset_x=("offset_x", "median"),
    typical_offset_y=("offset_y", "median"),
    offset_x_spread=("offset_x", lambda s: (s - s.median()).abs().median()),
    offset_y_spread=("offset_y", lambda s: (s - s.median()).abs().median()),
).reset_index()

outlier_summary = result.groupby("component").agg(n=("id", "count"), outliers=("outlier", "sum")).reset_index()
outlier_summary["outlier_fraction"] = outlier_summary["outliers"] / outlier_summary["n"]

OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT_TABLE) as writer:
    typical_offset.to_excel(writer, sheet_name="typical_offset_from_centroid", index=False)
    outlier_summary.to_excel(writer, sheet_name="outlier_summary", index=False)
    outliers.to_excel(writer, sheet_name="outliers_only", index=False)
    result.to_excel(writer, sheet_name="all_samples", index=False)

fig, axes = plt.subplots(4, 7, figsize=(21, 12))
for ax, name in zip(axes.flat, FOLDERS):
    g = result[result["component"] == name]
    normal, bad = g[~g["outlier"]], g[g["outlier"]]
    ax.scatter(normal["offset_x"], normal["offset_y"], s=3, c="0.5", alpha=0.5)
    ax.scatter(bad["offset_x"], bad["offset_y"], s=10, c="red")
    ax.axhline(0, color="0.8", linewidth=0.5)
    ax.axvline(0, color="0.8", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_title(name, fontsize=8)
    ax.tick_params(labelsize=6)
for ax in axes.flat[len(FOLDERS):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_FIGURE, dpi=200)

print("typical offset from body centroid, per component:")
print(typical_offset.round(1).to_string(index=False))
print()
print(f"{len(outliers)} of {len(result)} flagged as centroid-relative outliers")
print(outlier_summary.round(4).to_string(index=False))
print()
print("outlier mean near_exact_fraction:", outliers["near_exact_fraction"].mean())
print("overall mean near_exact_fraction:", result["near_exact_fraction"].mean())
print(f"saved {OUT_TABLE}")
print(f"saved {OUT_FIGURE}")
