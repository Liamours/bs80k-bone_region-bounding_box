# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "matplotlib", "openpyxl"]
# ///
"""Flag positions far from the typical (x, y) for that component, not a quality metric,
the position itself. A predicted box sitting somewhere no other patient's box sits is a
plausible sign of a wrong placement, worth checking against match quality separately rather
than assuming.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BB_DIR = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb")
OUT_FIGURE = Path(__file__).resolve().parents[2] / "result" / "figures" / "position_outliers.png"
OUT_TABLE = Path(__file__).resolve().parents[2] / "result" / "tables" / "position_outliers.xlsx"
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


df = pd.read_csv(BB_DIR / "bounding_boxes.csv")

flagged = []
for name, g in df.groupby("component"):
    dist = np.sqrt((g["x"] - g["x"].median()) ** 2 + (g["y"] - g["y"].median()) ** 2).to_numpy()
    z = robust_z(dist)
    part = g[["component", "id", "x", "y", "match_score", "near_exact_fraction", "ssim"]].copy()
    part["distance_from_median"] = dist
    part["robust_z"] = z
    part["outlier"] = z > Z_THRESHOLD
    flagged.append(part)

result = pd.concat(flagged, ignore_index=True)
outliers = result[result["outlier"]].sort_values(["component", "robust_z"], ascending=[True, False])

summary = result.groupby("component").agg(
    n=("id", "count"),
    outliers=("outlier", "sum"),
).reset_index()
summary["outlier_fraction"] = summary["outliers"] / summary["n"]

OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT_TABLE) as writer:
    result.to_excel(writer, sheet_name="all_samples", index=False)
    outliers.to_excel(writer, sheet_name="outliers_only", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

fig, axes = plt.subplots(4, 7, figsize=(21, 12))
for ax, name in zip(axes.flat, FOLDERS):
    g = result[result["component"] == name]
    normal, bad = g[~g["outlier"]], g[g["outlier"]]
    ax.scatter(normal["x"], normal["y"], s=3, c="0.5", alpha=0.5)
    ax.scatter(bad["x"], bad["y"], s=10, c="red")
    ax.invert_yaxis()
    ax.set_title(name, fontsize=8)
    ax.tick_params(labelsize=6)
for ax in axes.flat[len(FOLDERS):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_FIGURE, dpi=200)

print(f"{len(outliers)} of {len(result)} flagged as positional outliers")
print(summary.to_string(index=False))
print()
print("outlier mean near_exact_fraction:", outliers["near_exact_fraction"].mean())
print("overall mean near_exact_fraction:", result["near_exact_fraction"].mean())
print(f"saved {OUT_TABLE}")
print(f"saved {OUT_FIGURE}")
