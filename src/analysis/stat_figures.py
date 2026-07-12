# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "openpyxl", "matplotlib"]
# ///
"""Figures from the coverage, image stats, and crop size ratio tables."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
TABLES = BASE / "result" / "tables"
FIGURES = BASE / "result" / "figures"

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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "normal"


def boxplot_by_component(df: pd.DataFrame, columns: list[str], out_name: str) -> None:
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 2.2 * len(columns)))
    axes = [axes] if len(columns) == 1 else axes
    for ax, col in zip(axes, columns):
        data = [df.loc[df["component"] == name, col].dropna() for name in FOLDERS]
        ax.boxplot(data, tick_labels=FOLDERS, showfliers=False)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.05, hspace=0.7)
    fig.savefig(FIGURES / out_name, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


FIGURES.mkdir(parents=True, exist_ok=True)

stats = pd.read_excel(TABLES / "image_stats.xlsx", sheet_name="per_image")
boxplot_by_component(stats, ["height", "width", "area", "pixel_mean", "pixel_std"], "image_stats.png")

ratio = pd.read_excel(TABLES / "crop_size_ratio.xlsx", sheet_name="per_image")
boxplot_by_component(ratio, ["height_ratio", "width_ratio", "area_ratio"], "crop_size_ratio.png")

print(f"saved 2 figures to {FIGURES}")
