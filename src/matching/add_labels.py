# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Join each row's normal/abnormal label from that component's own txt label file.

Every region folder has a txt file of the same name, one line per image, filename and a
0 (normal) or 1 (abnormal) label (context/dataset.md). Adds that label onto the bounding
box dataset so downstream use does not need to separately read 26 label files. Per the
source paper (context/dataset.md), abnormal means a nidus, a suspected malignant finding,
falls inside that region, normal means it does not.
"""
from pathlib import Path

import pandas as pd

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
BB_DIR = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb")

df = pd.read_csv(BB_DIR / "bounding_boxes.csv")

labels = {}
for component in df["component"].unique():
    txt_path = RAW / component / f"{component}.txt"
    for line in txt_path.read_text().splitlines():
        filename, label = line.strip().split("\t")
        labels[(component, int(filename.removesuffix(".jpg")))] = int(label)

df["label"] = [labels.get((c, i)) for c, i in zip(df["component"], df["id"])]
df["diagnosis"] = df["label"].map({0: "normal", 1: "abnormal"})

missing = df["label"].isna().sum()
print(f"{missing} of {len(df)} rows have no matching label")
print(df["diagnosis"].value_counts(dropna=False).to_string())
print()
print(df.groupby("component")["label"].mean().to_string())

out = BB_DIR / "bounding_boxes.csv"
df.to_csv(out, index=False)
print(f"saved {out}, {len(df)} rows, {len(df.columns)} columns")
