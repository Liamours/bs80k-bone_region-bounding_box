# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "matplotlib"]
# ///
"""Preview: a few grounding_qa.jsonl records, region box and hotspot boxes drawn on the
whole body image they came from."""
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
JSONL = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-vqa-grounding\grounding_qa.jsonl")
OUT = Path(__file__).resolve().parents[2] / "result" / "figures" / "grounding_qa_preview.png"
BLUE = np.array([0, 0, 255], dtype=np.float64)
RED = np.array([255, 0, 0], dtype=np.float64)
ALPHA = 0.6

records = [json.loads(l) for l in JSONL.open(encoding="utf-8")]

with_hotspot = [r for r in records if r["hotspots"]]
without = [r for r in records if not r["hotspots"] and r["diagnosis"] == "normal"]

rng = random.Random(0)
picked, seen_regions = [], set()
for r in rng.sample(with_hotspot, len(with_hotspot)):
    if r["region"] not in seen_regions:
        picked.append(r)
        seen_regions.add(r["region"])
    if len(picked) == 3:
        break
for r in rng.sample(without, len(without)):
    if r["region"] not in seen_regions:
        picked.append(r)
        seen_regions.add(r["region"])
    if len(picked) == 5:
        break

fig, axes = plt.subplots(1, len(picked), figsize=(2.8 * len(picked), 10))
for ax, r in zip(axes, picked):
    whole = np.asarray(Image.open(RAW / r["image"]))
    base_rgb = np.stack([whole] * 3, axis=-1).astype(np.float64)

    line_mask = np.zeros((*whole.shape, 3), dtype=np.uint8)
    x, y, w, h = r["bbox"]
    cv2.rectangle(line_mask, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
    for hs in r["hotspots"]:
        hx, hy, hw, hh = hs["bbox"]
        cv2.rectangle(line_mask, (hx, hy), (hx + hw, hy + hh), color=(255, 0, 0), thickness=1)

    outlined = base_rgb.copy()
    on = line_mask.any(axis=-1)
    outlined[on] = (1 - ALPHA) * base_rgb[on] + ALPHA * line_mask[on]

    ax.imshow(outlined.astype(np.uint8))
    ax.axis("off")
    qa_answer = r["qa"][1]["answer"]
    ax.set_title(f"{r['region']} / {r['diagnosis']}\nabnormal? {qa_answer}", fontsize=8)

fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0, wspace=0.05)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"saved {OUT}")
for r in picked:
    print(f"  {r['image']:22s} {r['region']:12s} {r['diagnosis']:8s} hotspots={len(r['hotspots'])}")
