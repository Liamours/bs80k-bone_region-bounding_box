# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "openpyxl", "pillow"]
# ///
"""Region crop size relative to its matching whole body image, by id."""
from pathlib import Path

import pandas as pd
from PIL import Image

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
BASE = Path(__file__).resolve().parents[2]
STATS = BASE / "result" / "tables" / "image_stats.xlsx"
OUT = BASE / "result" / "tables" / "crop_size_ratio.xlsx"


def whole_body_dims(view: str) -> dict[int, tuple[int, int]]:
    folder = RAW / f"wholeBody{view}"
    return {int(p.stem): Image.open(p).size[::-1] for p in folder.glob("*.jpg")}


dims = {"ANT": whole_body_dims("ANT"), "POST": whole_body_dims("POST")}

crops = pd.read_excel(STATS, sheet_name="per_image", usecols=["id", "component", "height", "width"])
crops["view"] = crops["component"].apply(lambda c: "ANT" if c.endswith("ANT") else "POST")
crops["wb_height"] = crops.apply(lambda r: dims[r.view].get(r.id, (None, None))[0], axis=1)
crops["wb_width"] = crops.apply(lambda r: dims[r.view].get(r.id, (None, None))[1], axis=1)

before = len(crops)
crops = crops.dropna(subset=["wb_height", "wb_width"]).copy()
dropped = before - len(crops)

crops["height_ratio"] = crops["height"] / crops["wb_height"]
crops["width_ratio"] = crops["width"] / crops["wb_width"]
crops["area_ratio"] = (crops["height"] * crops["width"]) / (crops["wb_height"] * crops["wb_width"])

summary = crops.groupby("component").agg(
    n=("id", "count"),
    height_ratio_mean=("height_ratio", "mean"),
    height_ratio_std=("height_ratio", "std"),
    width_ratio_mean=("width_ratio", "mean"),
    width_ratio_std=("width_ratio", "std"),
    area_ratio_mean=("area_ratio", "mean"),
    area_ratio_std=("area_ratio", "std"),
).reset_index()

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as writer:
    crops.to_excel(writer, sheet_name="per_image", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"saved {OUT}, {len(crops)} rows, {dropped} dropped for missing whole body match")
