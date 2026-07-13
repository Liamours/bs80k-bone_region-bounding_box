# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "pandas"]
# ///
"""Does LIBS-160K's region-crop tsv/jsonl layer (arbitrary image_id, not tied to a patient id)
cover the 3491-per-view patients LIBS-160K has that BS-80K does not (context/wholebody_bbox.md)?

No id mapping exists between the tsv's image_id and a real patient id, so the only way to check
is the same thing this project has always done: try to locate the actual crop pixels inside a
whole body candidate by template matching, and see if a real match exists. A back of envelope
count already suggested yes (192456 tsv rows is close to 6586 patients x 26 region/view combos,
far more than BS-80K's own 2925 patients x 26 would produce), this checks it directly, on
pixels, not just row counts.

Picks one crop from 5 different regions, searches each against the full 13476 candidate whole
body pool (both sources, both views, bs80k-wholebody-bb/bounding_boxes.csv), reports the best
match. ~22ms per candidate, ~5 minutes per crop, background this.
"""
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matching"))
from core import locate, background_mask

LIBS = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw\LIBS-160K-EN")
RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
WB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-wholebody-bb\bounding_boxes.csv")

TEST_REGIONS = ["head", "pelvis", "left ankle joint", "right knee joint", "vertebrae"]


def pick_test_crops() -> dict[str, tuple[int, np.ndarray]]:
    texts = [json.loads(l) for l in (LIBS / "train" / "train_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    lines = (LIBS / "train" / "train_imgs.tsv").read_text(encoding="utf-8").splitlines()
    by_id = {}
    for line in lines:
        image_id, b64 = line.split("\t")
        by_id[int(image_id)] = b64

    picked = {}
    for region in TEST_REGIONS:
        for t in texts:
            if region in t["text"].lower() and t["text"].lower().startswith("this is an image"):
                first_id = t["image_ids"][0]
                crop = np.asarray(Image.open(BytesIO(base64.b64decode(by_id[first_id]))))
                picked[region] = (first_id, crop)
                break
    return picked


def main():
    wb = pd.read_csv(WB_CSV)
    candidates = []  # (id, view, source, image array)
    for _, row in wb.iterrows():
        if row["source"] == "bs80k":
            path = RAW / f"wholeBody{row['view']}" / f"{int(row['id'])}.jpg"
        else:
            label = "Abnormal"  # try both below, cheap existence check
            libs_dir = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw") / f"whole{row['view']}"
            p_abn = libs_dir / "Abnormal" / f"{int(row['id'])}.jpg"
            p_norm = libs_dir / "Normal" / f"{int(row['id'])}.jpg"
            path = p_abn if p_abn.exists() else p_norm
        candidates.append((int(row["id"]), row["view"], row["source"], path))

    print(f"{len(candidates)} whole body candidates to search per test crop")

    test_crops = pick_test_crops()
    print(f"testing {len(test_crops)} region crops: {list(test_crops.keys())}\n")

    for region, (image_id, crop) in test_crops.items():
        mask = background_mask(crop)
        best = None
        t0 = time.time()
        for cid, view, source, path in candidates:
            whole = np.asarray(Image.open(path).convert("L"))
            m = locate(crop, whole, mask=mask)
            if best is None or m["score"] > best[0]["score"]:
                best = (m, cid, view, source)
        dt = time.time() - t0
        m, cid, view, source = best
        print(f"[{region}] tsv image_id={image_id}: best match id={cid} view={view} source={source} "
              f"score={m['score']:.4f} peak_margin={m['peak_margin']:.4f} ({dt:.0f}s)")


if __name__ == "__main__":
    main()
