# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "pillow", "pandas", "scikit-learn"]
# ///
"""Full scale whole body bounding box, every wholeBodyANT/wholeBodyPOST id from BS-80K, plus
every whole body id LIBS-160K has that BS-80K does not.

Method already validated at N=150/300 sample scale (context/wholebody_bbox.md): plain
background_mask (threshold 2, src/matching/core.py) plus a 5px pad beat every noise cleanup
attempt tried (component area filter, morphological closing, distance from the main blob), all
of which removed real sparse signal at the body's true extremities along with actual noise.

IsolationForest outlier flag per view (scikit-learn, unsupervised, contamination 0.05), not a
single size cutoff, since context/wholebody_bbox.md's own visual check found both real small
patients and genuinely bad source images in the same flagged tail, a size only rule cannot tell
them apart, a human still has to look, this only narrows down where to look.

LIBS-160K's own wholeANT/wholePOST classification images are byte identical to BS-80K's own
wholeBodyANT/wholeBodyPOST for every one of the 3247 shared ids, and its Abnormal/Normal folder
placement agrees with BS-80K's own txt label 100% of the time, checked directly, not assumed
(context/libs160k.md). Beyond those 3247, LIBS-160K has ~3491 more ids per view BS-80K does not,
real new patients, not a different collection. Those get the same bbox method, one IsolationForest
per view fit across the combined population (BS-80K's own ids plus LIBS-160K's extra ids
together), not two separate models, so outlier scores stay comparable across both sources.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matching"))
from core import background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
LIBS = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw")
BB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb\bounding_boxes.csv")
OUT_DIR = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-wholebody-bb")
PAD = 5
THRESHOLD = 2
FEATURE_COLS = ["coverage", "aspect_ratio", "center_x_offset", "top_margin_frac", "bottom_margin_frac", "width_frac", "height_frac"]


def safe_int(stem: str) -> int | None:
    try:
        return int(stem)
    except ValueError:
        return None  # duplicate-download artifacts like "972(1)", skipped


def libs_only_ids(view: str, bs80k_ids: set[int]) -> dict[int, Path]:
    """id -> image path, for LIBS-160K ids in this view that BS-80K does not have."""
    libs_dir = LIBS / f"whole{view}"
    paths: dict[int, Path] = {}
    for label_dir in ("Abnormal", "Normal"):
        for p in (libs_dir / label_dir).glob("*.jpg"):
            pid = safe_int(p.stem)
            if pid is not None and pid not in bs80k_ids:
                paths[pid] = p
    return paths


def whole_body_bbox(img: np.ndarray) -> dict:
    H, W = img.shape
    mask = background_mask(img, threshold=THRESHOLD) > 0
    ys, xs = np.where(mask)
    if xs.size == 0:
        # no pixel above threshold at all, same fallback convention as the region bbox
        # pipeline's own near-blank crops (context/method.md), full canvas, flagged by caller
        x0, y0, x1, y1 = 0, 0, W - 1, H - 1
        fallback = True
    else:
        x0, y0 = max(0, int(xs.min()) - PAD), max(0, int(ys.min()) - PAD)
        x1, y1 = min(W - 1, int(xs.max()) + PAD), min(H - 1, int(ys.max()) + PAD)
        fallback = False
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
        "fallback": fallback,
    }


def main():
    region_ids = set(pd.read_csv(BB_CSV)["id"].unique())

    all_rows = []
    for view, folder in [("ANT", "wholeBodyANT"), ("POST", "wholeBodyPOST")]:
        bs80k_ids = sorted({int(p.stem) for p in (RAW / folder).glob("*.jpg")})
        extra = libs_only_ids(view, set(bs80k_ids))

        rows = []
        for i in bs80k_ids:
            img = np.asarray(Image.open(RAW / folder / f"{i}.jpg"))
            rows.append({"id": i, "view": view, "source": "bs80k", **whole_body_bbox(img)})
        for i, path in sorted(extra.items()):
            img = np.asarray(Image.open(path).convert("L"))
            rows.append({"id": i, "view": view, "source": "libs160k", **whole_body_bbox(img)})
        df = pd.DataFrame(rows)

        X = df[FEATURE_COLS].to_numpy()
        clf = IsolationForest(random_state=0, contamination=0.05)
        df["outlier"] = clf.fit_predict(X) == -1
        df["anomaly_score"] = -clf.score_samples(X)
        df["has_region_crops"] = df["id"].isin(region_ids) & (df["source"] == "bs80k")

        n_fallback = df["fallback"].sum()
        outlier_no_crops = (~df.loc[df["outlier"], "has_region_crops"]).mean()
        normal_no_crops = (~df.loc[~df["outlier"], "has_region_crops"]).mean()
        print(f"{view}: {len(df)} ids ({len(bs80k_ids)} bs80k + {len(extra)} libs160k-only), "
              f"{n_fallback} fallback (blank), "
              f"{df['outlier'].sum()} outliers ({df['outlier'].mean():.1%}), "
              f"outliers lacking region crops {outlier_no_crops:.1%}, "
              f"non-outliers lacking region crops {normal_no_crops:.1%}")
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "bounding_boxes.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}, {len(out)} rows")


if __name__ == "__main__":
    main()
