# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "pillow", "numpy", "pandas"]
# ///
"""Phase 3 of multi-step duplicate detection: verify every phase 2 (perceptual hash,
src/dedup/phase2_phash.py) cross-dataset match against this project's own already-recovered
ground truth location (bounding_boxes.csv), the same way phase 2's own calibration verified its
5 initial examples, not a raw pixel-similarity score.

v1 of this script (git history) compared the LIBS crop resized into bs80k's exact box, rigid,
same position, and found chest averaging a *negative* pearson correlation despite bs80k's own
chest boxes being one of this project's best solved regions. Checked directly before trusting
that number, not after: for the single worst chest case, a full unconstrained whole image search
(core.locate(), no forced position, no resize) landed at the *exact same position* as bs80k's
own already-known box (offset (0, 0)), with a real, unambiguous peak margin (0.091, not the
near-zero-signal `nan` the tiny locally-boxed search produced). The location is right, the raw
pixel correlation just isn't, most likely because bs80k's own copy and LIBS-160K's own copy are
two independently reprocessed versions of conceptually the same image, not byte identical, so a
moderate absolute correlation is the realistic ceiling for a genuine cross-dataset match, not
evidence of a wrong one. Verifying "did the crop land where bs80k's own ground truth already
says it should" is the right question, matching this project's method everywhere else, not "is
raw pixel correlation near 1.0," which was never the right bar for two independently processed
copies.
"""
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matching"))
from core import locate, background_mask

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
LIBS = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw\LIBS-160K-EN")
BB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb\bounding_boxes.csv")
PHASH_JSON = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-dedup\dedup_phase2_phash.json")
OUT_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-dedup\dedup_phase3_verification.csv")


def iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_libs_tsv_lookup() -> dict[str, dict[int, str]]:
    files = {
        "train": LIBS / "train" / "train_imgs.tsv",
        "test": LIBS / "train" / "test_imgs.tsv",
        "valid": LIBS / "valid" / "valid_imgs.tsv",
    }
    out = {}
    for split, path in files.items():
        d = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            iid, b64 = line.split("\t")
            d[int(iid)] = b64
        out[split] = d
    return out


def main():
    print("loading bounding_boxes.csv, phase 2 flags, and LIBS-160K tsv lookups...")
    bb = pd.read_csv(BB_CSV, low_memory=False)
    bb_lookup = {(row["component"], int(row["id"])): (int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"]))
                 for _, row in bb.iterrows()}
    phash = json.loads(PHASH_JSON.read_text())
    flags = phash["flags"]
    tsv_lookup = load_libs_tsv_lookup()
    print(f"{len(flags)} phase 2 flagged pairs to verify")

    whole_cache: dict[tuple[int, str], np.ndarray] = {}

    def get_whole(pid: int, view: str) -> np.ndarray:
        key = (pid, view)
        if key not in whole_cache:
            whole_cache[key] = np.asarray(Image.open(RAW / f"wholeBody{view}" / f"{pid}.jpg"))
        return whole_cache[key]

    rows = []
    for i, f in enumerate(flags):
        region, pid, folder = f["region"], f["bs80k_id"], f["bs80k_folder"]
        view = "ANT" if folder.endswith("ANT") else "POST"
        gt_box = bb_lookup.get((folder, pid))
        if gt_box is None:
            continue
        whole = get_whole(pid, view)

        b64 = tsv_lookup[f["libs160k_split"]].get(f["libs160k_image_id"])
        if b64 is None:
            continue
        libs_crop = np.asarray(Image.open(BytesIO(base64.b64decode(b64))))
        mask = background_mask(libs_crop)
        if not mask.any():
            continue
        m = locate(libs_crop, whole, mask=mask)
        found_box = (m["x"], m["y"], m["w"], m["h"])
        overlap = iou(found_box, gt_box)
        dx = m["x"] - gt_box[0]
        dy = m["y"] - gt_box[1]

        rows.append({
            "region": region, "bs80k_id": pid, "bs80k_folder": folder,
            "libs160k_split": f["libs160k_split"], "libs160k_image_id": f["libs160k_image_id"],
            "hamming_distance": f["hamming_distance"],
            "match_score": m["score"], "peak_margin": m["peak_margin"],
            "iou_vs_ground_truth": overlap, "dx": dx, "dy": dy,
        })
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{len(flags)} verified...")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}, {len(out)} rows")
    print(f"\nIoU vs ground truth: median {out['iou_vs_ground_truth'].median():.3f}, "
          f"mean {out['iou_vs_ground_truth'].mean():.3f}")
    print(f"fraction with IoU >= 0.5 (confidently same location): {(out['iou_vs_ground_truth'] >= 0.5).mean():.1%}")
    print(f"fraction with IoU == 0 (no overlap at all): {(out['iou_vs_ground_truth'] == 0).mean():.1%}")
    print("\nby region, fraction IoU >= 0.5:")
    print(out.groupby("region").apply(lambda g: (g["iou_vs_ground_truth"] >= 0.5).mean(), include_groups=False).to_string())
    print("\nby hamming distance, fraction IoU >= 0.5:")
    print(out.groupby("hamming_distance").apply(lambda g: (g["iou_vs_ground_truth"] >= 0.5).mean(), include_groups=False).to_string())


if __name__ == "__main__":
    main()
