# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Note every known duplicate finding directly on bounding_boxes.csv, additive only.

No row is ever dropped or merged, per explicit instruction: these are notes about known
duplication (context/dataset.md, context/libs160k.md), not a cleanup pass. A downstream
consumer decides what, if anything, to do with a flagged row, this script only makes the
existing findings visible on the data itself instead of leaving them sitting only in the
dedup json files and context docs.

Three additive columns:
  - duplicate_of_patient_id: this component's crop is byte identical to another bs80k patient's
    same-component crop, the other patient's id. Only set on the 24 of 26 components where this
    was actually confirmed (phase 1 MD5), elbowLPOST/elbowRPOST excluded, see below.
  - duplicate_of_sibling_component: this row's own crop is byte identical to this same patient's
    OTHER component (elbowLPOST <-> elbowRPOST, 99.7% of patients, phase 1 MD5). Different kind
    of duplicate than the one above, same patient not a different one.
  - libs160k_duplicate_matches: LIBS-160K split/image_id(hamming distance, IoU vs this row's own
    box) this row's crop perceptually matches (phase 2 pHash), semicolon joined, blank if none.
    Filtered to phase 3's own location verification (src/dedup/phase3_verify_phash.py), IoU >=
    0.5 against this row's own already-recovered box, not the raw, unfiltered phase 2 flag list,
    since phase 3 found 12.5% of raw flags do not actually land at the claimed location.
"""
import json
from pathlib import Path

import pandas as pd

BB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb\bounding_boxes.csv")
MD5_JSON = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-vqa-grounding\dedup_phase1_md5.json")
PHASE3_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-vqa-grounding\dedup_phase3_verification.csv")
IOU_MIN = 0.5

md5 = json.loads(MD5_JSON.read_text())
phase3 = pd.read_csv(PHASE3_CSV)
verified = phase3[phase3["iou_vs_ground_truth"] >= IOU_MIN]

# --- cross-patient duplicates: (component, id) -> other patient id ---
bs_dup = {h: locs for h, locs in md5["dup_groups"].items()
          if all(l.startswith("bs80k:") and not l.startswith("bs80k:wholeBody") for l in locs)}
cross_patient: dict[tuple[str, int], int] = {}
for locs in bs_dup.values():
    parsed = []
    for l in locs:
        folder, file = l.replace("bs80k:", "").split("/")
        parsed.append((folder, int(file.replace(".jpg", ""))))
    if len(parsed) == 2 and parsed[0][1] != parsed[1][1]:
        (folder_a, id_a), (folder_b, id_b) = parsed
        assert folder_a == folder_b
        cross_patient[(folder_a, id_a)] = id_b
        cross_patient[(folder_b, id_b)] = id_a

# --- same-patient sibling component duplicates (elbowLPOST <-> elbowRPOST) ---
sibling_component: dict[tuple[str, int], str] = {}
for locs in bs_dup.values():
    parsed = []
    for l in locs:
        folder, file = l.replace("bs80k:", "").split("/")
        parsed.append((folder, int(file.replace(".jpg", ""))))
    if len(parsed) == 2 and parsed[0][1] == parsed[1][1]:
        (folder_a, pid), (folder_b, _) = parsed
        sibling_component[(folder_a, pid)] = folder_b
        sibling_component[(folder_b, pid)] = folder_a

# --- libs160k perceptual-hash cross-dataset matches, phase 3 IoU-verified only ---
libs_matches: dict[tuple[str, int], list[str]] = {}
for _, row in verified.iterrows():
    key = (row["bs80k_folder"], int(row["bs80k_id"]))
    libs_matches.setdefault(key, []).append(
        f"{row['libs160k_split']}/{int(row['libs160k_image_id'])}"
        f"({int(row['hamming_distance'])},iou={row['iou_vs_ground_truth']:.2f})"
    )

df = pd.read_csv(BB_CSV, low_memory=False)
df["duplicate_of_patient_id"] = [cross_patient.get((c, i)) for c, i in zip(df["component"], df["id"])]
df["duplicate_of_sibling_component"] = [sibling_component.get((c, i)) for c, i in zip(df["component"], df["id"])]
df["libs160k_duplicate_matches"] = [
    ";".join(libs_matches[(c, i)]) if (c, i) in libs_matches else None
    for c, i in zip(df["component"], df["id"])
]

n_cross = df["duplicate_of_patient_id"].notna().sum()
n_sibling = df["duplicate_of_sibling_component"].notna().sum()
n_libs = df["libs160k_duplicate_matches"].notna().sum()
print(f"duplicate_of_patient_id set on {n_cross} rows")
print(f"duplicate_of_sibling_component set on {n_sibling} rows")
print(f"libs160k_duplicate_matches set on {n_libs} rows")
print(f"total rows unchanged: {len(df)}")

df.to_csv(BB_CSV, index=False)
print(f"saved {BB_CSV}, {len(df)} rows, {len(df.columns)} columns, no rows dropped")
