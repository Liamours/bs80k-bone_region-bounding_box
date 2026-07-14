# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "pillow", "numpy", "pandas"]
# ///
"""Real region bounding boxes for a SAMPLE of LIBS-160K's new-only patients (not in bs80k),
not the full ~3491. Full coverage would mean searching, for every new patient and every region,
the residual LIBS crops not already matched to a bs80k patient (phase 2, src/dedup/phase2_phash.py)
against that one patient's own whole body image, roughly a minute per (patient, region) pair,
~11 hours for all 3491 patients x 13 regions. Not attempted at that scale here, that is a
decision for later, not a quiet default. This proves the approach works and delivers real boxes
for a bounded sample (N_PATIENTS), the same "small scale first" discipline this project has used
throughout (context/method.md's own 20 id samples before every full run).

Candidate pool per region: LIBS-160K crops NOT already matched to a bs80k patient by phase 2
(the "residual"), since the ones phase 2 already matched belong to a known bs80k patient, not a
new one. Template matching (masked, core.locate, this project's own proven method) against the
one target patient's own whole body image, not a blind many-to-many search.

v1 of this script shipped without two checks and produced an unreliable sample as a result
(context/wholebody_bbox.md, "Region boxes for LIBS-160K's new-only patients: first sample
attempt, not reliable, do not use", `result/figures/suspicious_shared_match_check.png`): 31.7%
of its 104 matches turned out to be the same LIBS crop claimed by more than one different target
patient, a logical impossibility if real. Verified two distinct causes directly: a near blank
candidate crop (foreground coverage 0.006) matching anywhere with a high score, and several
fully saturated candidate crops (coverage 1.000, no black background at all, masking cannot
discriminate) for a small, low detail region, shoulder among them, consistent with this
project's own already documented shoulder weakness.

Two fixes, both essentially free, no extra candidate evaluations needed since coverage and
second-best score come from work already being done per candidate:

1. CROP_COVERAGE_MIN / CROP_COVERAGE_MAX reject a candidate crop before it is even searched, if
   its own foreground coverage sits outside a range every genuine bs80k region crop this project
   has ever measured has landed in (`context/method.md`'s own background fraction notes,
   ankle ~0.35-0.38 foreground, chest ~0.34, shoulder ~0.46-0.49, all comfortably inside
   [0.02, 0.95]). A crop at 0.006 or 1.000 coverage never resembles any of those.
2. MIN_CROSS_CANDIDATE_MARGIN requires the winning candidate's own score to beat the second best
   *competing candidate crop's* own score by a real margin, not just beat an absolute threshold.
   This is a different thing than `peak_margin`, which only checks for a second peak *within one*
   correlation surface, not against other candidate crops, and did not separate the 33 bad v1
   rows from the 71 unshared ones (0.105 vs 0.130 mean, barely different).
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

LIBS = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw\LIBS-160K-EN")
LIBS_WB = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw")
WB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-wholebody-bb\bounding_boxes.csv")
PHASH_JSON = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-vqa-grounding\dedup_phase2_phash.json")
OUT_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb\libs160k_new_patient_sample_v2.csv")

N_PATIENTS = 8
MATCH_SCORE_THRESHOLD = 0.5
CROP_COVERAGE_MIN = 0.02
CROP_COVERAGE_MAX = 0.95
MIN_CROSS_CANDIDATE_MARGIN = 0.05

REGION_PHRASES = {
    "right chest": "chestR", "left chest": "chestL",
    "right shoulder joint": "shoR", "left shoulder joint": "shoL",
    "right knee joint": "kneeR", "left knee joint": "kneeL",
    "right elbow joint": "elbowR", "left elbow joint": "elbowL",
    "right ankle joint": "ankleR", "left ankle joint": "ankleL",
    "pelvis": "pelvis", "vertebrae": "vertbra", "head": "head",
}


def find_region(text: str) -> str:
    tl = text.lower()
    matches = [code for phrase, code in REGION_PHRASES.items() if phrase in tl]
    assert len(matches) == 1, text
    return matches[0]


def load_libs_region_map() -> dict[str, list[tuple[str, int]]]:
    splits = [
        ("train", LIBS / "train" / "train_texts.jsonl"),
        ("test", LIBS / "train" / "test_texts.jsonl"),
        ("valid", LIBS / "valid" / "valid_texts.jsonl"),
    ]
    region_map: dict[str, set[tuple[str, int]]] = {code: set() for code in REGION_PHRASES.values()}
    for split_name, jsonl_path in splits:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            region = find_region(rec["text"])
            for iid in rec["image_ids"]:
                region_map[region].add((split_name, iid))
    return {k: sorted(v) for k, v in region_map.items()}


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
    print("loading region map, tsv lookups, and phase 2 matched set...")
    region_map = load_libs_region_map()
    tsv_lookup = load_libs_tsv_lookup()
    phash = json.loads(PHASH_JSON.read_text())
    matched = {(f["libs160k_split"], f["libs160k_image_id"]) for f in phash["flags"]}

    wb = pd.read_csv(WB_CSV, low_memory=False)
    new_patients = wb[(wb["source"] == "libs160k") & (wb["view"] == "ANT")].sample(N_PATIENTS, random_state=0)
    print(f"sampled {len(new_patients)} new-only patients (ANT view): {sorted(new_patients['id'].tolist())}")

    rows = []
    for _, prow in new_patients.iterrows():
        pid = int(prow["id"])
        libs_dir = LIBS_WB / "wholeANT"
        p_abn = libs_dir / "Abnormal" / f"{pid}.jpg"
        whole_path = p_abn if p_abn.exists() else libs_dir / "Normal" / f"{pid}.jpg"
        whole = np.asarray(Image.open(whole_path).convert("L"))

        for region_code, items in region_map.items():
            residual = [(split, iid) for split, iid in items if (split, iid) not in matched]
            best = None
            second_best_score = -1.0
            n_rejected_coverage = 0
            for split, iid in residual:
                b64 = tsv_lookup[split].get(iid)
                if b64 is None:
                    continue
                crop = np.asarray(Image.open(BytesIO(base64.b64decode(b64))))
                mask = background_mask(crop)
                coverage = (mask > 0).mean()
                if coverage < CROP_COVERAGE_MIN or coverage > CROP_COVERAGE_MAX:
                    n_rejected_coverage += 1
                    continue
                m = locate(crop, whole, mask=mask)
                if best is None or m["score"] > best[0]["score"]:
                    if best is not None:
                        second_best_score = best[0]["score"]
                    best = (m, split, iid)
                elif m["score"] > second_best_score:
                    second_best_score = m["score"]

            cross_margin = (best[0]["score"] - second_best_score) if best else None
            accept = (best is not None and best[0]["score"] >= MATCH_SCORE_THRESHOLD
                      and cross_margin is not None and cross_margin >= MIN_CROSS_CANDIDATE_MARGIN)
            if accept:
                m, split, iid = best
                rows.append({
                    "region": region_code, "id": pid, "source": "libs160k",
                    "x": m["x"], "y": m["y"], "width": m["w"], "height": m["h"],
                    "match_score": m["score"], "peak_margin": m["peak_margin"],
                    "cross_candidate_margin": cross_margin,
                    "matched_libs160k_split": split, "matched_libs160k_image_id": iid,
                })
                print(f"  patient {pid} [{region_code}]: matched {split}/{iid}, score={m['score']:.3f} "
                      f"peak_margin={m['peak_margin']:.3f} cross_margin={cross_margin:.3f} "
                      f"({n_rejected_coverage} candidates rejected by coverage)")
            else:
                best_score = best[0]["score"] if best else None
                print(f"  patient {pid} [{region_code}]: no confident match ({len(residual)} residual candidates, "
                      f"{n_rejected_coverage} rejected by coverage, best score {best_score}, cross_margin {cross_margin})")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}, {len(out)} region boxes recovered for {len(new_patients)} sample new patients")


if __name__ == "__main__":
    main()
