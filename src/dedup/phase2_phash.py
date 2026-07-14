# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "imagehash", "numpy"]
# ///
"""Phase 2 of multi-step duplicate detection: Perceptual Hash (pHash, DCT-based, 64-bit,
via the imagehash package), scoped to the one gap phase 1 (MD5, src/dedup/phase1_md5.py) named
explicitly: cross-dataset region-crop matches. MD5 found 0 there despite confirmed real overlap
between the two datasets, because LIBS-160K's region crops are visibly reprocessed, a recompress
defeats an exact byte hash but not a perceptual one.

Grouped by region (13 codes) rather than compared as one 76050 x 192456 matrix, both because the
regions are already known (bs80k's own folder name, LIBS-160K's own caption group) and because
it keeps the comparison tractable: bs80k's per-region group is ~2925-5850 images, LIBS-160K's is
~9770-9928, a 13-way split instead of one 14.6 billion cell matrix.

Hamming distance <= 4 (of 64 bits) flags a candidate near-duplicate, vectorized via numpy
(byte-wise popcount lookup table, chunked to bound memory), not python-level nested loops.

Threshold calibrated, not guessed: a first pass at <= 8 on chestR alone flagged 169103 pairs,
suspiciously many. Checked 5 flagged pairs directly against this project's own already-verified
ground truth (bounding_boxes.csv's own recovered box location for that bs80k id, template
matching the LIBS crop against the actual bs80k whole body image): distance 0, 2, and 4 each
landed exactly on the known ground truth location (offset (0,0), template match score
0.73-0.91). Distance 6 and 8 did not, off by dozens to hundreds of pixels, template match score
under 0.57 with a near-zero peak margin, a coincidental hash collision, not a real duplicate.
result/figures/phash_verification_check.png has the visual side of this check.
"""
import base64
import json
from io import BytesIO
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
LIBS = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw\LIBS-160K-EN")
OUT = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-dedup\dedup_phase2_phash.json")
HAMMING_THRESHOLD = 4

REGION_PHRASES = {
    "right chest": "chestR", "left chest": "chestL",
    "right shoulder joint": "shoR", "left shoulder joint": "shoL",
    "right knee joint": "kneeR", "left knee joint": "kneeL",
    "right elbow joint": "elbowR", "left elbow joint": "elbowL",
    "right ankle joint": "ankleR", "left ankle joint": "ankleL",
    "pelvis": "pelvis", "vertebrae": "vertbra", "head": "head",
}
BS80K_FOLDERS = {  # region code -> list of bs80k folders (both views where paired)
    "chestR": ["chestRANT", "chestRPOST"], "chestL": ["chestLANT", "chestLPOST"],
    "shoR": ["shoRANT", "shoRPOST"], "shoL": ["shoLANT", "shoLPOST"],
    "kneeR": ["kneeRANT", "kneeRPOST"], "kneeL": ["kneeLANT", "kneeLPOST"],
    "elbowR": ["elbowRANT", "elbowRPOST"], "elbowL": ["elbowLANT", "elbowLPOST"],
    "ankleR": ["ankleRANT", "ankleRPOST"], "ankleL": ["ankleLANT", "ankleLPOST"],
    "pelvis": ["pelvisANT", "pelvisPOST"], "vertbra": ["vertbraANT", "vertbraPOST"],
    "head": ["headANT", "headPOST"],
}

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hash_to_uint64(h: imagehash.ImageHash) -> np.uint64:
    bits = h.hash.flatten().astype(np.uint64)
    val = np.uint64(0)
    for b in bits:
        val = (val << np.uint64(1)) | b
    return val


def popcount_u64(arr: np.ndarray) -> np.ndarray:
    b = arr.view(np.uint8).reshape(*arr.shape, 8)
    return _POPCOUNT8[b].sum(axis=-1)


def find_region(text: str) -> str:
    tl = text.lower()
    matches = [code for phrase, code in REGION_PHRASES.items() if phrase in tl]
    assert len(matches) == 1, text
    return matches[0]


def load_libs_region_map() -> dict[str, list[tuple[str, int]]]:
    """region_code -> [(split, image_id), ...] across all 3 splits, deduped."""
    splits = [
        ("train", LIBS / "train" / "train_texts.jsonl"),
        ("test", LIBS / "train" / "test_texts.jsonl"),
        ("valid", LIBS / "valid" / "valid_texts.jsonl"),
    ]
    region_map: dict[str, set[tuple[str, int]]] = {code: set() for code in BS80K_FOLDERS}
    for split_name, jsonl_path in splits:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            region = find_region(rec["text"])
            for iid in rec["image_ids"]:
                region_map[region].add((split_name, iid))
    return {k: sorted(v) for k, v in region_map.items()}


def load_libs_tsv_lookup() -> dict[str, dict[int, str]]:
    """split -> {image_id: b64 string}"""
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
    print("loading LIBS-160K region map and tsv lookups...")
    region_map = load_libs_region_map()
    tsv_lookup = load_libs_tsv_lookup()
    for code, items in region_map.items():
        print(f"  {code}: {len(items)} libs160k images")

    all_flags = []
    stats = {}
    for code, folders in BS80K_FOLDERS.items():
        bs_paths = []
        for folder in folders:
            bs_paths.extend((RAW / folder).glob("*.jpg"))
        bs_ids = [(p, int(p.stem)) for p in bs_paths]

        libs_items = region_map[code]
        if not bs_ids or not libs_items:
            continue

        print(f"\n[{code}] hashing {len(bs_ids)} bs80k images...")
        bs_hashes = []
        bs_meta = []
        for p, pid in bs_ids:
            img = Image.open(p)
            bs_hashes.append(hash_to_uint64(imagehash.phash(img)))
            bs_meta.append((pid, p.parent.name))
        bs_arr = np.array(bs_hashes, dtype=np.uint64)

        print(f"[{code}] hashing {len(libs_items)} libs160k images...")
        libs_hashes = []
        libs_meta = []
        for split, iid in libs_items:
            b64 = tsv_lookup[split].get(iid)
            if b64 is None:
                continue
            img = Image.open(BytesIO(base64.b64decode(b64)))
            libs_hashes.append(hash_to_uint64(imagehash.phash(img)))
            libs_meta.append((split, iid))
        libs_arr = np.array(libs_hashes, dtype=np.uint64)

        print(f"[{code}] comparing {len(bs_arr)} x {len(libs_arr)} = {len(bs_arr)*len(libs_arr):,} pairs...")
        n_flagged = 0
        best_per_libs = np.full(len(libs_arr), 64, dtype=np.uint8)
        chunk = 200
        for start in range(0, len(bs_arr), chunk):
            bs_chunk = bs_arr[start:start + chunk]
            xor = bs_chunk[:, None] ^ libs_arr[None, :]
            dist = popcount_u64(xor)
            best_per_libs = np.minimum(best_per_libs, dist.min(axis=0))
            ys, xs = np.where(dist <= HAMMING_THRESHOLD)
            for y, x in zip(ys, xs):
                pid, folder = bs_meta[start + y]
                split, iid = libs_meta[x]
                all_flags.append({
                    "region": code, "bs80k_id": pid, "bs80k_folder": folder,
                    "libs160k_split": split, "libs160k_image_id": iid,
                    "hamming_distance": int(dist[y, x]),
                })
                n_flagged += 1

        stats[code] = {
            "n_bs80k": len(bs_arr), "n_libs160k": len(libs_arr), "n_flagged_pairs": n_flagged,
            "median_best_distance": float(np.median(best_per_libs)),
            "min_distance_overall": int(best_per_libs.min()) if len(best_per_libs) else None,
        }
        print(f"[{code}] {n_flagged} pairs with Hamming distance <= {HAMMING_THRESHOLD}, "
              f"median best-distance per libs160k image: {stats[code]['median_best_distance']:.1f}")

    OUT.write_text(json.dumps({"threshold": HAMMING_THRESHOLD, "stats": stats, "flags": all_flags}, indent=2))
    print(f"\nsaved {OUT}")
    print(f"total flagged cross-dataset near-duplicate pairs: {len(all_flags)}")


if __name__ == "__main__":
    main()
