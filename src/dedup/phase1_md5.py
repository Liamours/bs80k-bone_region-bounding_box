# /// script
# requires-python = ">=3.11"
# ///
"""Phase 1 of multi-step duplicate detection: exact MD5, across everything, both datasets.

Catches byte-identical duplicates only. Cheap and fast, but a single recompress or resize
defeats it, that's what phase 2 (perceptual hash) and phase 3 (pixel verification) are for.
Scope, deliberately everything, not just the whole body layer already spot-checked:
  - bs80k whole body images (wholeBodyANT/POST)
  - bs80k region crop images (all 26 region/view folders)
  - libs160k whole body images (wholeANT/wholePOST, Abnormal+Normal)
  - libs160k region-crop tsv images (train_imgs.tsv, test_imgs.tsv, valid_imgs.tsv)
Reports: intra-LIBS duplicates (including train/test/valid split leakage), and any cross-dataset
exact match beyond the whole body layer already confirmed (context/libs160k.md).
"""
import base64
import hashlib
import json
from pathlib import Path

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
LIBS = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw")
LIBS_EN = LIBS / "LIBS-160K-EN"
OUT = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-vqa-grounding\dedup_phase1_md5.json")


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def hash_bs80k_wholebody() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for view in ("ANT", "POST"):
        for p in (RAW / f"wholeBody{view}").glob("*.jpg"):
            out.setdefault(md5_bytes(p.read_bytes()), []).append(f"bs80k:wholeBody{view}/{p.name}")
    return out


def hash_bs80k_regions() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    region_dirs = [d for d in RAW.iterdir() if d.is_dir() and d.name not in ("wholeBodyANT", "wholeBodyPOST")]
    for d in region_dirs:
        for p in d.glob("*.jpg"):
            out.setdefault(md5_bytes(p.read_bytes()), []).append(f"bs80k:{d.name}/{p.name}")
    return out


def hash_libs_wholebody() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for view in ("ANT", "POST"):
        for label in ("Abnormal", "Normal"):
            folder = LIBS / f"whole{view}" / label
            for p in folder.glob("*.jpg"):
                out.setdefault(md5_bytes(p.read_bytes()), []).append(f"libs160k:whole{view}/{label}/{p.name}")
    return out


def hash_libs_region_crops() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    splits = [
        ("train", LIBS_EN / "train" / "train_imgs.tsv"),
        ("test", LIBS_EN / "train" / "test_imgs.tsv"),
        ("valid", LIBS_EN / "valid" / "valid_imgs.tsv"),
    ]
    for split_name, tsv_path in splits:
        for line in tsv_path.read_text(encoding="utf-8").splitlines():
            image_id, b64 = line.split("\t")
            raw = base64.b64decode(b64)
            out.setdefault(md5_bytes(raw), []).append(f"libs160k:{split_name}/{image_id}")
    return out


def merge(*dicts: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for d in dicts:
        for h, locs in d.items():
            merged.setdefault(h, []).extend(locs)
    return merged


def main():
    print("hashing bs80k whole body...")
    bs_whole = hash_bs80k_wholebody()
    print(f"  {sum(len(v) for v in bs_whole.values())} files, {len(bs_whole)} unique hashes")

    print("hashing bs80k region crops...")
    bs_region = hash_bs80k_regions()
    print(f"  {sum(len(v) for v in bs_region.values())} files, {len(bs_region)} unique hashes")

    print("hashing libs160k whole body...")
    libs_whole = hash_libs_wholebody()
    print(f"  {sum(len(v) for v in libs_whole.values())} files, {len(libs_whole)} unique hashes")

    print("hashing libs160k region-crop tsv (train/test/valid)...")
    libs_region = hash_libs_region_crops()
    print(f"  {sum(len(v) for v in libs_region.values())} files, {len(libs_region)} unique hashes")

    everything = merge(bs_whole, bs_region, libs_whole, libs_region)
    dup_groups = {h: locs for h, locs in everything.items() if len(locs) > 1}
    print(f"\n{len(dup_groups)} hash groups with more than one file, out of {len(everything)} total unique hashes")

    intra_libs_region = {h: locs for h, locs in dup_groups.items() if all(l.startswith("libs160k:") and "/" in l and l.split(":")[1].split("/")[0] in ("train", "test", "valid") for l in locs)}
    train_test_leak = {h: locs for h, locs in intra_libs_region.items()
                       if len({l.split(":")[1].split("/")[0] for l in locs} & {"train", "test"}) == 2}
    cross_dataset = {h: locs for h, locs in dup_groups.items()
                     if any(l.startswith("bs80k:") for l in locs) and any(l.startswith("libs160k:") for l in locs)}
    cross_new = {h: locs for h, locs in cross_dataset.items() if not all("wholeBody" in l or "whole" in l.split(":")[1].split("/")[0] for l in locs)}

    print(f"\nintra libs160k region-crop duplicate groups: {len(intra_libs_region)}")
    print(f"  of those, appear in BOTH train and test split: {len(train_test_leak)} (train/test leakage)")
    print(f"cross-dataset exact match groups (any bs80k + any libs160k): {len(cross_dataset)}")
    print(f"  of those, involving the region-crop layer specifically (not just whole body): {len(cross_new)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "dup_groups": dup_groups,
        "intra_libs_region": intra_libs_region,
        "train_test_leak": train_test_leak,
        "cross_dataset": cross_dataset,
        "cross_new_region_matches": cross_new,
    }, indent=2))
    print(f"\nsaved {OUT}")

    if cross_new:
        print("\nsample cross-dataset region-crop exact matches:")
        for h, locs in list(cross_new.items())[:10]:
            print(f"  {locs}")
    if train_test_leak:
        print("\nsample train/test leakage groups:")
        for h, locs in list(train_test_leak.items())[:10]:
            print(f"  {locs}")


if __name__ == "__main__":
    main()
