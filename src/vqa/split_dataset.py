# /// script
# requires-python = ">=3.11"
# ///
"""Patient id level train/val/test split of grounding_qa.jsonl, 80/10/10.

Every region and view for one patient stays in the same split, no id appears in more than
one, so a model never sees the same patient's other regions during evaluation.
"""
import json
import random
from pathlib import Path

DIR = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-vqa-grounding")
IN = DIR / "grounding_qa.jsonl"

records = [json.loads(l) for l in IN.open(encoding="utf-8")]

ids = sorted({int(r["image"].split("/")[-1].removesuffix(".jpg")) for r in records})
shuffled = ids.copy()
random.Random(0).shuffle(shuffled)
n = len(shuffled)
n_train = int(n * 0.8)
n_val = int(n * 0.9)
split_of = {}
for pid in shuffled[:n_train]:
    split_of[pid] = "train"
for pid in shuffled[n_train:n_val]:
    split_of[pid] = "val"
for pid in shuffled[n_val:]:
    split_of[pid] = "test"

by_split = {"train": [], "val": [], "test": []}
for r in records:
    pid = int(r["image"].split("/")[-1].removesuffix(".jpg"))
    by_split[split_of[pid]].append(r)

for name, rows in by_split.items():
    out = DIR / f"grounding_qa_{name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_abnormal = sum(1 for r in rows if r["diagnosis"] == "abnormal")
    print(f"{name}: {len(rows)} rows, {n_abnormal} abnormal ({n_abnormal / len(rows):.1%}), saved {out}")

assert set(split_of.values()) == {"train", "val", "test"}
assert sum(len(v) for v in by_split.values()) == len(records)
print(f"\n{len(ids)} patient ids, {len(shuffled[:n_train])} train / {len(shuffled[n_train:n_val])} val / {len(shuffled[n_val:])} test, no overlap by construction (each id assigned once)")
