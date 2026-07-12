# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "openpyxl"]
# ///
"""Per-id presence/absence across the 26 bone region folders."""
from pathlib import Path

import pandas as pd

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
OUT = Path(__file__).resolve().parents[2] / "result" / "tables" / "component_coverage.xlsx"

FOLDERS = [
    "ankleLANT", "ankleLPOST", "ankleRANT", "ankleRPOST",
    "chestLANT", "chestLPOST", "chestRANT", "chestRPOST",
    "elbowLANT", "elbowLPOST", "elbowRANT", "elbowRPOST",
    "headANT", "headPOST",
    "kneeLANT", "kneeLPOST", "kneeRANT", "kneeRPOST",
    "pelvisANT", "pelvisPOST",
    "shoLANT", "shoLPOST", "shoRANT", "shoRPOST",
    "vertbraANT", "vertbraPOST",
]

id_sets = {name: {int(p.stem) for p in (RAW / name).glob("*.jpg")} for name in FOLDERS}
all_ids = sorted(set.union(*id_sets.values()))

coverage = pd.DataFrame(
    {name: [int(i in id_sets[name]) for i in all_ids] for name in FOLDERS},
    index=all_ids,
)
coverage.index.name = "id"
coverage = coverage.reset_index()

summary = pd.DataFrame({
    "component": FOLDERS,
    "present": [len(id_sets[name]) for name in FOLDERS],
    "missing": [len(all_ids) - len(id_sets[name]) for name in FOLDERS],
    "total_ids": len(all_ids),
})

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as writer:
    coverage.to_excel(writer, sheet_name="coverage", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"saved {OUT}, {len(all_ids)} ids total")
