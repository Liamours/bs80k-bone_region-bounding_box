# Duplicate detection across bs80k and LIBS-160K

Multi-step image duplicate detection: does bs80k have any internal duplicate patients, and does
it overlap with LIBS-160K, a second bone scan collection this project's own images were checked
against. Purely an image-integrity question, checked with three progressively more tolerant
methods, exact hash, perceptual hash, then direct location verification against this project's
own already-recovered boxes. This file covers only that image-integrity investigation.

## Phase 1: exact MD5 (`src/dedup/phase1_md5.py`)

Checked bs80k's own whole body and region layers, LIBS-160K's own whole body layer, and all 3 of
LIBS-160K's own tsv splits (train_imgs.tsv, test_imgs.tsv, valid_imgs.tsv) together, not each in
isolation.

### bs80k's own duplicates

Two different things, cleanly separated by whether the duplicate pair shares one patient id or
two:

- **`elbowLPOST` is byte identical to `elbowRPOST` for 2917 of 2925 patients (99.7%)**, same
  patient id both times, not a rare glitch, close to universal for the posterior elbow crop
  specifically. No other region/view pair shows this. This explains something glossed over
  earlier without being flagged: `result/figures/bounding_box_quality.md`'s own metrics table has
  identical numbers for `elbowLPOST` and `elbowRPOST` down to 4 decimal places, at the time read
  as a coincidence of two similarly easy regions, it is not a coincidence, the crop pixels
  bs80k ships for those two folders are the same file for nearly every patient.
- **4 pairs of different patient ids are byte identical across 24 of bs80k's own 26 region
  folders each**: `(828, 2244)`, `(312, 570)`, `(831, 3216)`, `(832, 3245)`. Not a coincidental
  single-region match, the same pair recurs across ankle, chest, head, knee, pelvis, shoulder,
  and vertebra, both views, every folder checked except 2. These read as the same physical
  patient entered under two different ids, a real bs80k data integrity issue. `bounding_boxes.csv`
  counts these 4 real patients as 8 in its raw row count; `duplicate_of_patient_id` (see below)
  flags this without merging or deleting any row.

### bs80k vs LIBS-160K, cross-dataset

- **35.0% of LIBS-160K's 192456 region-crop rows (67405 rows) sit in an exact duplicate group**
  with each other, 32469 groups total, mostly pairs, some triples/quads, one group of 6.
- **95.3% of those duplicate groups (30933 of 32469) span more than one of LIBS-160K's own
  splits**, the same image content sitting under a different id in train, test, and/or valid.
  Only 1536 groups are duplicated within a single split. A real leakage risk for anyone training
  and evaluating on LIBS-160K's own official split boundary, not a rare edge case, the majority
  pattern for every duplicate found. Not this project's data to fix, only flagged.
- **0 exact cross-dataset region-crop matches**, despite real overlap independently confirmed
  (both datasets in some sense draw on the same source population). Expected, not a
  contradiction: MD5 needs byte identical files, and LIBS-160K's own region crops are visibly
  reprocessed compared to bs80k's own. This is the gap phase 2 exists to close. The whole body
  layer's own cross-dataset matches (6484 groups, LIBS-160K's whole body images are byte
  identical to bs80k's own for every shared id) are untouched by this, those files genuinely are
  byte identical.

## Phase 2: Perceptual Hash (pHash), `src/dedup/phase2_phash.py`

**Perceptual Hash (pHash, DCT-based, 64 bit, `imagehash` package, `hash_size=8`)**, grouped by
region (13 codes, both bs80k views combined per region, all 3 LIBS-160K splits combined per
region) rather than one 76050 x 192456 matrix, since the region is already known on both sides.

The Hamming distance threshold was calibrated, not guessed. A first pass at <= 8 on one region
alone flagged 169103 pairs, suspiciously many. Checked 5 flagged pairs directly against this
project's own already-verified ground truth (`bounding_boxes.csv`), by template matching the
candidate crop against the actual bs80k whole body image and comparing the located (x, y) to the
known box:

- distance 0: located at the exact known box location, offset (0, 0), template match score 0.69
- distance 2: exact, offset (0, 0), score 0.77
- distance 4: exact, offset (0, 0), score 0.73
- distance 6: off by (38, 514) pixels, score 0.57, near zero peak margin, a coincidental hash
  collision, not a real match
- distance 8: off by dozens of pixels, score 0.46, same story

Settled on **Hamming distance <= 4** as the threshold, confirmed reliable, not assumed.
`result/figures/phash_verification_check.png` has the visual side of the distance 0 and 8 cases.

Full 13-region sweep, that threshold: **114456 cross-dataset near-duplicate pairs**, 33788 total
bs80k-id/region matches (one bs80k id can match in more than one region). Coverage (fraction of
bs80k's 2925 patients matched into LIBS-160K) is not uniform across regions:

- chest, shoulder, pelvis, vertebrae, head: 92.9-94.8% coverage, median best-distance across the
  whole region population sits right at the confirmed-reliable value of 4
- knee, ankle: 88.9-91.1% coverage, median best-distance 6, a value calibration found unreliable,
  so these numbers are coverage of the region generally, not a claim every one of those matches
  is individually as trustworthy as chest's
- elbow: 66.7-68.3% coverage, clearly lower, median best-distance 8. Ties back to the elbow
  duplication finding above, elbow crops carry less individually distinguishing detail than other
  regions, consistent with both findings, not a coincidence.

Full detail (all 114456 pairs, per-region summary) is in
`result/tables/dataset_duplication_and_regions.xlsx`, sheet "Duplicate Detail (pHash)".

## Phase 3: verifying phase 2's matches at full scale, `src/dedup/phase3_verify_phash.py`

Phase 2 calibrated its threshold on 5 examples. Phase 3 checks all 114456 of them, and checks the
right thing: does the candidate crop, searched freely across the whole bs80k whole body image
with this project's own `core.locate()`, land at the same location bs80k's own already-recovered
box already says it should, by IoU, not a raw pixel similarity score.

A first version of this script compared the crop resized into bs80k's exact box, rigid, no
search. It found chest averaging a *negative* pearson correlation despite chest being one of this
project's best solved regions, a red flag, checked before trusting it: for the single worst case,
a full unconstrained search landed at the *exact same position* as bs80k's own box, offset (0, 0),
with a real, unambiguous peak margin, not the near-zero-signal result the rigid, same-position
comparison implied. The location was right, raw pixel correlation between two independently
reprocessed copies of the same image just isn't a reliable absolute bar, unlike bs80k's own
purely internal comparisons. Rewrote phase 3 around location agreement instead.

Result, full scale: **median IoU 1.000, mean 0.849, 87.5% of all 114456 pairs land with IoU >=
0.5 against bs80k's own ground truth box**. By region: ankle/elbow/head/knee/pelvis/vertebra all
confirm at 97-99%. Chest confirms at a solid but lower 78-80%. Shoulder confirms at only 61-64%,
consistent with, not contradicting, this project's own already documented shoulder box
imprecision (`context/method.md`): phase 3 checks agreement against bs80k's own shoulder box,
and that box itself is known to be imprecise, so a genuinely correct match can still show low
IoU there. By Hamming distance: 0 confirms 95.7%, 2 confirms 93.5%, 4 confirms 76.3%, a real,
expected decline that matches the original 5-example calibration's own finding that 4 was the
loosest reliable threshold, now confirmed at full scale.

## bounding_boxes.csv columns produced by this work

Additive only, no row ever dropped or merged (`src/dedup/add_duplicate_notes.py`):

- `duplicate_of_patient_id`: set on 192 rows, the 4 cross-patient duplicate pairs x 2 ids x the
  24 of 26 components actually confirmed identical.
- `duplicate_of_sibling_component`: set on 5834 rows (2917 patients x 2), `elbowLPOST` and
  `elbowRPOST` pointing at each other.
- `libs160k_duplicate_matches`: phase 3 IoU-verified (>= 0.5) cross-dataset matches only, not the
  raw phase 2 flag list, semicolon joined `split/image_id(hamming_distance,iou)`. 59009 of 76050
  rows have at least one.

## Stance on LIBS-160K's own intra-dataset duplication and split leakage

Decided, not left open: this is not this project's dataset to fix, dedupe, or re-split. What this
project does instead:

- Flag it prominently, here and in `result/tables/dataset_duplication_and_regions.xlsx`, so
  anyone who trains or evaluates directly on LIBS-160K's own official split boundary knows 95.3%
  of its intra-duplicate groups cross a split, a real leakage risk, not a rare edge case.
- Confirm this project's own scripts are not exposed to that leakage. Checked directly, not
  assumed: neither `src/dedup/phase2_phash.py` nor `phase3_verify_phash.py` gates anything on a
  `train`/`test`/`valid` split boundary, phase 2's own sweep pools all 3 splits together per
  region on purpose, deliberately blind to which split a LIBS-160K image came from.
