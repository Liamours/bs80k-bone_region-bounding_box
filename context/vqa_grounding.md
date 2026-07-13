# Combining bs80k and LIBS-160K into a VQA and grounding dataset

## A resource not used until now: the whole body images' own nidus boxes

`context/dataset.md` already noted that `wholeBodyANT`/`wholeBodyPOST` each hold an `ant`/`post`
subfolder of Pascal VOC xml files, one per whole body image, marking physician drawn nidus
(suspected malignant) and physiological hot spot (benign) boxes, and said not to reuse them as
the answer for this project's own region location recovery. That's still right for the region
location task. But for a grounding dataset they are a second, independent, already existing
box source, at a finer grain than this project's own 13 region boxes: individual hot spots, each
one classed `Normal` or `Abnormal` in the xml's own `<name>` tag, checked directly, 9629 Normal
and 4194 Abnormal objects across all of wholeBodyANT's xml files alone.

Coverage is not the same 2925 id set as the region crops: 2924 ids have an ANT xml, 2555 have a
POST xml, checked directly against headANT.txt's 2925 ids, 223 ANT-xml ids fall outside the
region-crop set and 224 region-crop ids have no ANT xml. Handled by simple absence, a row just
gets an empty `hotspots` list when its id has no xml for that view, not treated as an error.

## Cross-validation: do the two independent label systems agree

Before building anything, checked whether this project's own recovered region boxes are
geometrically consistent with the physician's own nidus boxes, since if they weren't, chaining
them into one dataset would just be compounding two different bugs. Sampled 300 patient ids,
checked every `Abnormal` class nidus box's center point against this project's own 13 region
boxes for the matching id and view:

- 560 Abnormal nidus boxes seen across the sample
- 498 (89%) fell inside one of the 13 region boxes at all, the rest fall in body parts none of
  the 13 regions cover, hands, feet, thighs, forearms, ribs away from the chest crop, expected,
  the 13 regions do not tile the whole body
- of the 498, 469 (94%) landed in a region whose own separate label (from that region's txt
  file, `context/dataset.md`) was already abnormal, a real agreement between two independently
  derived signals, not the same measurement twice
- 29 (5.8%) landed in a region labeled normal despite an Abnormal-class nidus center falling
  inside it, a real disagreement, not chased further, plausibly a boundary nidus near a region
  edge or genuine label noise between the two annotation passes, worth knowing before trusting
  hotspot presence as a hard ground truth for any one row

## What was built

`src/vqa/build_grounding_dataset.py`, no LLM, no cross dataset image matching, three ingredients:

1. `bounding_boxes.csv`, this project's own region location plus that region's own normal or
   abnormal label
2. the whole body xml nidus boxes, joined onto a region box by simple point-in-box containment,
   center of the nidus box against the region's own recovered box
3. LIBS-160K's 39 caption templates (`context/libs160k.md`), reused by exact `(region, caption
   type)` lookup, the literal template text, not a paraphrase, not a generic skeleton with
   substitution, since a wrong substitution could silently reintroduce the wording
   inconsistencies already found in that file (vertebrae's "image" instead of "shadow", head's
   reordered abnormal phrasing), while an exact lookup by construction cannot

One JSON record per `bounding_boxes.csv` row, 76050 total, saved to
`C:\Users\lulay\Desktop\wbbs-dataset\bs80k-vqa-grounding\grounding_qa.jsonl`, outside the repo,
same convention as `bounding_boxes.csv` itself. Each record: `image`, `region`, `view`, `bbox`,
`diagnosis`, `caption_description`, `caption_diagnosis`, `hotspots` (possibly empty list of
`{bbox, class}`), and a `qa` list of question/answer pairs, a location question always, a yes/no
abnormality question always, and a hot spot location question only when `hotspots` is non-empty.

Bug caught while pulling sample records out to look at directly, not from a systematic check:
`hotspots` first included both `Normal` and `Abnormal` class xml boxes, so a benign physiological
uptake spot could end up as the answer to the "where is the abnormal tracer uptake" question,
mislabeled as a finding. Fixed in `hotspots_in_box()`, `Abnormal` class only now, matching what
the cross-validation section above already did. Regenerated after the fix.

Checked directly against the corrected file: of the 3560 abnormal rows in the full 76050, 3304
(92.8%) have at least one contained `Abnormal` hotspot box, consistent with the 300 id sample's
94% figure above. Of the 72490 normal rows, 1042 (1.44%) still carry one, the region/nidus label
disagreement from the cross-validation section, present at full scale, not just the sample.
Overall 4346 of 76050 rows (5.7%) carry at least one hotspot, down from an incorrect 17.2% before
the fix.

## Known limitations, honestly, before this gets used for anything

- The region/nidus label disagreement above (1.44% of normal rows) means `diagnosis` and
  `hotspots` can disagree for a given row, a downstream consumer should not assume `hotspots`
  empty implies normal or vice versa, both fields are kept, neither is silently corrected to
  match the other
- One concrete cause of that disagreement, seen directly in `grounding_qa_preview.png`: a
  `chest_R` row labeled normal contained 9 `Abnormal` hotspot boxes, all running down the spine,
  not the ribs, because this project's own recovered chest box is wide enough to overlap the
  vertebra column in raw pixel space. The hotspots are real, but they belong to a diffuse spinal
  uptake pattern, not a chest finding, misattributed by plain point-in-box containment against
  two anatomically adjacent, overlapping region boxes. Anywhere two region boxes overlap this can
  happen, not unique to chest and vertebra, just the pair caught by direct inspection so far
- Caption text is still the same 39 fixed sentences LIBS-160K itself uses, exact reuse fixes
  correctness but not variety, MedGround's principle (`context/libs160k.md`) of generating more
  varied natural queries, then verifying a generated claim's location against this project's own
  box, is the natural next step, not done here
- `qa` questions are template generated English, one fixed phrasing per question type, the same
  caveat as the caption text
- Not evaluated yet against any actual model or task, this is the dataset, not a benchmark result

## v2: quality flags and a patient level split

Four design questions asked and answered (`AskUserQuestion`) before touching code, each is a
constraint on what follows, not left open:

- **Output format** stays flat JSON, plain numeric `bbox` fields, no special tokens. Whether a
  downstream trainer wants bbox as plain text numbers (Shikra style), binned tokens (Kosmos-2
  style), or wrapped in a marker like `<DET>` is a later, separate conversion step once a target
  model is actually picked, not this dataset's concern. `<SEG>` style segmentation tokens are out
  of scope entirely, this project only ever produced box level ground truth, `background_mask()`
  in `src/matching/core.py` is a foreground/background split for template matching, not an
  anatomical or lesion segmentation, and is not reused as one.
- **Caption/question diversity** stays template based, no LLM paraphrase pass added here, that
  remains the MedGround-principle future work already noted above.
- **Class imbalance** (yes/no abnormal question, ~95%/5% No/Yes overall) left raw, undocumented
  skew would be worse than documented skew, balancing is a training time concern.
- **Known-unreliable rows** get quality flag fields added, nothing dropped, nothing silently
  fixed.

`src/vqa/build_grounding_dataset.py` now adds two fields to every record:

- `low_precision_region`: true when `region` is `shoulder_L` or `shoulder_R`, the two regions
  whose own near-exact-fraction sits around 0.24-0.25 against ~1.0 for every other region
  (`result/figures/bounding_box_quality.md`).
- `region_overlap`: which other of this project's own recovered region boxes (same patient, same
  view) geometrically overlap this row's box, plain rectangle intersection, generalizing the
  chest_R/vertebra false attribution case above from one anecdote into a field every row carries.
  Checked directly: the same chest_R row from `wholeBodyPOST/1761.jpg` now reports
  `region_overlap: ["shoulder_L", "shoulder_R", "vertebrae"]`, confirming the field catches the
  known case, not just runs without error.

Overlap turned out far more common than the one inspected case suggested, 36873 of 76050 rows
(48.5%) overlap at least one sibling region. Chest and vertebra overlap in essentially every
single patient and view, checked directly (chest_L/vertebrae and chest_R/vertebrae each around
5850 patient/view instances, matching 2925 ids times 2 views almost exactly), an anatomical fact
about how wide this dataset's own chest crops are relative to the spine, not a defect in the
overlap check. `hotspots` non-empty together with `region_overlap` non-empty is the specific
condition a training pipeline should treat as a weaker hotspot attribution, same as the chest_R
case.

`src/vqa/split_dataset.py`: patient id level train/val/test split, 80/10/10 by shuffled id
(`random.Random(0)`, same seeding convention as `preview_bounding_boxes.py`), so no patient's
regions cross a split boundary, checked directly, zero id intersection between all three split
files. `grounding_qa_train.jsonl` (60840 rows, 2340 ids, 4.7% abnormal),
`grounding_qa_val.jsonl` (7592 rows, 292 ids, 4.7% abnormal), `grounding_qa_test.jsonl` (7618
rows, 293 ids, 4.4% abnormal), all saved alongside `grounding_qa.jsonl` in
`bs80k-vqa-grounding/`, row counts sum to the full 76050 exactly.

## v3: whole body grounding, "localize the patient's whole body"

Once `bs80k-wholebody-bb/bounding_boxes.csv` existed at full scale (`context/wholebody_bbox.md`,
6494 rows, every id and view), the same "localize X" question became answerable one level up,
not a region inside the whole body image, but the whole body inside its own fixed scanner
canvas. `build_grounding_dataset.py` now also writes one `region: "whole_body"` record per
`bs80k-wholebody-bb/bounding_boxes.csv` row, appended after the region records, so the file now
has 82544 total (76050 region plus 6494 whole_body).

Diagnosis for a whole_body row comes from `wholeBodyANT.txt`/`wholeBodyPOST.txt` directly, the
same 0/1 label convention as every region folder (`context/dataset.md`), abnormal if any nidus
exists anywhere in the image, not derived or inferred. Hotspots reuse the same `hotspots_in_box`
containment check, against the whole body box instead of a region box, so it naturally picks up
every abnormal nidus in the image, 34.4% of whole_body rows have at least one, versus 5.7% for
region rows, expected, a whole body box covers far more area than any one region.

Deliberately does not carry `caption_description`, `caption_diagnosis`, `low_precision_region`,
or `region_overlap`. The first two have no LIBS-160K source, LIBS-160K's 39 templates are all
region level, inventing whole body caption text would break the exact reuse this project has
held to for every other caption field. The other two are region level quality signals that do
not have a whole body equivalent yet. A consumer should check `region == "whole_body"` before
expecting the region-only fields, the schema is not uniform across the two record kinds by
choice, not oversight.

`split_dataset.py` re-run unchanged, still patient id level, now spans all 3247 ids (whole_body
records exist for every id, not just the 2925 with region crops): 65956 train / 8268 val / 8320
test rows, zero id overlap confirmed again, sums to 82544 exactly.

whole_body records also carry `outlier`/`anomaly_score` straight from
`bs80k-wholebody-bb/bounding_boxes.csv`, not dropped, not excluded from the dataset. Decided
this the same way `low_precision_region`/`region_overlap` were decided in v2: the flag mixes two
different things, genuinely bad source images and legitimately small real patients
(`context/wholebody_bbox.md`'s own visual check found both in the same flagged tail), so
dropping every flagged row would silently remove real, usable scans along with the bad ones. A
consumer that only wants the bad-image kind still has to look, the flag only narrows down where.

## v4: every known duplicate finding noted on the data itself, nothing deleted

Explicit instruction: note everything about the phase 1/phase 2 duplicate findings
(`context/dataset.md`, `context/libs160k.md`) directly on the data, do not delete or merge
anything. `src/dedup/add_duplicate_notes.py` adds three additive columns to
`bounding_boxes.csv` (never touches row count, 76050 before and after):

- `duplicate_of_patient_id`: set on 192 rows, the 4 cross-patient duplicate pairs x 2 ids x the
  24 of 26 components actually confirmed identical (elbowLPOST/elbowRPOST excluded, see next)
- `duplicate_of_sibling_component`: set on 5834 rows (2917 patients x 2), elbowLPOST and
  elbowRPOST pointing at each other, a same-patient duplicate, a different kind of finding than
  the row above, not a different real person
- `libs160k_duplicate_matches`: set on 62014 of 76050 rows, semicolon joined
  `split/image_id(hamming distance)` from the phase 2 pHash sweep

`build_grounding_dataset.py` carries all three straight through onto region records unchanged
(`None`/`[]` when absent). Fixing this required fixing a real, separate bug found only by
re-running the generator for the first time since whole body bbox was extended to LIBS-160K's
new patients (v3 above): `load_wholebody_labels()` only read bs80k's own txt label files, so any
`libs160k`-only whole body id crashed with a `KeyError`. Fixed by falling back to LIBS-160K's own
Abnormal/Normal folder placement for ids bs80k has no label file for, the same signal already
checked byte identical and 100% label-agreeing for every id bs80k does have
(`context/libs160k.md`). This also means `grounding_qa.jsonl`'s `whole_body` records were stale
since that earlier extension, only 6494 of them instead of the full 13476, silently missing
every LIBS-160K-only patient until this run. Now 89526 total records (76050 region + 13476
whole_body), splits regenerated (71126 train / 9382 val / 9018 test, 6738 patient ids, zero id
overlap reconfirmed).
