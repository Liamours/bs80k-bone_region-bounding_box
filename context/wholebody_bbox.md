# Whole body bounding box: cropping the fixed scanner canvas to the actual body

## The problem

`wholeBodyANT`/`wholeBodyPOST` images are all a fixed 256x1024 canvas (`context/dataset.md`),
but the scanner's field of view is fixed while patients are not, a child's whole body scan
does not fill the same canvas an adult's does. This is a different bounding box than the rest
of this project: not a region crop's location inside the whole body image, but the whole body's
own location inside its fixed scanner canvas.

## First pass: plain threshold, and the real check available for it

Reused this project's own `background_mask()` (`src/matching/core.py`, threshold 2, already
used for region crops) directly on a whole body image, bbox from the min/max coordinates of
whatever is above threshold. No manual annotation exists for this task, but a real check does:
this project's own already recovered region boxes (`bounding_boxes.csv`) are parts of the body,
so a correct whole body box must contain the union of all of them, for a given id and view.

`src/wholebody/naive_threshold_experiment.py`, N=20: 15/20 contained the region union outright,
the 5 misses were all off by 1-5px, likely region-box noise more than a real miss. Scaled to
N=150 (`bbox_and_outliers.py`'s containment check): 72.7% contain outright, median slack when
containing is only 2px, this is already a fairly tight box, not a wildly wrong one.

## Three cleanup attempts, all made it worse, a real negative result

Visual inspection (`src/wholebody/preview_naive_threshold.py`,
`result/figures/naive_threshold_preview.png`) found the plain threshold bbox often nearly spans
the full 256px canvas width, and a direct check confirmed why for 3 of 5 inspected ids: the
box's own leftmost and rightmost pixel came from a single 1px noise speck, not real anatomy.
Sounds like an obvious fix, filter small connected components. Tried three ways, all checked
against the same region-union containment metric, all worse than doing nothing:

- plain area filter (`component_filter_sweep.py`): 15/20 down to 3/20 at any area cutoff tried
- morphological closing before the same area filter (`closing_sweep.py`): still short of the
  unfiltered box until the kernel grew large enough to stop being any tighter than it
- distance from the main body blob instead of size (`distance_filter_sweep.py`): same story,
  73%(raw N=150) down to the low-to-mid 50-60% range across every distance tried

Why: sparse, low count tracer signal at the body's true extremities, fingertips, toes, the top
of the scalp, forms small, sometimes disconnected components the same way a genuine noise
speck does. Area, shape after closing, and distance from the main mass cannot reliably tell
"real but sparse" from "noise" apart, because both are small and both can sit right at the
body's true edge. Filtering by any of them removes some real signal along with the noise,
pulling the box inward past the true edge more often than it trims a stray pixel.

## What worked: leave the mask alone, pad the box a few pixels instead

Simplest thing tried, and it beat all three cleanup methods: keep the plain threshold bbox, add
a fixed pixel margin on every side. N=150: pad 0px, 72.7% contain; pad 5px, 92.7%; pad 8px,
96.7%. No noise/signal discrimination needed, no risk of trimming real anatomy, the padding
only ever grows the box. `src/wholebody/bbox_and_outliers.py` uses threshold 2 plus a 5px pad.

## Multi-metric features and unsupervised outlier detection

Per bbox: coverage fraction, aspect ratio, horizontal center offset from the canvas center, top
and bottom margin fraction, width fraction, height fraction. `IsolationForest` (scikit-learn,
unsupervised, `contamination=0.05`), not a single-metric cutoff, deliberately, since body size
genuinely varies (a real small patient and a thresholding failure can both look small on any one
metric alone).

N=300 (ANT), 15 flagged (5.0%). Visually checked the top 5
(`src/wholebody/preview_outliers.py`, `result/figures/wholebody_outlier_preview.png`):

- ids 1654, 779, 722: not usable whole body scans at all, a blurry featureless blob, a mostly
  blank frame with a few saturated bright dots, and near pure noise with two saturated dots,
  no recognizable skeleton in any of them. Genuine bad source images, not a thresholding
  algorithm problem.
- id 1240: a completely normal, well formed, clearly smaller bodied scan, proportions read as a
  child. Exactly the real, legitimate small-body case this task was worried about conflating
  with an error.
- id 2513 (not flagged, shown for comparison): ordinary adult sized scan.

## A real, independent cross-check: outliers and the paper's own excluded 322

`context/dataset.md` already notes 322 of 3247 whole body ids have no region crops at all,
the original paper's own automatic segmentation failed on them (section 3.3). Checked whether
this project's own outlier flag lines up with that pre-existing, independent marker of a
troublesome image, not something this project labeled:

- overall N=300 sample: 8.7% lack region crops, close to the known population rate of 9.9%
  (322/3247), a sanity check that the sample is representative
- of the 15 flagged outliers: 66.7% (10/15) lack region crops
- of the 285 non-outliers: 5.6% lack region crops

All 4 visually inspected ids above (1654, 779, 722, 1240) lack region crops. A different method,
built for a different task, independently landing on largely the same troublesome ids as the
original paper's own segmentation failures is a real agreement, not a coincidence, and it means
this project's whole body outlier flag is picking up a mix of two genuinely different things,
bad source images and legitimately unusual bodies, that a downstream consumer would want to
treat differently, not lumped into one "outlier" bucket without a look.

## Full scale run

`src/wholebody/generate_wholebody_bounding_boxes.py`, every `wholeBodyANT` and `wholeBodyPOST`
id, both views, same method (threshold 2, pad 5px, IsolationForest per view,
`contamination=0.05`). Saved to `bs80k-wholebody-bb/bounding_boxes.csv`, 6494 rows (3247 ids x
2 views), same naming convention as `bs80k-bone_region-bb/bounding_boxes.csv`. 0 fully blank
images in either view, the near-blank fallback path exists but was never actually hit here.

The N=300 ANT-only sample's outlier/region-crop correlation holds at full population scale, on
both views, not just the one sampled:

- ANT: 163 outliers (5.0%, `contamination` is a target rate, this is confirmation it lands
  where set), 58.9% of them lack region crops versus 7.3% of non-outliers
- POST: 163 outliers (5.0%), 60.1% lack region crops versus 7.3% of non-outliers

Same order of enrichment as the sample (66.7% vs 5.6% there), an outlier is roughly 8x as likely
to be one of the paper's own 322 excluded ids as a non-outlier is, on the full population, not
just the one sample that happened to be checked first.

Columns: `id`, `view`, `x`, `y`, `width`, `height`, the 7 features (`coverage`,
`aspect_ratio`, `center_x_offset`, `top_margin_frac`, `bottom_margin_frac`, `width_frac`,
`height_frac`), `fallback`, `outlier`, `anomaly_score`, `has_region_crops`.

## Extended with LIBS-160K's own extra patients

Checked directly, full scale, not assumed: LIBS-160K's `wholeANT`/`wholePOST` classification
folders are byte identical to BS-80K's own `wholeBodyANT`/`wholeBodyPOST` for every one of the
3247 shared ids on both views, and LIBS-160K's Abnormal/Normal folder placement agrees with
BS-80K's own txt label 100% of the time. Not independent data, a repackaging of BS-80K's own
whole body layer (`context/libs160k.md`).

Beyond those 3247, LIBS-160K has 3491 more ids per view that BS-80K does not have at all, real
new patients. `generate_wholebody_bounding_boxes.py` now runs the same bbox method (threshold 2,
pad 5px) on those too, one `IsolationForest` per view fit across the combined 6738-id population
so outlier scores stay on the same scale for both sources, rather than two separately normalized
models. `bounding_boxes.csv` grew from 6494 to 13476 rows, gained a `source` column
(`bs80k`/`libs160k`), `has_region_crops` is trivially false for every `libs160k` row (they were
never BS-80K patients, no region crop folders exist for them).

Outlier rate split cleanly by source once fit on the combined population: bs80k rows, 10.3% both
views; libs160k-only rows, 0.09% ANT, 0.0% POST. The new patients are almost never flagged, the
outlier detector is still concentrating on BS-80K's own already-known problem population (the
322 excluded ids and friends), not spuriously flagging the new cohort, a reassuring sign the
method generalizes rather than just fitting noise in the original population.

## A second, more conservative automated signal: likely_corrupt_image

Revisited the "bad image versus real small body" split deliberately rather than leaving it pure
manual review forever. A real whole body silhouette is always tall and narrow regardless of how
small the patient is, a corrupt image's own largest connected component carries no such
constraint. Checked against 9 already visually inspected outliers: the 3 confirmed pure noise
cases (no skeleton at all, `result/figures/wholebody_outlier_preview.png`) sat at largest
connected component aspect ratio (height / width of its own bounding box) 0.91-1.36. Every case
with any real skeletal structure, including 2 genuinely ambiguous blob shaped corrupt images and
2 genuine but atypically proportioned real scans (`result/figures/aspect_ratio_check.png`), sat
at 1.77 or above.

`likely_corrupt_image` (`generate_wholebody_bounding_boxes.py`) uses a conservative threshold of
1.5, precision favored over recall on purpose: it only flags the clearest no-real-content cases,
23 of 674 outliers (3.4%, 16 ANT + 7 POST), not an attempt to resolve the whole ambiguous middle
automatically. That middle, blob shaped corrupt images that still have a tall-ish silhouette, or
real scans with an unusual pose or crop, is a genuinely fuzzy problem this one feature does not
solve, it narrows the human review queue, it does not replace it. Never drops a row, same
convention as `outlier` itself.

## Region boxes for LIBS-160K's new-only patients: first sample attempt, not reliable, do not use

`src/wholebody/build_libs_new_patient_region_boxes.py` ran a bounded sample (8 new patients x 13
regions) matching residual (not already phase 2 matched) LIBS-160K crops against each patient's
own whole body image, saved to `bs80k-bone_region-bb/libs160k_new_patient_sample.csv`. Checked
before trusting it, not after: 33 of the 104 rows (31.7%) share the exact same matched LIBS crop
across more than one different target patient, a logical impossibility if the match were real,
one crop cannot be two different people's ankle.

Verified directly, not just by the count, `result/figures/suspicious_shared_match_check.png`:
the `ankleR` crop claimed by 6 of 8 patients (`test/9387`) is a near blank crop, foreground
coverage 0.006, matches essentially anywhere with a high score since there is almost no real
content to match against wrongly, "located" at three completely different, anatomically
nonsensical positions (knee height, the other leg, up near the head) across three of its six
claimants. A second, different failure mode explains the other repeat offenders (`shoL` x5,
`chestL` x4, `vertbra` x4 patients): all three sit at foreground coverage 1.000, fully saturated,
no black background at all, so masking cannot help discriminate, and a small low detail crop can
correlate similarly well against many different real patients. `shoL` specifically being weak is
consistent with, not a contradiction of, this project's own extensively documented shoulder
precision problem elsewhere in this file and in `context/method.md`.

Root cause: the script had no data quality pre-filter on candidate crops (near blank or fully
saturated crops should be rejected before searching, not after), and no cross-candidate margin
(best score against the second-best competing crop's own score, not just the second-peak
suppression margin used everywhere else in this project, which only checks for a second peak
*within* one search, not against other candidate crops). A single absolute score threshold
(0.5) was not strict enough to catch either failure mode, mean peak_margin for the 33 bad rows
(0.105) barely differs from the 71 unshared rows (0.130).

Even the 71 unshared rows are not confirmed correct, only not caught by this one check. Unlike
bs80k's own population, there is no independent ground truth to verify a new patient's box
against, the cross-check this project used everywhere else (comparing a found location to this
project's own already-recovered box) does not exist for a patient bs80k never had. Treat this
whole sample as a proof that the approach is directionally possible, not as usable output.
`libs160k_new_patient_sample.csv` should not be joined into any deliverable as is.

## Not done yet

- Fix `build_libs_new_patient_region_boxes.py`: reject near blank and fully saturated candidate
  crops before searching, add a cross-candidate margin (best vs second-best competing crop) and
  require it to be large before accepting a match, then re-run the sample and, if it holds up,
  decide on scaling beyond the 8 patient proof of concept. Not done yet, a real compute cost
  (the first run took about 2 hours for 8 patients), a decision for later, not a default

- The ambiguous middle between "confirmed corrupt" and "confirmed real," roughly 96% of flagged
  outliers, still needs a human look, no further automated signal attempted yet
- No decision yet on what a downstream annotation consumer should do with a flagged bad image,
  exclude it, keep it with a flag, something else
