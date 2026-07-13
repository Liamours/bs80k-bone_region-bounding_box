# LIBS-160K, probed for a future combined VQA and grounding dataset

## Location

`C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw\LIBS-160K-EN`, two splits, `train/` and `valid/`. Lives outside this repo, same as the bs80k raw data.

## Files

Each split has an images tsv and a captions jsonl:

- `train/train_imgs.tsv`, `train/test_imgs.tsv` (a second split sits inside the train folder), `valid/valid_imgs.tsv`
- `train/train_texts.jsonl`, `train/test_texts.jsonl`, `valid/valid_texts.jsonl`

Row counts: train_imgs.tsv 128304 images, test_imgs.tsv 32058, valid_imgs.tsv 32094.

## Image tsv format

Two tab separated columns per line, no header: an integer image id, then a base64 encoded JPEG. Decoded id 1 from train_imgs.tsv directly to check, grayscale, 57 by 153 pixels, visibly a chest bone scan crop, ribs and spine edge visible. That size is close to this project's own chestLANT crop size, roughly 62 by 154 (`context/method.md`), a real, checked similarity, not assumed.

## Caption jsonl format

39 lines per split, each `{"text_id": int, "text": str, "image_ids": [int, ...]}`. Every line's `image_ids` count is a real per caption image count, checked directly, not estimated.

The 39 lines break down as 13 body regions times 3 caption templates per region. The 13 regions: chest left/right, shoulder joint left/right, knee joint left/right, elbow joint left/right, ankle joint left/right, pelvis, vertebrae, head, the same 13 part and side combinations this project's own bs80k folders use, just spelled out as full words instead of folder names.

The 3 caption templates per region, using right chest as the example:

1. "This is an image of a radionuclide bone scan of the patient's right chest.", the description caption, applies to essentially every image of that region, 9860 for right chest
2. "This image shows the patient's right chest in an elevated shadow of tracer uptake.", the abnormal caption, a small subset, 598 for right chest
3. "This image does not show an abnormally dense, sparse, or absent shadow of the patient's right chest skeletal tracer.", the normal caption, the rest, 9262 for right chest

Checked directly: caption 2's count plus caption 3's count equals caption 1's count exactly for right chest, 598 plus 9262 is 9860, and the same held for every region checked. This is a normal or abnormal label per image, expressed as one of two caption sentences, not free text per image, the same normal/abnormal distinction as bs80k's own per folder txt files (`context/dataset.md`), just captioned rather than numbered 0 or 1.

## Text analysis

`src/analysis/libs160k_text_analysis.py` analyzed the 39 caption strings directly, word frequency, unsupervised clustering, and pattern extraction, all checked by running code against the actual text, not read off the earlier sample by eye.

Vocabulary is tiny and repetitive by construction, 622 word tokens total across all 39 captions, only 39 distinct words. Top words by count: of (47), image (40), this (39), patient's (39), the (36), an (34), tracer (26), shadow (25), joint (24), a (22), right (15), left (15). joint appearing 24 times matches the 8 joint regions, shoulder, knee, elbow, ankle each left and right, times 3 caption types exactly. right and left at 15 each match the 5 paired regions times 3 types.

Region detection, matching each caption against the 13 known region phrases, found exactly one match per caption, no ambiguity. Caption type classification, description versus abnormal versus normal by simple substring rules, split cleanly 13/13/13.

Masking the region name out of each caption and comparing what is left, the sentence skeleton, is not as uniform as the earlier read suggested:

- description: 1 skeleton only, "This is an image of a radionuclide bone scan of _ patient's REGION.", the only variation across all 13 regions is "a" versus "the" before "patient's", already normalized out above
- abnormal: 3 different skeletons, not 1. 7 regions use "shows _ patient's REGION in an elevated shadow of tracer uptake", 5 regions use "shows _ patient's REGION as a tracer uptake enhancement shadow", and head alone uses a third, reordered phrasing, "shows a tracer uptake of _ patient's REGION in an elevated shadow"
- normal: 2 skeletons. 12 regions use "...absent shadow of _ patient's REGION skeletal tracer", vertebrae alone uses "...absent image of _ patient's REGION skeletal tracer", image instead of shadow, a real wording inconsistency in the source captions, not something this project introduced

Unsupervised KMeans clustering on plain TF-IDF vectors of the 39 captions, no region or type label given to it, recovers caption type almost too well, k=3 against the 3 known types scores an adjusted rand index of 1.000, perfect agreement. The same method asked to find 13 clusters against the 13 known regions scores -0.068, no better than random. The boilerplate phrasing dominates the vector space enough that region specific words, chest, knee, and so on, do not separate out on their own. A hierarchical clustering dendrogram, `result/figures/libs160k_text_dendrogram.png`, shows the same thing visually, description separates out first and most distinctly, cosine distance around 0.83 from everything else, abnormal and normal are more similar to each other, both starting "this image", and within abnormal the two competing phrasings form their own distinct sub branches, with head's uniquely reordered phrasing sitting closest to the "in an elevated shadow" branch.

None of this changes the earlier read that LIBS-160K's region taxonomy and normal/abnormal scheme line up with bs80k's own, but it does mean the caption text itself is not perfectly standardized, worth knowing before writing any code that assumes a single fixed sentence template per caption type.

## What this means for a combined dataset

Not built yet, this is a probe, not a design. The region taxonomy lines up closely enough with bs80k's own 13 part and side combinations that combining the two is plausible, and the one directly checked image size matches this project's own chest crop size closely. What LIBS-160K does not have, based on what was read here, is anything resembling a bounding box or a location within a larger image, its images already arrive as small region crops with a caption, same as bs80k's own region crop folders before this project's own bounding box work.

Whether LIBS-160K's images are literally sourced from bs80k: now confirmed, not an open question anymore (`context/wholebody_bbox.md`). LIBS-160K's own `wholeANT`/`wholePOST` whole body classification images are byte identical to bs80k's own `wholeBodyANT`/`wholeBodyPOST` for every one of the 3247 shared ids, checked at full scale, and its Abnormal/Normal label agrees with bs80k's own txt label 100% of the time. Beyond those 3247, LIBS-160K has 3491 more whole body ids per view that bs80k does not have at all, real new patients, not a different collection reused under new numbers.

## The region-crop layer also extends to LIBS-160K's new patients, confirmed by template matching

A back of envelope count first suggested this: LIBS-160K's region-crop tsv/jsonl total is 192456 rows, 2.53x too many to be explained by bs80k's own 2925 region-crop patients alone (2925 x 26 = 76050), but close to LIBS-160K's own claimed 6586 patients (6586 x 26 = 171236).

Checked directly, on pixels, not just row counts: took one crop from 5 different region captions (head, pelvis, left ankle joint, right knee joint, vertebrae), template matched each against the full combined 13476-candidate whole body pool (bs80k plus libs160k-only, both views, `bs80k-wholebody-bb/bounding_boxes.csv`), same masked `matchTemplate` approach this project has used throughout. All 5 found a real match (score 0.83-0.92, peak margin 0.17-0.49, a real, unambiguous peak, not noise), and 3 of those were visually confirmed side by side (`result/figures/libs_region_mapping_check.png`), the anatomical pattern, kneecap position, spine node position, pelvis shape, lines up between the LIBS crop and the located region essentially exactly, the lower score than this project's usual 0.99+ is LIBS-160K's own images being visibly grainier, not a weak match.

2 of the 3 visually confirmed matches (right knee joint, vertebrae) landed inside a `libs160k`-only whole body image, not one of bs80k's own known 2925. The region-crop layer genuinely extends to the new patients, this is not just a caption-vocabulary asset, it is a real, recoverable spatial anchor for a population beyond bs80k's own.

The scaling problem this creates: one crop against the full 13476-candidate pool took 3-5 minutes. LIBS-160K has 192456 region-crop rows total, brute forcing all of them the same way would take on the order of two years of compute, not feasible as is. Recovering this at any real scale needs a smarter approach than searching every crop against every candidate, not attempted yet, a real next decision, not a detail.

## Multi-step duplicate check, phase 1: exact MD5 (`src/dedup/phase1_md5.py`)

Prompted by a direct concern: LIBS-160K's own images already look reprocessed, not a plain copy
(`result/figures/libs_region_mapping_check.png` shows visibly grainier crops than bs80k's own),
so any duplication that survives a recompress or resize needs a method that tolerates that,
not just exact hashing. MD5 is step one, cheap, fast, catches byte identical only, this is its
ceiling, not the final answer, phase 2 (perceptual hash) and phase 3 (pixel verification of
whatever phase 2 flags) are the next steps, not run yet.

What MD5 alone already found, checked across bs80k's whole body and region layers, LIBS-160K's
whole body layer, and all 3 of LIBS-160K's own tsv splits (train_imgs.tsv, test_imgs.tsv,
valid_imgs.tsv) together, not each in isolation:

- **35.0% of LIBS-160K's 192456 region-crop rows (67405 rows) sit in an exact duplicate group**,
  32469 groups total, mostly pairs (31219), some triples/quads, one group of 6.
- **95.3% of those duplicate groups (30933 of 32469) span more than one split**, the same image
  content sitting under a different `image_id` in `train`, `test`, and/or `valid`. Only 1536
  groups are duplicated within a single split. 81 groups specifically touch both `train` and
  `test`. Anyone training and evaluating on LIBS-160K's own official split boundary is very
  likely training on images that reappear in its own test set under a different id, a real
  leakage risk, not a rare edge case, this is the majority pattern for every duplicate found.
- **0 exact cross-dataset matches in the region-crop layer**, despite `context/libs160k.md`'s
  own template-matching section above already confirming real overlap exists. Expected, not a
  contradiction: MD5 needs byte identical files, and LIBS-160K's region crops are visibly
  reprocessed. This is exactly the gap phase 2 (perceptual hash) exists to close. The whole body
  layer's own cross-dataset matches (6484 groups) are untouched by this, those files are
  genuinely byte identical, confirmed earlier in this file.

`context/dataset.md` has the bs80k-only side of this same sweep, a systematic elbowLPOST /
elbowRPOST duplication affecting 99.7% of patients, and 4 pairs of bs80k patient ids that read as
the same real patient entered twice.

## Planned approach, borrowed from MedGround

For later, not implemented yet. MedGround, Bridging the Evidence Gap in Medical Vision-Language Models with Verified Grounding Data, Zhang, Wu, Luo, Wang, Lv, submitted January 2026, https://arxiv.org/abs/2601.06847, general medical imaging, not chest X-ray specific, describes an automated pipeline turning existing segmentation resources into grounding data. Per its own abstract, read directly, not secondhand: expert segmentation masks serve as spatial anchors, the pipeline extracts localization targets, shape cues, and spatial information from those masks, a vision-language model generates natural, clinically grounded queries reflecting morphology and location, then a multi-stage verification step, formatting checks, geometry and medical rules, and visual judging, filters out inadequate samples before they enter the final dataset, named MedGround-35K, 35000 samples. Only the abstract has been read, not the full method, this is what is confirmed so far.

The intent is to copy this principle, not this project's exact implementation, for the bs80k plus LIBS-160K combination. This project's own recovered bounding boxes (`bs80k-bone_region-bb/bounding_boxes.csv`) are the direct equivalent of MedGround's segmentation masks, an existing, already computed spatial anchor per region per patient, not something that would need to be built from scratch. LIBS-160K's 39 caption templates are a starting vocabulary, not a finished one, per the text analysis above they are already somewhat inconsistent even at 39 fixed sentences, so a query synthesis step generating more varied, less template bound questions and grounded statements, followed by a verification step checking a generated claim's location against this project's own bounding box and quality metrics, reads as the natural next step once this is actually built, not before.
