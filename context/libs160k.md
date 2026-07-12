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

Not built yet, this is a probe, not a design. The region taxonomy lines up closely enough with bs80k's own 13 part and side combinations that combining the two is plausible, and the one directly checked image size matches this project's own chest crop size closely. What LIBS-160K does not have, based on what was read here, is anything resembling a bounding box or a location within a larger image, its images already arrive as small region crops with a caption, same as bs80k's own region crop folders before this project's own bounding box work. Whether LIBS-160K's images are literally sourced from bs80k or a related but separate collection is not confirmed, not checked further here.
