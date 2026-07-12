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

## What this means for a combined dataset

Not built yet, this is a probe, not a design. The region taxonomy lines up closely enough with bs80k's own 13 part and side combinations that combining the two is plausible, and the one directly checked image size matches this project's own chest crop size closely. What LIBS-160K does not have, based on what was read here, is anything resembling a bounding box or a location within a larger image, its images already arrive as small region crops with a caption, same as bs80k's own region crop folders before this project's own bounding box work. Whether LIBS-160K's images are literally sourced from bs80k or a related but separate collection is not confirmed, not checked further here.
