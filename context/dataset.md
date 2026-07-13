# Dataset

## Source paper

Huang, Z., Pu, X., Tang, G., Ping, M., Jiang, G., Wang, M., Wei, X., Ren, Y. (2022). BS-80K: The first large open-access dataset of bone scan images. Computers in Biology and Medicine, 151, 106221. https://doi.org/10.1016/j.compbiomed.2022.106221

Copy at `reference/BS-80K.pdf`, gitignored, kept local only, see `.gitignore`. Full text read page by page for this note.

Paper reports 82544 images from 3247 patients. Our own count of `wholeBodyANT`/`wholeBodyPOST` jpg files is also 3247 each, an earlier note in this file said 3249 and called it an unresolved difference, that number came from counting every entry in the folder including the `ant`/`post` xml subfolder and the `.txt` label file as if they were images, filtering to `*.jpg` gives 3247, matching the paper exactly.

Segmentation method the paper builds on: J.-Y. Huang, P.-F. Kao, Y.-S. Chen, A set of image processing algorithms for computer-aided diagnosis in nuclear medicine whole body bone scan images, IEEE Trans. Nucl. Sci. 54 (3) (2007) 514-522. Local copy at `reference/20070329_TNS-00104-2006_manuscript.pdf`, gitignored, kept local only. Read in full, page by page, see `context/method.md` for the reference point and segmentation detail pulled from it.

## Acquisition

- Dual head gamma camera, low energy high resolution parallel hole collimator, energy window centered on the 99mTc 140 keV peak, 20% window width
- Whole body image format is reported as 1024 x 256, matches our own measured wholeBodyANT/POST samples, PIL size (256, 1024), width by height
- Source images were DICOM, converted to JPEG first for de-identification and for compatibility with image libraries, region segmentation happens after that conversion, so a region crop and its whole body source come from the same already-JPEG image, not from two separately compressed versions
- The 2007 segmentation method paper (Huang, Kao, Chen, cited above) reports its own acquisition, a different and much smaller hospital dataset, not BS-80K itself, at the same 1024 x 256 resolution and 16-bit grayscale depth for the raw acquired counts (GE Infinia dual head camera, Buddhist Tzu Chi General Hospital, LEHR collimators, 20% window centered on the 140 keV 99mTc peak). This 16-bit figure describes that paper's own source data, not confirmed as BS-80K's own bit depth before its DICOM to JPEG conversion, JPEG itself is 8-bit per channel regardless of original scanner depth. See `context/method.md` for detail

## Locations

- Input (raw, mainly): `C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw`
- Output (ours): `C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb`

Both live outside this repo folder.

## Folder naming

Part name + left or right (if it applies) + view. Example: `kneeLANT` = knee + Left + Anterior.

## Region folders, cropped bone regions by view and chirality

- ankleLANT, ankleLPOST, ankleRANT, ankleRPOST
- chestLANT, chestLPOST, chestRANT, chestRPOST
- elbowLANT, elbowLPOST, elbowRANT, elbowRPOST
- headANT, headPOST
- kneeLANT, kneeLPOST, kneeRANT, kneeRPOST
- pelvisANT, pelvisPOST
- shoLANT, shoLPOST, shoRANT, shoRPOST
- vertbraANT, vertbraPOST

## Whole body folders, raw and uncropped

- wholeBodyANT
- wholeBodyPOST

These are the raw side of the raw/crop pair. Each region folder above holds crops taken from these.

## Why region folders have fewer images than whole body folders

Paper section 3.3: not every whole body image could be segmented automatically, so region slices exist for 2925 patients instead of the full patient count. This matches our own `component_coverage.xlsx`, every one of the 26 region folders has exactly 2925 ids, none missing among themselves, just fewer than the whole body folders.

## Labels

Every folder, region and whole body alike, has a txt file with the same name as the folder. Each line lists one image filename and a label, 0 for normal, 1 for abnormal.

Per paper section 3.3, this label is not independent of the bounding box annotations, it is derived from them. A physician draws a box around each nidus, suspected malignant, or physiological hot spot, benign, on the whole body image. A whole body image is labeled abnormal if it contains any nidus box. A region slice is labeled abnormal if any nidus box falls inside that region, otherwise normal.

Checked directly against paper Table 2: headANT, headPOST, vertbraANT, vertbraPOST, pelvisANT, pelvisPOST normal/abnormal counts from our own txt files match the paper exactly, for example head is 2690/235 on both views in both the paper and our files. wholeBodyANT and wholeBodyPOST are close but not identical, our files give 2064/1183 and 2194/1053, the paper gives 2065/1182 and 2195/1052, one sample's label differs on each view. Plausibly a later relabel, the paper itself says annotations are reviewed multiple times and the database keeps growing, not chased further here.

## Existing bounding boxes, different purpose

wholeBodyANT and wholeBodyPOST each contain an `ant` and `post` subfolder. Each xml file there holds bounding box information for one whole body image in the parent folder. Per paper section 3.3 and Fig. 5, purple boxes mark niduses, suspected malignant, green boxes mark physiological hot spots, benign. These are the object detection ground truth for the paper's own hot spot benchmark. They do not mark where a regional crop sits inside the whole body image, do not reuse them as the answer for this project.

The paper does not publish or mention a bounding box for a region slice's location inside its whole body source image anywhere. That box is only what this project is trying to recover, it is not sitting in the dataset waiting to be read.

## Confirmed

- Every region crop id was matched to a same-numbered whole body file with none dropped, see `src/analysis/crop_size_ratio.py` output, so the crop file number maps directly to the same number in the matching wholeBodyANT or wholeBodyPOST file
- Per the source paper section 3.2.2, region slices come from an automated pipeline, reference points, then borderlines or contour, then extraction, not a manual or arbitrary crop, see `context/method.md`
- Per the source paper section 3.2, the whole body image is converted to JPEG before segmentation, so a region crop is not a second, separately compressed copy of the whole body image
- Whole body and region jpg counts both match the paper's published numbers exactly once non-image files are excluded from the folder count, and head/vertebra/pelvis label counts match paper Table 2 exactly, see Labels above

## Not yet confirmed

- Output file format for the bounding box we produce, not specified yet
- The one-sample label difference on whole body ANT and POST against paper Table 2, see Labels above

## Two real data quality findings from a full MD5 duplicate sweep (`src/dedup/phase1_md5.py`)

Run to check bs80k and LIBS-160K against each other for exact duplicates, checked bs80k's own
region crop layer too, not just the cross-dataset question. Found two different things, cleanly
separated by whether the duplicate pair shares one patient id or two:

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
  patient entered under two different ids, a real bs80k data integrity issue, not something
  either this project or LIBS-160K introduced. `bounding_boxes.csv` and everything built from it
  currently counts these 4 real patients as 8, not corrected yet, a decision still open.

See `context/libs160k.md` for the LIBS-160K side of the same sweep (much larger intra-dataset
duplication, and why 0 exact cross-dataset region-crop matches were found despite confirmed real
overlap).

## Source

Notes below given directly by the dataset owner, not yet independently checked file by file.

1. Folder name is part name plus left or right, if it applies, plus view.
2. Every folder, including region folders, has a txt file of the same name recording filename and label, 0 normal or 1 abnormal, per line.
3. wholeBodyANT and wholeBodyPOST each hold ant and post subfolders where xml files carry the whole body image's own bounding box.
4. The database keeps growing as more data is added, counts and folder lists here can go stale.
