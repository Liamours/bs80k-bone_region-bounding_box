# Dataset

## Source paper

Huang, Z., Pu, X., Tang, G., Ping, M., Jiang, G., Wang, M., Wei, X., Ren, Y. (2022). BS-80K: The first large open-access dataset of bone scan images. Computers in Biology and Medicine, 151, 106221. https://doi.org/10.1016/j.compbiomed.2022.106221

Copy at `reference/BS-80K.pdf`. Paper reports 82544 images from 3247 patients, 13 region-wise slices per view. Our own folder scan found 3249 ids in wholeBodyANT/POST, a small difference from the paper's 3247, not resolved here.

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

## Labels

Every folder, region and whole body alike, has a txt file with the same name as the folder. Each line lists one image filename and a label, 0 for normal, 1 for abnormal. This is a classification label, not a bounding box, keep it separate from the crop location work here.

## Existing bounding boxes, different purpose

wholeBodyANT and wholeBodyPOST each contain an `ant` and `post` subfolder. Each xml file there holds bounding box information for one whole body image in the parent folder. Per the source paper, these mark suspectable hot spots for the object detection benchmark, not where a regional crop sits inside the whole body image. Do not reuse these as the answer for this project.

## Confirmed

- Every region crop id was matched to a same-numbered whole body file with none dropped, see `src/analysis/crop_size_ratio.py` output, so the crop file number maps directly to the same number in the matching wholeBodyANT or wholeBodyPOST file
- Per the source paper section 3.2.2, region slices come from an automated pipeline, reference points, then borderlines or contour, then extraction, not a manual or arbitrary crop, see `context/method.md`

## Not yet confirmed

- Output file format for the bounding box we produce, not specified yet

## Source

Notes below given directly by the dataset owner, not yet independently checked file by file.

1. Folder name is part name plus left or right, if it applies, plus view.
2. Every folder, including region folders, has a txt file of the same name recording filename and label, 0 normal or 1 abnormal, per line.
3. wholeBodyANT and wholeBodyPOST each hold ant and post subfolders where xml files carry the whole body image's own bounding box.
4. The database keeps growing as more data is added, counts and folder lists here can go stale.
