# Intent

## Goal

Produce a bounding box for every cropped image in the bs80k dataset, marking where that crop sits inside its raw image.

## Current state

- Raw image: whole body scan, `wholeBodyANT` / `wholeBodyPOST`
- Cropped image: a bone region already cut from the whole body scan, no coordinates kept, one folder per region, view, and chirality (see `context/dataset.md` for the full folder list and paths)

## Wanted

- Bounding box (x, y, width, height) locating the cropped bone region inside its whole body scan
- One bounding box per region crop, written to `bs80k-bone_region-bb`

## Out of scope for now

- Anything past producing the bounding box, no classification, no further annotation

## Done means

- A bounding box exists for a sample and, checked by eye or overlay, the box lines up with the crop in the raw image
