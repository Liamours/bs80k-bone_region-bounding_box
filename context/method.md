# Method notes

## Approach

Template matching: search the raw image for the region that matches the cropped image, then report that region as a bounding box.

## Candidate tool

- OpenCV `matchTemplate`, a common tool for this kind of exact or near exact crop location problem

## How the crops were actually made

Per the source paper (`context/dataset.md`, section 3.2.2), the region slices come from a 3 stage pipeline on each whole body image, not a manual or arbitrary crop:

1. Reference points for head, shoulder, vertebrae, pelvis are located, then a correction step relocates any wrong points
2. Borderlines for head, elbow, knee joints, ankle joints are set from those points, vertebrae instead gets an irregular contour from a border following technique on a thresholded image
3. Region slices are extracted from the whole body image using those borderlines or the contour

The paper's own wording only names head, elbow, knee, ankle for the borderline step, and vertebrae for the contour step. Shoulder, chest, and pelvis get reference points in stage 1 but are not named again in stage 2, the paper does not say how their box edges are actually set. Worth keeping in mind if a single fixed procedure is assumed for every region, the source text itself does not fully spell that out.

This backs two things seen in our own output:

- `result/dataset_samples/vertbra_samples.png` shows a jagged, non-rectangular silhouette, consistent with the contour based extraction described for vertebrae, unlike the plain rectangles seen for other regions
- `result/tables/crop_size_ratio.xlsx` found height ratio to whole body near constant per region, std near 0 for several, while width ratio varies more, consistent with reference-point-driven, patient-specific borderlines rather than a fixed-size template

Since slices are described as "extracted", not resized or warped, plain pixel-value template matching is a reasonable starting point for the rectangular regions. Vertebrae is the exception, its crop has non-rectangular content, so an exact rectangle match may score lower there even at the correct location.

The paper also says the whole body image is converted from DICOM to JPEG before this segmentation step. A region crop and its whole body source both come from the same already-JPEG image, so we are not matching against two independently compressed versions of the same picture.

## Segmentation method behind the pipeline

Cited as [37] in the source paper: J.-Y. Huang, P.-F. Kao, Y.-S. Chen, A set of image processing algorithms for computer-aided diagnosis in nuclear medicine whole body bone scan images, IEEE Trans. Nucl. Sci. 54 (3) (2007) 514-522. Not read yet, worth pulling if the reference-point and borderline steps need to be reproduced rather than reverse engineered by template matching alone.

## Open questions

- What match score threshold separates a correct match from a wrong one, needs checking against real samples, not assumed
- What format to write the output box in, not decided yet
- How shoulder, chest, and pelvis box edges are actually set, the source paper names reference points for them but not a borderline step, see above

## Status

Nothing implemented or run yet. This file is a plan, not a report.
