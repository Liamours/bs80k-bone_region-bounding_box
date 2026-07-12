# Method notes

## Approach

Template matching: search the raw image for the region that matches the cropped image, then report that region as a bounding box.

## Candidate tool

- OpenCV `matchTemplate`, a common tool for this kind of exact or near exact crop location problem

## How the crops were actually made

Per the source paper (`context/dataset.md`, section 3.2.2), the region slices come from a 3 stage pipeline on each whole body image, not a manual or arbitrary crop:

1. Reference points for head, shoulder, vertebrae, pelvis are located, then a correction step relocates any wrong points
2. Borderlines for head, elbow, knee, ankle are set from those points, vertebrae instead gets an irregular contour from a border following technique on a thresholded image
3. Region slices are extracted from the whole body image using those borderlines or the contour

This backs two things seen in our own output:

- `result/analysis/vertbra_samples.png` shows a jagged, non-rectangular silhouette, consistent with the contour based extraction described for vertebrae, unlike the plain rectangles seen for other regions
- `result/analysis/crop_size_ratio.py` found height ratio to whole body near constant per region (std near 0 for several) while width ratio varies more, consistent with reference-point-driven, patient-specific borderlines rather than a fixed-size template

Since slices are described as "extracted", not resized or warped, plain pixel-value template matching is a reasonable starting point for the rectangular regions. Vertebrae is the exception, its crop has non-rectangular content, so an exact rectangle match may score lower there even at the correct location.

## Open questions

- What match score threshold separates a correct match from a wrong one, needs checking against real samples, not assumed
- What format to write the output box in, not decided yet
- Exact reference method behind the pipeline is cited as [37] in the source paper, not retrieved separately here

## Status

Nothing implemented or run yet. This file is a plan, not a report.
