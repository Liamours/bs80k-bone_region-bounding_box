# Method notes

## Approach

Template matching: search the raw image for the region that matches the cropped image, then report that region as a bounding box.

## Candidate tool

- OpenCV `matchTemplate`, a common tool for this kind of exact or near exact crop location problem

## Open questions

- Does the crop ever get resized, rotated, or color adjusted before saving. If yes, plain template matching will not find it and a scale or rotation aware search is needed instead
- What match score threshold separates a correct match from a wrong one, needs checking against real samples, not assumed
- Which whole body file, ANT or POST, a given region crop belongs to, needs a naming rule checked against real folders (see `context/dataset.md`)
- What format to write the output box in, not decided yet

## Status

Nothing implemented or run yet. This file is a plan, not a report.
