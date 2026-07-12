---
name: bbox-from-crop
description: Find the bounding box of a cropped image inside its raw source image using template matching.
---

# bbox from crop

## Input

- raw image
- cropped image, a sub region of the raw image with no known coordinates

## Method

- Template match the cropped image against the raw image, for example OpenCV `matchTemplate`
- Take the best match location as the top left corner of the bounding box
- Bounding box is (x, y, width, height), width and height come from the cropped image size

## Watch for

- A crop resized or rotated relative to the raw image breaks plain template matching, check for this before trusting a match
- Low contrast or repeated texture regions can produce a wrong match, check the match score before accepting it

## Output

- One bounding box per raw/crop pair
- No accuracy numbers here until this has been run against real samples and checked
