# Method notes

## Summary of experiments

24 of 26 folders are solved and confirmed reliable. Shoulder is the one unresolved folder. Anchoring its search on vertebra's own matched position resolves most gross mislocalization, wrong side of the spine, or landing near the pelvis, knee, or foot. What is left is a harder, more intrinsic precision problem within the right neighborhood, tied to how little real signal the shoulder crop actually contains. Detail and backing for every row is in the sections that follow.

| # | Experiment | Tested | Result |
|---|---|---|---|
| 1 | Plain `matchTemplate` baseline | Does plain cross correlation find each crop | 18 of 26 folders perfect immediately, vertebra, chest, shoulder scored lower |
| 2 | Second peak margin | Is the low score ambiguity, many similar spots, or a content mismatch | Vertebra: confident single peak, not ambiguity. Chest and shoulder: genuinely ambiguous search |
| 3 | Head to pelvis search band | Does restricting where to search fix chest/shoulder | No, made shoulder worse, likely cut off the true spot |
| 4 | Visual inspection of one shoulder case | What does a failure actually look like | Crop is 66% black background, drowning out the real signal |
| 5 | Masked search, exclude background | Does excluding background from correlation fix it | Fixes chest, does not fix shoulder |
| 6 | Fixed threshold sweep, 2 to 50 | Does a stricter signal cutoff help shoulder | No, gets worse, also caused empty mask crashes at high thresholds |
| 7 | Percentile mask sweep, 50th to 95th | Same idea, relative instead of fixed | Same negative result, no crashes this time |
| 8 | Mask the evaluation too | Was scoring unfairly penalizing a crop's own black padding | Yes, chest and vertebra jump to about 1.0, this solved 24 of 26 |
| 9 | Verify masked eval is not free credit | Could a wrong window also score about 1.0 from the masking trick | No, deliberately wrong windows still score near 0, legitimate |
| 10 | Full dataset run, 76050 rows | Does the 20 sample finding hold at scale | Yes, confirms 24 of 26 solved, found 2 genuine failures from near blank crops |
| 11 | Positional outlier detection | Can position alone, no quality metric, flag wrong placements | Everything except shoulder under 4% outlier rate, shoulder 13-30% |
| 12 | Eyeballed flagged outliers | Are flagged outliers really wrong | Confirmed, one case landed on the knee instead of the shoulder |
| 13 | Where shoulder outliers actually land | Is there one specific false attractor | No, scattered, pelvis 31%, knee 39%, feet 24% |
| 14 | Centroid relative outlier check | Does normalizing to each patient's own body center change the story | No, confirms the same finding independently, plus a clean per-region offset table |
| 15 | Output format research | How should the bounding boxes be stored | CSV, settled |
| 16 | Rotation and scale invariant matching | Is shoulder's problem a fixed rotation/scale mismatch | No, quality barely changes, winning angles do not agree with each other |
| 17 | Vertebra anchored search band | Does anchoring on vertebra's own top edge, not head's, help | Real but partial improvement, all 4 folders move the right direction, still far from solved |
| 18 | Left-right confusion, checked separately | Is part of the problem a left/right mixup | Real but small (5-15%), mostly the same failures as landing far away, not a separate problem, and the vertebra band above resolves nearly all of it. What remains is precision within the right neighborhood, not location |

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

Cited as [37] in the source paper: J.-Y. Huang, P.-F. Kao, Y.-S. Chen, A set of image processing algorithms for computer-aided diagnosis in nuclear medicine whole body bone scan images, IEEE Trans. Nucl. Sci. 54 (3) (2007) 514-522. Local copy at `reference/20070329_TNS-00104-2006_manuscript.pdf`, gitignored, kept local only. Read in full, page by page. Detail pulled from it is split into the sections below.

Ref 37's own goal is automatic lesion detection, not producing a public dataset, and its own preprocessing and test data are a separate pipeline and a separate, smaller dataset from BS-80K. Ref 37's own preprocessing (noise removal outside the body, Gaussian smoothing, histogram equalization, section 2.1 of that paper) is described in service of that paper's own lesion detection work. BS-80K's own paper only describes DICOM to JPEG conversion before segmentation, nothing about fuzzy thresholding or the rest of ref 37's preprocessing chain. The two are the same segmentation method by citation only, this repo has not confirmed BS-80K's own pipeline reproduces ref 37's preprocessing step for step, only that BS-80K cites ref 37 as the basis for its region segmentation.

## Preprocessing and image versions in ref 37

Ref 37 works with three distinct versions of each whole body image, this matters for the sections below on which pixels end up in a final region crop:

- I_org (AP/PA): the image as received by the algorithm, already brightness/contrast adjusted by a technologist per that paper's own account (section 1)
- I_pre (AP/PA): I_org after (a) thresholding out noise outside the body silhouette at the histogram valley, (b) a 5x5 Gaussian smoothing mask, sigma 1.4, applied inside the body region, (c) histogram equalization targeting a max gray level of 255, an 8-bit scale (eq. 2), (d) cropping to the true body frame by detecting the leftmost, rightmost, topmost, and bottommost body extremities
- I_fuzzy (AP/PA): I_pre after fuzzy sets histogram thresholding (section 2.2.1, method from Tobias and Seara 2002), which suppresses soft tissue and reserves the thick, wide bones, shoulder, pelvis, vertebra, for locating reference points

Reference points are located mostly on I_fuzzy. Arm bottom points are located on I_pre instead because low activity limbs get over suppressed by the fuzzy thresholding (section 2.2.2.5). Locating happens mainly on the posterior view, I_PA^fuzzy, because vertebrae and sacrum show up better there, some pelvis points use the anterior view, I_AP^fuzzy, because iliac activity shows up better there (section 2.2.2.4).

## Reference point heuristics, ref 37 section 2.2.2

This fills the gap this file already flagged above, how shoulder, chest, and pelvis box edges are actually set. BS-80K's own paper cites ref 37 as its basis but does not spell these out itself, what follows is ref 37's own concrete method, offered as the most likely instantiation, not confirmed as exactly what BS-80K ran. All scans below are on the fuzzy thresholded image unless stated otherwise. "Width" means rightmost non-zero pixel column minus leftmost non-zero pixel column in a given row.

Neck (2.2.2.1): scan every row from the image top down to 25% of image height, compute width per row, the row with the minimum width gives the neck points, its own leftmost and rightmost non-zero pixels.

Shoulder apex (2.2.2.2): from the neck row down to 25% image height, compute the distance from the left image border to the first non-zero pixel per row. Where the current row's distance differs from the previous row's by more than 3 times the previous row's distance, that row is the left shoulder apex. The right apex is found the same way from the right border.

Four more points build the shoulder to thorax boundary, found by a minimum gray level pixel search anchored at the apex, then a diagonal pixel by pixel screen back toward the apex's own row or column. The exact scan direction here (row wise vs column wise) is genuinely ambiguous in the paper's own wording, it calls both a row scan and what looks like a column scan "horizontal", likely a translation artifact. What is clear: these four points together with the two apex points are what separates the shoulder wedge from the thorax underneath it, and this is the concrete answer to how a shoulder or chest edge gets set. Reimplementing the exact scan direction would need checking against the paper's own figure 7a rather than assuming from text alone.

Vertebra (2.2.2.3): on I_PA^fuzzy, scan rows from 30% to 50% of image height, minimum width row gives the two vertebra points. These are copied onto I_AP^fuzzy at the same pixel coordinates, but the paper's own wording swaps the left/right label when doing so, consistent with anterior and posterior views showing opposite sides of the body from a viewer's position. Worth checking empirically against this repo's own paired ANT/POST samples once bounding boxes exist for the same patient on both views, not something this read can confirm on its own, a crop by itself does not carry the whole image coordinates needed to check it.

Pelvis (2.2.2.4), six points, alternating between AP and PA fuzzy images:
- Top left/top right: downward from the vertebra point, compute the horizontal distance between the shoulder apex column and the first non-zero pixel per row, same 3x jump rule as the shoulder apex, marks the iliac crest flare
- Left/right extremity (AP view, mapped from the top points): starting at 50% image height, compute the horizontal distance between the shoulder apex column and the first non-zero pixel found scanning from the far edge inward per row, the row with the minimum distance gives the extremity point. The paper does not state how far down this scan continues, only where it starts
- Bottom left/bottom right: starting at 60% image height, accumulate the count of non-zero pixels between the top point and the vertebra point per row, the row where that count reaches zero is the bottom point
- Two more points exist purely to keep leg segmentation from bleeding into the pelvis, defined algebraically in the paper's eq. 11 as the extremity point's coordinates shifted by minus/plus 10 pixels horizontally and minus 20 vertically. Both formulas as printed reference the same, left, extremity point, the second one would be expected to reference the right extremity point by symmetry with everything else in the paper, this reads as a likely typo in the source rather than something to silently correct without the original figure

Arms (2.2.2.5): on I_PA^pre, not fuzzy, see above, from the pelvis top left point down to 60% image height, accumulate the count of non-zero pixels between the left image border and the pelvis extremity column per row, the row where that count reaches zero is the arm bottom point. The right side is found the same way from the right border.

## How each region boundary is derived, ref 37 section 2.2.3

Order of operations in the paper: head, then arms and shoulders, then pelvis, then legs, then vertebra, then the rest of the trunk, thorax, always worked out on the posterior fuzzy image first, then mapped onto the anterior fuzzy image.

- Head: plain rectangle, corners are the neck points and the head's own left/right extremities
- Shoulder: a small polygon, not a rectangle, bounded by the apex point and the two extra points on that side from the section above
- Humerus, elbow, radius/ulna, palm: the whole arm is first confined to a polygon (apex, both extra points, the two pelvis bleed-guard points, arm bottom point), then inside that polygon the paper scans width per row between 30% and 70% of arm height, the row of maximum width is always the elbow, rows above and below where width falls under 30% of that maximum mark the elbow's own top and bottom edge. Humerus is everything above the elbow's top edge, back to the shoulder. The paper states radius/ulna and palm are segmented "in the same way" below the elbow but does not give the specific width thresholds separating radius/ulna from palm, this sub split is not spelled out
- Pelvis: the six point polygon from the section above, not a plain rectangle. A bladder/urine sub-region inside that box is separately found by region growing from the brightest pixel in a fixed sub-window and excluded, this only affects the lesion detection step downstream, it does not change the pelvis box itself
- Legs, femur/knee/tibia-fibula/foot: confined by the two pelvis bottom points and the two bleed-guard points, split left/right by a center line, then the same max-width-row-is-the-joint logic as the arm locates the knee, femur is above the knee, tibia/fibula and foot are segmented "in a similar way" below the knee, again without a stated split criterion between the two
- Vertebra: the only region using pixel by pixel edge following rather than a box. The lower part, between the vertebra points and the pelvis, is just the fuzzy threshold result on that local strip. The upper part, between the vertebra points and the neck, where ribs and kidneys overlap the spine, uses an explicit rule (2.2.3.5): starting at the vertebra reference point, move one row up at a time. If the pixel straight above the current edge point is background, the new edge for that row is the first non-zero pixel found scanning inward from there. If that pixel is not background, scan outward for the first background pixel, accept it as the new edge only if it is within 4 pixels of the current edge column, otherwise keep the current column unchanged for that row. The other edge mirrors this. The paper shows this working on a scoliosis case too, not just a straight spine. This backs the jagged, non-rectangular silhouette already seen in this repo's own `vertbra_samples.png`
- Thorax: whatever trunk area is left after the vertebra's own jagged contour is removed, split into four quadrants by that contour and a horizontal midline, used to detect and region-grow-remove kidney urine activity the same way as pelvis. The remaining thorax shape is an irregular complement of a rectangle minus the vertebra silhouette, not a plain box either, though closer to one than vertebra is

23 named parts per view, head, shoulder, humerus, elbow, radius/ulna, palm each left/right, vertebra, thorax left/right, pelvis, femur, knee, tibia/fibula, foot each left/right, times 2 views is the paper's own reported "46 homogeneous regions" (abstract, section 2), confirmed by counting the labels in the paper's own Fig. 15 (23 numbered regions per view). "46" is not 46 distinct anatomical names, it is 23 names times 2 views.

## Original vs preprocessed pixels in the final crop

For head (Fig. 9), the shoulder/arm group (Fig. 10), and legs (Fig. 12), the paper explicitly shows the segmented boundary computed on I_fuzzy then mapped back to image I_org, the figure captions say so directly and the figures show org-looking grayscale content inside the box, not the suppressed fuzzy content. For pelvis (Fig. 11), vertebra (Fig. 13), and thorax (Fig. 14), the paper's own figures and captions do not state or show a comparable mapping back to I_org. The paper does not say these three work differently, it also does not confirm they work the same way, this is an open point for these three regions specifically, not a settled fact for the whole pipeline.

This is a different claim from anything in BS-80K's own paper, context/dataset.md already notes BS-80K's paper only describes reference points, then borderlines or contour, then extraction, without saying which pixel version is copied into the final released crop. Ref 37 supports the assumption that a final region crop should carry original image gray levels rather than fuzzy thresholded ones, for the head/arm/leg groups specifically, it does not settle the question for pelvis/vertebra/thorax, and it is ref 37's own practice, not a restatement from BS-80K's own paper. Keep expecting plain pixel value template matching to work for the regions ref 37 confirms, keep the pelvis/vertebra/thorax case as unconfirmed rather than assumed identical.

## Region name mapping attempt, ref 37 to BS-80K's 26 folders

BS-80K's 26 folders reduce to 13 part+side names (see context/dataset.md): ankleL, ankleR, chestL, chestR, elbowL, elbowR, head, kneeL, kneeR, pelvis, shoL, shoR, vertbra. Ref 37's 13 named parts (its own Table 1): head, vertebra, shoulder, humerus, elbow, radius/ulna, palm, thorax, pelvis, femur, knee, tibia/fibula, foot.

Reasonably clean name and role matches:
- head to head
- vertbra to vertebra
- pelvis to pelvis
- shoL/shoR to shoulder, with the caveat that ref 37's own "shoulder" is a small wedge distinct from its own separate "humerus", BS-80K's coarser 26 folder scheme may have merged shoulder and humerus into one "sho" crop, not confirmed here
- elbowL/elbowR to elbow
- chestL/chestR to thorax, name differs but role matches, rest of the trunk after vertebra is removed, not confirmed by either paper directly
- kneeL/kneeR to knee

No match found:
- ankleL/ankleR has no counterpart in ref 37's named list at all. Ref 37's leg chain is femur, knee, tibia/fibula, foot, with knee as the one named joint region between the two long bones, there is no separately named ankle joint region the way "elbow" is named between humerus and radius/ulna. BS-80K's "ankle" could be a renamed tibia/fibula crop, or an independent joint region BS-80K's own pipeline adds that ref 37 does not describe, do not assume either without more evidence
- ref 37 also names humerus, radius/ulna, palm, and femur as distinct regions, none of these appear as BS-80K folders at all, hands and palms are entirely absent from BS-80K's public 26 folders. Whether BS-80K's own pipeline merges these into its coarser folders or drops them from the public release is not stated in either paper

## Other details from ref 37 relevant to this project

- Resolution 1024 x 256 pixels (section 3.1), matches this repo's own measured wholeBodyANT/POST samples and BS-80K's own reported resolution, already in context/dataset.md, this is independent corroboration from a second source, not new information
- Ref 37's own acquisition, its own 2004 test set, GE Infinia dual head camera, Buddhist Tzu Chi General Hospital, LEHR collimators, 20% window centered on the 140 keV 99mTc peak, section 3.1, reports 16-bit grayscale depth for the raw acquired counts. This is ref 37's own hospital data, a different and smaller dataset than BS-80K, not a report about BS-80K's own images. Added to context/dataset.md's Acquisition section with that caveat attached
- Ref 37's own histogram equalization step targets a max gray level of 255, an 8-bit scale, so by the point its algorithm scans for reference points the working image is already effectively 8-bit regardless of original acquisition depth, consistent with matching against 8-bit JPEG data being a reasonable starting point
- Ref 37 reports its own small scale segmentation success rate, head/hand/vertebra/thorax segmented without error and pelvis/legs mis-segmented in 3 of 162 images due to urine contamination (end of section 2.2.3), a different, much smaller, and earlier figure than BS-80K's own reported coverage, 2925 of 3247 patients, already in context/dataset.md, do not conflate the two

## Scoring a candidate match

No independently labeled ground truth (x, y) exists for a region crop's location (`context/dataset.md`), so a predicted box can only be checked by content, cutting a w by h window out of the raw whole body image at the predicted (x, y) and comparing it against the real crop file already on disk.

`src/eval/crop_match_metrics.py` implements this comparison. Given two same-size grayscale arrays it reports six metrics, picked to be non-redundant, each catching a failure mode the others can miss:

- near_exact_fraction, share of pixels within a small gray level tolerance, catches gross misalignment directly, a literal exact match at tolerance 0 is too strict given real photon count noise
- mae and rmse, mean absolute and root mean squared pixel difference, rmse is pulled far more than mae by one bright misaligned hot spot, so a wide gap between the two points at a small region driving the error rather than a uniform offset
- ssim, structural similarity, compares local luminance, contrast, and structure window by window instead of as one global average, catches local distortion a single correlation or histogram number can miss
- pearson_corr, correlation between flattened pixel values, stays near 1.0 under a uniform brightness or contrast offset between the two windows, which is also its blind spot, a wrong crop rescaled to a similar pattern could still score high here
- hist_intersection, normalized histogram intersection, ignores pixel position and compares only the mix of gray values present, catches a wrong region even when position sensitive metrics look reasonable, and stays high under a small position offset that hurts those same metrics

Self test in the file's own `__main__` block uses synthetic 40x40 arrays, not real dataset images, since no template matching implementation exists yet to produce a real candidate window. Run and confirmed: identical arrays score perfectly on all six metrics, a shifted, noisy, rescaled, single hot spot pixel, or unrelated array each score worse on at least one metric. For example a uniform brightness/contrast rescale left pearson_corr at 1.0 but dropped hist_intersection to 0.0 and pushed mae past 38, and a single bright misaligned pixel left mae near 0.15 while rmse rose to 6.18, over 5 times mae.

## Baseline results

`src/matching/baseline_template_match.py` is a first, plain baseline: OpenCV `matchTemplate` (`TM_CCOEFF_NORMED`) searches the matching whole body image for each region crop, then `src/eval/crop_match_metrics.py` scores the window found against the real crop. Run on a fixed random sample of 20 ids (seed 0, present in all 26 folders) across all 26 region folders, 520 matches total, results in `result/tables/baseline_template_match.xlsx`.

The 20-sample means split cleanly into three groups, not one uniform result:

- ankle, elbow, head, knee, pelvis (18 of 26 folders): near-exact fraction 1.0 and ssim above 0.98 in every one of these components, the plain baseline finds the exact source location essentially every time
- vertebra (2 folders): near-exact fraction 0.62-0.64, ssim 0.68-0.73, consistently in the right place but not pixel-exact, matches the contour based, non-rectangular extraction already noted above
- chest, shoulder (6 folders): near-exact fraction 0.34-0.49, ssim 0.17-0.36, clearly worse than every other region

## Second peak margin, telling shape mismatch apart from a genuinely ambiguous search

`second_peak_score` in `src/matching/baseline_template_match.py` suppresses a template sized window around the best match in the correlation surface, then takes the best remaining value elsewhere in the image. `peak_margin` is the best score minus that value, a small margin means other locations in the whole body image score nearly as well as the one picked, a large margin means the picked location stands out clearly.

Real numbers from the same 20-id run, mean peak_margin per component:

- ankle, elbow, head, knee, pelvis: 0.23-0.63, comfortably separated
- vertebra: 0.26-0.45, also comfortably separated, in the same range as the clean regions above
- chest: 0.03-0.05
- shoulder: 0.02-0.03, the smallest margin of any region by a wide margin itself

This separates two different problems that looked similar from ssim alone. Vertebra has a clear, unambiguous peak, its imperfect pixel match is a shape problem, the crop itself is not a rectangle, so a rectangular window can never score perfectly even at the right location, consistent with the contour based extraction already noted above. Chest and shoulder have no clear peak at all, the best scoring location is barely better than other candidates elsewhere in the same whole body image. That points to genuine search ambiguity, plausibly repetitive rib or torso texture producing several similarly scoring locations, rather than, or in addition to, a shape mismatch. The earlier note in this file guessing a polygon shaped crop as the likely cause undersold this, a low peak margin would happen even for a correctly shaped rectangular crop if the surrounding content is repetitive enough that plain pixel correlation cannot tell candidate locations apart.

## Constrained search attempt, band between head and pelvis

Tried the constraining idea from the open question below: `src/matching/constrained_chest_shoulder.py` matches head and pelvis first, both essentially exact on their own, then restricts the chest/shoulder search to the vertical band between the head match's bottom edge and the pelvis match's top edge, same 20 ids, same whole body images, results in `result/tables/constrained_chest_shoulder.xlsx`.

The band applied on all 160 attempts, never too small to fit the crop. Comparing to the full image baseline above:

- peak_margin went up for every one of the 8 components, chest roughly 2 to 3 times larger, shoulder up to 5 times larger in one case, confirming the band does cut out competing candidate locations
- match quality did not follow. chest is a mixed picture, ssim improved for chestLANT and chestRANT, got worse for chestLPOST and chestRPOST. shoulder got clearly worse across all 4 folders, near-exact fraction dropped from roughly 0.40-0.47 down to roughly 0.15-0.18, ssim from roughly 0.23-0.28 down to roughly 0.12-0.13

A higher peak margin here did not mean a better match. A band derived this way is not a working fix as tried.

## Inspecting one shoulder case directly

Looked at shoRANT, id 1821, both matches drawn on the whole body image plus the real crop and both extracted windows side by side.

The unconstrained match landed near the ankles, an anatomically wrong location that happened to score slightly higher than anywhere in the true shoulder area, this is what a low peak margin looks like in practice, not a close call between two reasonable locations but the true weak signal losing to unrelated noise elsewhere in the image. The constrained match landed right after the head, anatomically plausible this time. But neither extracted window actually reproduces the real crop's own look, a large solid black wedge next to a brighter textured area. Checked directly: 66% of this crop's pixels are at or near zero.

That number changes the read on chest and shoulder. A template that is two thirds flat background has very little content left to actually discriminate one candidate location from another, normalized cross correlation degrades on a mostly blank template regardless of where the search is allowed to look. This is not simply the earlier ambiguity story (many locations look equally plausible) and not simply the earlier band story (the true spot got cut out), it looks more like the template itself is too information-poor for plain pixel correlation to reliably locate anywhere, constrained or not. Revises both the "search ambiguity, not shape" framing above and the band width fix attempted above, neither addresses a mostly-blank template.

The vertebra comparison from earlier still holds for a different reason, vertebra's crop is a thin, tall, fully textured strip, its problem is a real shape mismatch against a rectangle, not a blank template.

## Masked template matching

`src/matching/core.py`'s `locate` now takes an optional mask, passed straight to OpenCV `matchTemplate`. `background_mask` builds one from a crop, 255 above a small threshold, 0 at or near zero. `src/matching/masked_template_match.py` reruns the same 20 ids across all 26 folders with the mask applied, results in `result/tables/masked_template_match.xlsx`.

First attempt produced garbage, `x=-1` and `nan` scores, on every ankle sample and others. A masked search window that lands on a fully blank stretch of the whole body image divides by zero in the normalization step, `matchTemplate` returns `nan`/`inf` there, and plain `minMaxLoc` happily reports one of those as the best score. `locate` now replaces any non finite value with `-inf` before calling `minMaxLoc` whenever a mask is given, confirmed by hand on one case (id 1898, ankleLANT) that this recovers the same answer the unmasked search already found there.

With that fix, real before and after numbers for chest, shoulder, vertebra, mean over the same 20 ids:

- chest: near-exact fraction roughly doubles, 0.34-0.49 to 0.74-0.79, ssim roughly doubles too, 0.17-0.36 to 0.68-0.77, peak margin jumps by 10x or more, 0.03-0.05 to 0.50-0.66. Masking is a real, substantial fix for chest, confirming the mostly-blank-template read above
- shoulder: did not improve, near-exact fraction and ssim both slightly worse than unmasked, 0.40-0.47 down to 0.32-0.35, 0.23-0.28 down to 0.22-0.24, peak margin ticks up, 0.02-0.03 to 0.05-0.08, but stays far below every other region including chest now. Whatever is wrong with shoulder is not simply background swamping the signal the way it was for chest, unresolved
- vertebra: near-exact fraction and ssim are exactly unchanged, this metric compares the full crop against the full extracted window and the mask only affects the search step, not this comparison, so an unchanged number here is expected, not a sign masking failed. peak margin roughly doubles, 0.26-0.45 to 0.41-0.72, the search itself is far more confident it found the right spot, consistent with vertebra's issue being shape, not search confidence

One more note worth keeping: ankle's own background fraction, computed the same way, averages roughly 0.62-0.65, higher than shoulder's roughly 0.51-0.54, yet ankle already matched perfectly unmasked. Background fraction alone does not decide whether a region is hard to match, shoulder is not simply "more blank than average."

## Inspecting one shoulder case's mask directly

Same case as before, shoRANT id 1821, this time with `background_mask` applied (threshold 2). The mask itself is shaped correctly, a clean diagonal split matching the real crop's own black wedge boundary. But it is speckled with extra small holes scattered inside the nominal signal region too, and the masked search still only reaches score 0.639 at its own best location, whose extracted window still does not show the crop's distinctive diagonal edge, so this still is not the confirmed true location.

Checked the crop's actual pixel values directly: of 3060 pixels, only 73 (2.4%) are clearly signal, above 20. 967 pixels (32%) sit in a 3 to 20 range, this project's mask threshold of 2 counts all of those as signal, but that range reads as photon count noise floor, not real anatomical signal, given the clearly-signal pixels only go up to 38. The mask built for this case is not wrong in shape, but it is diluted, roughly a third of what it marks as usable is closer to noise than to bone signal. This is a plausible reason masking helped chest, whose signal region is presumably less speckled, and did not help shoulder.

## Mask threshold sweep, a real answer, and a second problem found

Tried it: `src/matching/threshold_sweep.py` reruns the same 20 ids at thresholds 2, 10, 20, 30, 50, for all of chest and shoulder, both vertebra folders, and kneeLANT as a control that already worked perfectly. Results in `result/tables/threshold_sweep.xlsx`.

A higher threshold does not help shoulder. near-exact fraction stays flat around 0.30-0.35 across thresholds 2, 10, 20, and ssim gets steadily worse, not better, for example shoLANT ssim goes 0.240 at threshold 2, to 0.215 at 10, to 0.158 at 20. Whatever is wrong with shoulder is not fixed by a cleaner mask, this closes off the idea from the section above as tried and not working.

Chest and the kneeLANT control are stable and already good from threshold 2 through 20, no real change in near-exact fraction or ssim across that range, so there was nothing to gain there either.

A second, more serious problem showed up at the higher thresholds, 30 and 50. background_fraction_mean climbs steeply, and for chest at 50, for kneeLANT at 30 and 50, and for all of shoulder at 30 and 50, match_score_mean shows up as `-inf`. This is not a display glitch, it means at least one of the 20 crops in that group has no pixels at all above that threshold, an entirely empty mask, and `matchTemplate` has nothing to search with. `locate`'s own `-inf` sanitizing correctly stops that sample from reporting a fake perfect score, but it also means the aggregate mean for that group is not meaningful once even one sample fails this way. kneeLANT, a region that matches perfectly at threshold 2, starts failing outright at threshold 30, a fixed absolute threshold is not safe to raise blindly across regions or samples, some crops are simply dim overall.

A fixed absolute threshold, in short, is answered now, raising it does not help the one region it was meant to help, and risks breaking regions that did not need help at all. A threshold relative to each crop's own pixel distribution, for example its own brightest 20%, would not have this empty-mask failure mode the same way, not tried yet.

## Percentile mask sweep

Tried it: `percentile_mask` in `src/matching/core.py` keeps a crop's own brightest fraction, for example above its 90th percentile, rather than one fixed pixel value for every crop. `src/matching/percentile_sweep.py` reruns the same 20 ids and same component set at percentiles 50, 70, 80, 90, 95, results in `result/tables/percentile_sweep.xlsx`.

The empty-mask failure from the fixed threshold sweep is gone, no `-inf` anywhere in this run, a percentile of a crop's own pixels cannot be empty for that crop the way a fixed absolute cutoff can.

It still does not help shoulder. Not flat this time, actually worse the more selective the mask gets: shoLANT ssim goes 0.227 at percentile 50, down to 0.221, 0.197, 0.164, landing at 0.170 at percentile 95, and peak_margin drops the same way, 0.059 down to roughly 0.015-0.027 at the higher percentiles. Excluding more of the crop, keeping only its very brightest pixels, makes the match less confident and less accurate for shoulder, the opposite of the hoped for effect.

Chest and the kneeLANT control stay exactly as good across the whole percentile range, same near-exact fraction and ssim at every percentile, confirming both were already solved and are not sensitive to how the mask is built. Vertebra moves the other way from shoulder, its peak_margin rises with a more selective mask, 0.444 at percentile 50 up to 0.645 at percentile 90, consistent with vertebra actually having plenty of strong signal that benefits from excluding more background, unlike shoulder.

Masking, in every form tried so far, fixed threshold low, fixed threshold high, and now relative percentile at several levels, does not fix shoulder, and excluding more of it only makes shoulder's own match worse. Whatever the shoulder problem actually is, it is very unlikely to be an amount-of-background problem at this point.

## Correction, the evaluation itself was penalizing a padded region, not just the search

A different idea, not about the search at all: several crops carry a black padded region the source pipeline added, not real anatomy (the diagonal wedge already seen in chest and shoulder samples, and vertebra's own non-rectangular silhouette). Comparing the full crop against the full extracted window, as `compare` always did before, scores that padding against real whole body content there and penalizes an otherwise correct match for content the crop was never meant to carry. `compare` in `src/eval/crop_match_metrics.py` now takes an optional mask and restricts every metric to the masked-in pixels when given one, self test extended to cover it, `src/matching/masked_template_match.py` now reports both the original full-array metrics and this masked-evaluation version side by side.

Real numbers change the picture substantially:

- vertebra: near-exact fraction and ssim both jump to essentially 1.0 once evaluated with the mask. Checked directly whether the masked search had even moved the found location versus the plain unmasked baseline, it had not, same (x, y) in 100% of the 20 samples for both vertebra folders. Vertebra's search was correct from the very first baseline, the "shape mismatch" explanation earlier in this file was wrong, what looked like an imperfect match was entirely this project's own full-array evaluation penalizing the crop's own non-rectangular padding, not a real limitation of the match itself
- chest: near-exact fraction and ssim both also jump to essentially 1.0 under masked evaluation, not just the roughly 0.75-0.79 reported in "Masked template matching" above, that number was still measured with the old full-array comparison. Checked the same way, masking did change the found (x, y) location for chest, only 10-35% of the 20 samples land on the same spot as the unmasked baseline per folder, so unlike vertebra, chest genuinely needed the masked search, not only the masked evaluation, both mattered
- shoulder: does improve, ssim roughly doubles, for example shoLANT 0.240 to 0.553, but stays far below every other region, nowhere near the near 1.0 seen everywhere else once fairly evaluated. Masking also changes shoulder's found location almost every time, 0-10% of samples match the unmasked baseline's spot. So shoulder is not explained by this evaluation problem either, fair search and fair evaluation together still do not produce a good match

This resolves ankle, elbow, head, knee, pelvis, chest, and vertebra, 24 of 26 folders, all effectively solved once search is masked where needed and evaluation is masked always. Shoulder alone remains a real, unexplained problem, not an artifact of how this project was scoring it.

## Checking the masked evaluation is not just rewarding a mostly blank comparison

A fair worry about the correction above: with a third or so of a crop's pixels excluded as background on both sides, could a wrong window still score near 1.0 simply because the excluded portion trivially agrees, real signal quality aside. Checked directly rather than argued about.

`ssim` under `compare`'s mask sets the excluded pixels to 0 in both images before scoring, which does give that portion free credit, `near_exact_fraction`, `mae`, `rmse`, `pearson_corr`, and `hist_intersection` instead drop the excluded pixels entirely rather than zero them, so those five have no such free credit built in.

Tested against a vertebra crop (background fraction 28%) and a chest crop (also 28%), each scored with the same mask against three windows, the actual correct match, an all black fake window, the worst case for this concern since it maximizes how much of the comparison could be trivial, and a window shifted by a plausible 10 pixels. The correct match scores near-exact fraction 1.0 and ssim above 0.996 in both cases. The all black fake window scores near-exact fraction 0.0, pearson correlation 0.0, and ssim 0.006 to 0.11, nowhere near 1.0 despite carrying the same background fraction as the real comparisons. The 10 pixel shift, a small and plausible near miss rather than an adversarial case, still drops ssim to 0.32. The masked evaluation discriminates a genuinely correct match from a wrong one clearly, the background exclusion is not doing the work the near-1.0 scores reported above depend on.

## Full dataset run

`src/matching/generate_bounding_boxes.py` ran the masked search plus masked evaluation approach across the complete dataset, all 26 folders, all 2925 shared ids, 76050 predictions, roughly 29 minutes. Output at `bs80k-bone_region-bb/bounding_boxes.csv`, the project's actual deliverable per intent.md, not an internal analysis table. Per-component means match the 20-id sample closely, 24 of 26 folders essentially exact, shoulder still clearly worse.

2 of the 76050 rows have a non finite match_score, ankleRANT id 2821 and ankleRPOST id 2577, both landing at the fallback (0, 0). Checked directly, both crops are almost pure background, max pixel value 5 and 6 respectively across the whole crop. With that few pixels passing the background mask, correlation becomes numerically degenerate across the entire search, not just at a few candidate spots, and `locate` has nothing finite to report. These 2 rows are not real position predictions and should not be trusted or used downstream as is. Also worth noting, `ssim` for these two rows reads roughly 0.98, an instance of the exact concern checked and dismissed above, an empty mask zeroes both images entirely and gives a trivial high score, that check found this does not happen for a normal, mostly-populated mask, but a genuinely empty one is a different case and this confirms it, isolated to 2 rows out of 76050.

## Positional outliers, checked across the full dataset

`src/matching/position_outliers.py` flags a prediction whose (x, y) sits far from the median (x, y) for that same component, a robust z-score on distance from the median (median and MAD, not mean and standard deviation, so the outliers themselves do not skew the baseline they are measured against), threshold 3.5, the standard cutoff from Iglewicz and Hoaglin. This checks position directly, not a quality metric, motivated by the concern that an odd position is itself a plausible sign of a wrong placement. Results in `result/tables/position_outliers.xlsx`, scatter plot per component in `result/figures/position_outliers.png`.

3232 of 76050 predictions flagged, 4.25% overall, but not evenly spread. Every non-shoulder component sits between 0.10% and 4.17% outliers, ankle, elbow, knee, pelvis all under 0.4%, chest around 1.7-1.8%, head and vertebra a bit higher, 2.4-4.2%. All 4 shoulder folders are in a different range entirely, 12.9% to 29.7% of shoulder predictions are positional outliers, ten times or more the rate anywhere else. The scatter plot shows why directly, every other component is one tight cluster of points with a handful of outliers at the edges, shoulder's own correct cluster is a small group off to one side while a large, separately scattered cloud of points spans most of the image.

Flagged outliers also have clearly worse match quality, not just an unusual position: mean near-exact fraction for flagged rows is 0.33 versus 0.88 across all rows. Position and quality agree with each other here, and this independently re-finds the 2 degenerate ankleR rows above, both sit at (0, 0), far from their component's own median position.

## Where shoulder outliers actually land

Looked at the shoulder outliers by eye. First pass looked only at the single most extreme outlier per shoulder folder by distance, 8 cases, all 8 landed at the feet. That is a selection artifact, not a finding, the feet are mechanically the farthest possible point from the true shoulder location near the top of the image, so the largest-distance cases are the least informative sample to look at.

A random, representative sample of shoulder outliers instead, then binning all 2526 shoulder outliers by y position to check the pattern held: 137 (5%) land close to the true shoulder position anyway (still flagged, but a smaller miss), 782 (31%) land around the pelvis, 995 (39%) around the knee, 612 (24%) around the feet. Wrong shoulder placements scatter broadly down most of the body's length, not toward one specific false attractor. This fits the earlier peak margin finding better than a specific confusable landmark would, a low information template correlating similarly poorly almost everywhere, rather than one particular wrong spot correlating suspiciously well.

## Centroid relative check

A different way to ask the same position question: instead of comparing a prediction's raw (x, y) to the median (x, y) for that component, `src/matching/centroid_outliers.py` first centers every patient's own predictions on that patient's own body centroid, the center of mass of the whole body image's own silhouette mask, `background_mask` applied to the whole body image itself rather than a crop. This does not assume every patient sits identically placed in the frame. Results in `result/tables/centroid_outliers.xlsx`, scatter plot in `result/figures/centroid_outliers.png`.

This also directly gives the typical offset from centroid asked for, per component, median offset in x and y, in `centroid_outliers.xlsx`'s own typical_offset_from_centroid sheet. The numbers read anatomically sensibly on their own: head sits about 279 pixels above centroid, vertebra about 135 above, chest about 127 above, shoulder about 185-193 above, elbow about 76-88 above, pelvis is close to centroid itself, within about 8-12 pixels, knee about 208 below, ankle about 396 below, the farthest of any region. Left and right mirror each other in x as expected.

Outlier rates by this method land close to the earlier position based check: every non-shoulder component between 0.5% and 3.6%, shoulder at 15.5% to 31.4%. Two different methods, raw position against a shared baseline and position against each patient's own body center, agree closely on both which region has a real problem and roughly how large it is. The scatter plot shows the same shape as before too, shoulder's wrong placements resolve into a few distinct clusters rather than one, consistent with landing near the pelvis, knee, or foot found by looking at cases directly above, not a uniform scatter.

## Output format

Checked how bounding box datasets are usually stored before settling this. The three common annotation formats are all built for a different problem than this one, a variable number of labeled objects per image: COCO is one JSON file with images, annotations, and categories sections, box as [x, y, width, height]. Pascal VOC is one XML file per image, box as [xmin, ymin, xmax, ymax]. YOLO is one text file per image, one line per object, box as normalized center x/y/width/height plus a class id.

This project's own case is simpler in shape and richer in another way, exactly one expected box per component per patient, not a variable count, and each box carries 6 quality metrics from `crop_match_metrics.py` plus the match score and peak margin from the search itself, not a single class label and confidence. That is closer to a flat table than to any of the three formats above, so `bounding_boxes.csv`, already in place at `bs80k-bone_region-bb`, one row per component per patient, x/y/width/height plus every metric as its own column, fits this project's own shape better than adopting one of those formats as is would. If a COCO style export is ever needed for compatibility with existing object detection tooling, it can be generated from this CSV later, the CSV stays the source of record.

## Rotation and scale invariant matching, tried on shoulder

Tried the one major remaining explanation from the open questions below, that shoulder's true content might be rotated or scaled relative to a plain axis aligned crop, something translation only `matchTemplate` cannot express regardless of masking. `src/matching/rotation_scale_match.py` tries the crop at 7 angles (-15 to 15 degrees) times 5 scales (0.9 to 1.1) around its own center, 35 variants, keeps whichever variant and location scores best, same 20 ids as every other shoulder experiment above, results in `result/tables/rotation_scale_match.xlsx`.

Raw match_score does go up meaningfully, for example shoLANT 0.588 to 0.684, but that is expected on its own and not by itself evidence of a real fix, taking the best of 35 tries will tend to find a higher score than taking the best of 1 even against an unrelated candidate, simply from trying more options. The fair, masked evaluation metrics are the ones that matter here, and they barely move: near-exact fraction goes from 0.230-0.245 to 0.214-0.264 across the 4 folders, actually down slightly for 2 of the 4, ssim goes from 0.553-0.590 to 0.563-0.582, again down slightly for 2 of the 4. Nothing close to the size of improvement masking gave chest.

The winning angle also does not point to a real rotation. 0 degrees is the single most common winning angle in 3 of the 4 folders, 26 of 80 samples overall picked 0, and the rest spread fairly evenly from -15 to 15 rather than clustering on one nonzero value. A genuine, consistent rotation mismatch would be expected to show up as most samples agreeing on roughly the same nonzero angle, this does not. Winning scale leans mildly upward, 1.05 or 1.10 most common, but combined with quality metrics barely moving, this reads as mild overfitting from searching more transform options rather than a real scale mismatch.

Rotation and scale invariant search, tried within these ranges, does not meaningfully fix shoulder either. This closes off the geometric mismatch explanation as tried, at least for simple rigid rotation and uniform scale in this range, not confirmed as the cause.

## Vertebra anchored search band

The head to pelvis band tried earlier used head's own matched bottom edge as the top boundary, and made shoulder worse, likely cutting off the true location. Checked the real relationship first this time rather than guessing at a boundary: among shoulder matches not already flagged as a positional outlier, shoulder's y sits a median 13 to 15 pixels below vertebra's own matched top edge, standard deviation 18 to 29 pixels, a much tighter relationship than the head based band assumed. Vertebra is one of the 24 already solved folders, so its own match is a reliable anchor to build from.

`src/matching/vertebra_anchored_shoulder.py` matches vertebra first, then restricts the shoulder search to a generous band, 100 pixels above vertebra's matched top edge to 250 pixels below it, wide enough to comfortably cover the empirical spread above with margin to spare, not a tight cut. Same 20 ids, results in `result/tables/vertebra_anchored_shoulder.xlsx`.

This is the first attempt where all 4 shoulder folders move the same direction rather than a mixed or negative result. Peak margin improves meaningfully and consistently, roughly 0.025 to 0.04 higher across all 4, a real gain in how confidently the search lands somewhere, not just a wider spread of options to pick from. Near-exact fraction and ssim improve too, modestly but consistently, near-exact fraction up 0.004 to 0.026, ssim up 0.002 to 0.026, only one of the 8 numbers moved the wrong way and only slightly. Nowhere near the roughly 1.0 seen for the 24 solved folders, this is a real, if partial, improvement, not a fix.

## Left-right confusion checked separately from far-away mismatches

A fair question: is part of the remaining shoulder problem a left/right mixup rather than a wrong region entirely. Checked on the full dataset, using vertebra's own matched x as a spine centerline and comparing each shoulder box's center against it. 5.3% to 14.8% of shoulder predictions land on the wrong side of the spine outright, worse in the posterior view, 14.1-14.8%, than anterior, 5.3-6.4%.

But cross tabulating wrong side against far from the typical vertical offset shows these are mostly the same failures, not two separate problems. A clean mirror swap, right height, wrong side only, is under 1.5% for every folder. Most of what looks like a left/right mixup is really a box that landed somewhere else entirely, near the pelvis, knee, or foot found earlier, which happens to also fall on the "wrong" side of the spine simply because it is nowhere near the shoulder at all. A dedicated left/right heuristic would not address much beyond what fixing the vertical mislocalization already addresses.

Checked the same breakdown on the vertebra anchored band's own output, same 20 ids, and both problems are nearly gone there, 0 or 1 of 20 per folder land on the wrong side, 0 or 1 of 20 land outside the expected vertical zone. The band resolves gross mislocalization, both the far-away misses and the left/right confusion that mostly rode along with them.

What is left after that is a different kind of problem. Even the shoulder boxes that land in exactly the right neighborhood, correct side, correct height, still only score around 0.24 near-exact fraction and 0.56 to 0.60 ssim under masked evaluation, nowhere near the roughly 1.0 seen once search location is right for every other folder. So the remaining issue is not really about where the search looks anymore, it is precision once it is looking in the right place. This connects back to the very first shoulder finding, the crop is mostly background with only about 2.4% of pixels clearly signal, too little distinctive content for the correlation peak to lock on precisely even inside the correct neighborhood, not just too little to find the neighborhood in the first place.

## Open questions

- Shoulder's actual problem is still unexplained after ruling out search ambiguity alone, the head-to-pelvis band, background masking at several thresholds and percentiles, the evaluation methodology, confirming the same picture two different ways at full dataset scale, and rigid rotation plus uniform scale. Anchoring the search on vertebra instead of head gave the first real, if partial, improvement, worth rerunning at full dataset scale and rechecking the positional outlier rate specifically, not done yet. The remaining candidates for what still limits it are a more flexible geometric transform matchTemplate genuinely cannot express even with rotation and scale added, for example a non-rigid warp, or that shoulder's true content is simply too close to the noise floor for pixel correlation of any kind to work reliably, back to the 2.4% clearly-signal-pixel finding from inspecting one shoulder case earlier
- Whether to detect and flag a fully empty mask explicitly rather than let it fall through to a meaningless (0, 0) prediction, only known to affect 2 of 76050 rows so far, not fixed yet
- How shoulder, chest, and pelvis box edges are actually set: ref 37 answers this in outline for shoulder/thorax and for pelvis, see the sections above, both use several reference points rather than a plain axis aligned box, the exact scan direction for the four shoulder/thorax boundary points is still ambiguous in ref 37's own text, and ref 37 does not confirm whether pelvis crops use I_org or I_fuzzy pixels in the final output, see "Original vs preprocessed pixels in the final crop" above
- Whether BS-80K's own pipeline follows ref 37's 46 region breakdown exactly or merges/drops sub-regions before producing its own 26 folders, not confirmed, see the mapping section above, "ankle" specifically has no named counterpart in ref 37
- Whether radius/ulna vs palm, and tibia/fibula vs foot, use a stated separating rule, ref 37 says these are found "in the same way" as elbow/knee but does not give a threshold for them specifically

## Status

Template matching plus masked evaluation now resolves 24 of 26 region folders essentially exactly, see "Correction, the evaluation itself was penalizing a padded region, not just the search" above, the current, corrected understanding. Ankle, elbow, head, knee, pelvis needed no masking at all. Vertebra needed masked evaluation only, its search was already exact from the first baseline. Chest needed both a masked search and masked evaluation. Shoulder is the one folder still unsolved, ruled out as a fix so far: search ambiguity alone, a head-to-pelvis search band, background masking at several fixed thresholds and percentiles, the evaluation methodology itself, and rigid rotation plus uniform scale, all either did not help or made it worse. Anchoring the shoulder search on vertebra's own matched top edge instead of head's, see "Vertebra anchored search band" above, gave the first real if partial improvement, all 4 shoulder folders moved the right direction. Checking gross mislocalization specifically, wrong side of the spine or landing far from the expected height, see "Left-right confusion checked separately" above, shows the vertebra band resolves nearly all of it, both were mostly the same underlying failure, not two problems. What remains even in correctly located cases is precision within the right neighborhood, near-exact fraction around 0.24 and ssim around 0.56-0.60, still far short of the roughly 1.0 seen on solved folders, tied to the crop's own low signal content rather than a location bug. Not yet rerun at full dataset scale.

Run on the full dataset, 76050 predictions, see "Full dataset run" above, output at `bs80k-bone_region-bb/bounding_boxes.csv`. Matches the sample closely. 2 rows are known failures, a genuinely empty background mask, see above. Positional outlier detection, both against a shared per-component baseline and against each patient's own body centroid, "Positional outliers" and "Centroid relative check" above, agree with each other and independently confirm the same picture at full scale, every other component has an outlier rate under about 4%, shoulder alone runs 13% to 31%. The centroid check also gives a clean, anatomically sensible typical offset from body center for every region, in `centroid_outliers.xlsx`. `src/eval/crop_match_metrics.py`, the match scoring module used throughout, is implemented and its own self test has been run against synthetic arrays, including the mask case, see the section above that.
