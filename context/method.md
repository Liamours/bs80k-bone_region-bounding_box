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

A higher peak margin here did not mean a better match. The most likely reading, not confirmed by inspecting an actual image yet, is that the band cuts off the genuinely correct location rather than just removing wrong competing ones, particularly for shoulder, where ref 37's own account has the shoulder apex located starting right at the neck row, close to or above where this band's top edge, the head match's own bottom edge, was placed. A band derived this way is not a working fix as tried. Loosening the top of the band, or deriving it from something other than the head crop's own matched bottom edge, would be the next thing to try before concluding constraining the search is the wrong approach entirely.

## Open questions

- Why the head-to-pelvis band made shoulder matches worse instead of better despite a higher peak margin, most likely the band excludes the true location rather than just excluding wrong ones, not confirmed by looking at an actual case yet
- What match score threshold separates a correct match from a wrong one: the baseline above gives real numbers, near-exact fraction and ssim near 1.0 for 18 of 26 folders, a real spread from 0.34 to 0.64 for the other 8, still provisional, not yet checked case by case against ground truth since none exists, only against the metrics themselves and now peak_margin
- What format to write the output box in, not decided yet
- How shoulder, chest, and pelvis box edges are actually set: ref 37 answers this in outline for shoulder/thorax and for pelvis, see the sections above, both use several reference points rather than a plain axis aligned box, the exact scan direction for the four shoulder/thorax boundary points is still ambiguous in ref 37's own text, and ref 37 does not confirm whether pelvis crops use I_org or I_fuzzy pixels in the final output, see "Original vs preprocessed pixels in the final crop" above
- Whether BS-80K's own pipeline follows ref 37's 46 region breakdown exactly or merges/drops sub-regions before producing its own 26 folders, not confirmed, see the mapping section above, "ankle" specifically has no named counterpart in ref 37
- Whether radius/ulna vs palm, and tibia/fibula vs foot, use a stated separating rule, ref 37 says these are found "in the same way" as elbow/knee but does not give a threshold for them specifically

## Status

Baseline template matching is implemented and has been run on a 20-id sample across all 26 region folders, see "Baseline results" above. It works essentially perfectly for 18 of 26 folders. Vertebra's lower score is explained, a shape mismatch with a clear peak, see "Second peak margin" above. Chest and shoulder's lower score is partly explained, a genuinely ambiguous search rather than only a shape mismatch, but a first fix attempt, constraining the search to a head-to-pelvis band, made shoulder worse rather than better, see "Constrained search attempt" above, so this is still open. Not yet run on the full dataset. `src/eval/crop_match_metrics.py`, the match scoring module used throughout, is implemented and its own self test has been run against synthetic arrays, see the section above that.
