# Known limitations and next improvements

- A fixed, color-dependent plate detector may fail when the base plate is occluded, another color is used, or lighting changes substantially.
- Region-specific feature thresholds can misclassify small reflective fasteners, washers, circlips and connector details.
- The motor body is not visible; `visible-proxy` verdicts that are OK are explicitly conditional.
- The six test images have no independently supplied ground-truth fault labels; output agreement with the pipeline alone is not an accuracy score.
- Current scripts do not prove detection of every possible missing or misplaced component on unseen images.
- Next steps: independently label missing/misplaced components, test variations in lighting/rotation, measure per-component false-negative rates, and calibrate thresholds on additional real assemblies.
