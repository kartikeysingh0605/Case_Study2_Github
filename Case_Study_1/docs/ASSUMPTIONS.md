# Assumptions — Case Study 1

1. The camera is fixed and approximately top-down; plate translation and small rotations are handled by locating the blue base plate and perspective-normalizing it.
2. The reference images show the expected assembly layout. Inspection regions are anchored to this normalized geometry; different assembly layouts need recalibration.
3. The base plate remains sufficiently visible and blue for its HSV-color segmentation to locate it.
4. Lighting and component appearance vary within the range tolerated by the feature thresholds; severe shadows, reflections or occlusion can cause uncertainty or false decisions.
5. The visible-proxy policy considers the *hidden motor body* conditionally present **only when the specified visible motor proxies pass**; this is an assumption, not visual observation. The default strict policy does not waive it.
6. `OK` means the implemented checks pass under the selected policy; it is not a certified guarantee of mechanical correctness.
7. Included test images have no independently verified fault labels. The saved verdicts are the pipeline's predictions.
