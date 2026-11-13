# Case Study 2 — Explicit assumptions

1. **Orientation:** The top of the camera frame is North, followed by South, East, and West based on image geometry, not real-world compass bearings.
2. **Stationary camera:** The camera position and frame perspective remain fixed throughout the recording.
3. **Counting classes:** COCO classes car, motorcycle, bus, and truck are eligible. Pedestrians, cyclists and unrecognized vehicles are outside the current detector classes.
4. **Entry counting:** A tracked vehicle is counted once when it is assigned an entry arm and reaches the junction according to the configured polygons.
5. **Turning movements:** A movement is completed only after an entry and an exit arm are confirmed. Tracks with uncertain exits remain unresolved; same-arm exits are U-turn candidates requiring review.
6. **Queue region:** Vehicles count as queued only if tracked within a configured incoming-lane polygon and meeting the algorithm's image-displacement stop condition.
7. **Queue maxima:** The observed maximum is the largest simultaneous number of stationary tracked vehicles in a queue region, including those still waiting when the clip ends.
8. **Waiting time:** A measured wait starts at the first detected stationary queue event and ends when the tracked vehicle crosses that arm's directional stop line. Missing or incomplete measurements are excluded from the average, not treated as zero.
9. **Stopped threshold:** Stopping is inferred from bounding-box ground-point displacement in image pixels over time; this is perspective-sensitive and is not a calibrated physical distance or speed measurement.
10. **Parked vehicles:** The queue ROIs are intended to exclude parking areas; persistent parked-looking objects and vehicles that never complete junction passage are excluded from confirmed traffic movements. Queue contamination remains a known risk.
11. **Partial tracks:** Tracks cut off at the beginning/end of the recording and occluded tracks may not have complete origin/destination data. A track ID is not guaranteed to correspond to a unique physical vehicle.
12. **FPS and evidence:** The original `trafic.mp4` is 19,699 frames at 30 FPS (656.63 seconds). All frames were processed with stride 1. At least three full signal cycles must be verified manually; they have not yet been confirmed from the full clip.
13. **Validation status:** Automated output-file integrity checks pass, but the reported 204 entries, 128 completed movements, and queue/wait values have not been validated by a complete human count.
14. **Speed bonus:** Vehicle speed in km/h is not reported because an independent pixel-to-ground-distance calibration has not been performed.
