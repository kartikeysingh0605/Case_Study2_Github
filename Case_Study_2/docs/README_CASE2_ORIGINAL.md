# Case Study 2 — Traffic Intersection Analytics

## Overview

Python + OpenCV + YOLO11n + ByteTrack pipeline for a fixed daytime four-arm signalized intersection. It assigns tracked vehicles to entry and exit arms, produces a 4x4 turning matrix, estimates queue maxima and observed queue-to-stop-line waiting times, and exports an annotated video and CSV/JSON results.

**Status:** Functional, fully processed recording; traffic counts and waiting-time estimates remain provisional pending manual verification. Do not present these as ground-truth measurements.

## Data and dependencies

- Input: `trafic.mp4` (recorded source clip, 10:56.63, 30 FPS, 19,699 frames).
- Hardware: Windows 11 laptop; CPU supported; no GPU required.
- Python 3.12; dependencies in `requirements.txt`.
- Model weights `yolo11n.pt` are downloaded by Ultralytics when first run, or can be supplied locally.

## Setup

```powershell
cd "C:\Users\HP\Documents\Trafic\traffic_case_study2_starter"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Place the original video at `..\trafic.mp4`, with the project's reviewed geometry in `accuracy_v2\config\zones_submission.json`.

```powershell
.\.venv\Scripts\python.exe .\accuracy_v2\traffic_case2_v2.py `
  --video "..\trafic.mp4" `
  --zones ".\accuracy_v2\config\zones_submission.json" `
  --mode analyze --model yolo11n.pt --device cpu --imgsz 960 --stride 1 `
  --out ".\output_v2_review"
```

`--stride 1` processes all 19,699 frames at the source 30 FPS. CPU execution may take longer than the 10:56 clip. For a geometry check without running inference, replace `--mode analyze` with `--mode preview --preview-time 300`.

## Outputs

- `output_v2_review/annotated.mp4` — annotated tracks and arm zones.
- `output_v2_review/vehicles.csv` — individual track records and available events.
- `output_v2_review/turning_counts.csv` — origin-to-destination counts.
- `output_v2_review/arm_summary.csv` — per-arm entry, maximum queue and waiting-time summaries.
- `output_v2_review/summary.json` — full statistics and diagnostic flags.

As of V2, the pipeline reported 204 entries, 128 completed turning movements, 76 unresolved entries, and 28 observed waits. These numbers are *estimates* and are not independently verified.

## Explicit assumptions

See `ASSUMPTIONS.md`.

## Known limitations

- YOLO can miss distant, partly occluded or tiny vehicles; ByteTrack IDs may fragment or swap.
- Overlapping car/truck detections can refer to one vehicle (confirmed in a 40-second diagnostic sample); duplicate entries may occur.
- Directional stop lines and lane polygons are manually estimated; these materially affect movement, queue and wait calculations.
- Queue detection uses image-plane displacement, not metric-speed calibration; parked objects or traffic in adjacent lanes may contaminate results.
- Waiting-time averages use only completed observed waits, not every entering vehicle.
- All 19,699 frames are processed, but signal-cycle count and numerical accuracy still require manual video review.

## With more time

Manually label three full cycles; quantify precision/recall for entries and turning movements; tune detection/tracking with representative difficult cases; use validated lane-specific queue polygons and directional crosswalk lines; consider camera homography for metric speeds.

## Submission notes

Include the recorded original clip and full annotated video using an accessible link if too large for the code ZIP/Git repository. Submit both case studies together as specified by the assignment, if available.
