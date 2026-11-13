# Vinfotech CV Engineer — Case Study 2: Traffic Intersection Analytics

**Submission candidate: V2 (full video run)** · **Status: preliminary estimates, not independently verified ground truth**.

This project measures traffic in a fixed-camera four-arm intersection using pretrained YOLO11n, ByteTrack, OpenCV, and directional geometry. It exports an annotated video, vehicle-level tracking records, an entry-arm total, a 4×4 turning matrix, per-arm maximum observed queue lengths, and observed waiting-time statistics.

## Recording and results

- Original recording: `trafic.mp4`; 656.63 seconds, 30 FPS, 19,699 frames.
- Source clip and annotated output: obtain from the shared video links in `VIDEO_LINKS_TO_FILL.md` (not included in GitHub source ZIP).
- V2 provisional report: **204 entries**, **128 completed movements**, **76 unresolved**. These numbers have **not** been independently verified against all vehicles.
- Original video must cover at least three complete signal cycles; this has not yet been manually certified.
- Data outputs: `results/summary.json`, `results/vehicles.csv`, `results/turning_counts.csv`, `results/arm_summary.csv`.

## Installation (Windows 11, Python 3.12)

Open PowerShell in `Case_Study_2`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the original video to `videos/original/trafic.mp4`, then run:

```powershell
.\.venv\Scripts\python.exe .\src\traffic_case2_v2.py `
  --video .\videos\original\trafic.mp4 `
  --zones .\config\zones_reviewed_draft.json `
  --mode analyze --model yolo11n.pt --device cpu --imgsz 960 --stride 1 `
  --out .\outputs_reproduced
```

The model weights may download on the first run. To preview zones without inference, change `--mode analyze` to `--mode preview --preview-time 300`. The scripts operate on CPUs; processing time is longer than video playback. Output reproduction may vary between model and dependency versions unless they are pinned.

## Method, assumptions and limitations

- See `docs/APPROACH_200_WORDS.md` for the requested approach write-up.
- See `docs/ASSUMPTIONS.md` for explicit assumptions.
- Vehicles stopped at signals are estimated from image-plane movement; without real-world calibration, queues and waits are sensitive to camera perspective and ROI placement.
- Occlusions, distant cars, parked-vehicle contamination, and car/truck duplicate boxes can cause mistakes.
- Incomplete trajectories are deliberately reported as unresolved, not assigned guessed destinations.
- The 4×4 turning matrix is an automated estimate; do **not** present it as manually validated ground truth.

## Submission

Upload **this source package** as a ZIP/repository and supply a separate Google Drive folder containing the exact **same V2** original and annotated videos. Vinfotech's assignment requests **both Case Study 1 and Case Study 2 together in one repository and ZIP**. Case Study 1 is **not included** in this Case Study 2 package; add it separately if available.
