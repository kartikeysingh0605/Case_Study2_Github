# Case Study 1 — Gear Assembly Verification

Vinfotech Computer Vision Engineer take-home assignment: CPU-based inspection of a fixed-camera gear assembly against three reference-good images.

## What is included

- `inspect.py` — single-image and batch CLI; annotated image, JSON evidence and CSV summary
- `align.py`, `locate.py`, `preprocess.py`, `client_report.py` — additional project utilities/report formatting
- `reference_good/` — three good-reference images supplied with the project
- `test_images/` — six original unlabelled test images
- `output/final_results/` — six saved annotated test images, normalized previews, inspection JSON, and `inspection_summary.csv`
- `docs/` — approach, assumptions, and known limitations

## Installation (Windows, Python 3.11–3.13, CPU only)

```powershell
cd Case_Study_1
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run on one image

```powershell
.\.venv\Scripts\python.exe inspect.py --image test_images/test_01.png --policy visible-proxy --output output/live_check
```

## Run on all six images

```powershell
.\.venv\Scripts\python.exe inspect.py --folder test_images --policy visible-proxy --output output/new_batch
```

The inspection script prints a verdict for each image, identifies suspected parts, saves annotated images and per-image JSON, and writes `inspection_summary.csv` for batch runs. Its default `strict` policy is *fail closed* if the hidden motor body cannot be seen. The demonstrated `visible-proxy` policy is explicitly conditional on visible motor-related features; an `OK` under this policy is **not direct proof** that the hidden motor body is present.

## Provided results

The included `output/final_results/inspection_summary.csv` records results of the supplied six-image batch using `visible-proxy`:

| Image | Recorded verdict |
| --- | --- |
| test_01.png | NOT OK |
| test_02.png | OK (conditional) |
| test_03.png | OK (conditional) |
| test_04.png | NOT OK |
| test_05.png | OK (conditional) |
| test_06.png | NOT OK |

These are pipeline outputs, **not independently verified fault labels**. Stored output CSV/JSON may include paths from the original development computer; regenerate outputs via the commands above for your own paths.

## Limitations

See `docs/KNOWN_LIMITATIONS.md`. In particular, the geometry and colors are specific to the provided blue-plate configuration and visible components. The motor body is occluded; the program documents this as an assumption rather than claiming to see it. The code and included results have been packaged and syntax-checked; the full six-image runtime has not been re-executed here.
