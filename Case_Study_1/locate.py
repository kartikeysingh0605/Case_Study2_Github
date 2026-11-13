
import cv2
import numpy as np
from pathlib import Path
import argparse

# ==========================================
# STEP 1: Set project paths
# ==========================================

ROOT = Path(__file__).resolve().parent

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

REFERENCE_PATH = (
    ROOT / "reference_good" / "good_ref_01.png"
)

# ==========================================
# STEP 2: Define component regions
# ==========================================

# Format: (x, y, width, height)
# Coordinates based on good_ref_01.png
# Resolution: 1600 x 1200

REGIONS = {
    "Bolt TL": (130, 168, 108, 106),
    "Bolt TR": (1300, 165, 110, 105),
    "Bolt BL": (130, 941, 108, 108),
    "Bolt BR": (1300, 941, 110, 108),

    "Brass output gear": (237, 477, 455, 448),
    "Output nut": (408, 642, 104, 118),

    "Idler gear": (670, 571, 303, 302),
    "Idler circlip": (767, 645, 115, 113),

    "Compound gear": (741, 305, 397, 380),
    "Compound circlip": (880, 431, 105, 119),

    "Pinion gear": (1062, 339, 196, 195),
    "Cable connector": (1136, 283, 72, 79),
}


# ==========================================
# STEP 3: Draw expected component regions
# ==========================================

def draw_regions(image):

    result = image.copy()

    for number, (name, box) in enumerate(
        REGIONS.items(), start=1
    ):

        x, y, w, h = box

        # Yellow ROI rectangle
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        # Number with dark outline
        cv2.putText(
            result,
            str(number),
            (x + 5, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            4
        )

        cv2.putText(
            result,
            str(number),
            (x + 5, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

    return result


# ==========================================
# STEP 4: Extract brass gear color features
# ==========================================

def find_brass_color(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Approximate brass/gold HSV range
    lower_gold = np.array([12, 35, 80])
    upper_gold = np.array([32, 200, 255])

    color_mask = cv2.inRange(
        hsv,
        lower_gold,
        upper_gold
    )

    # Restrict detection to expected gear ROI
    x, y, w, h = REGIONS["Brass output gear"]

    region_mask = np.zeros_like(color_mask)

    region_mask[y:y+h, x:x+w] = (
        color_mask[y:y+h, x:x+w]
    )

    # Calculate proportion of brass-colored pixels
    gold_pixels = cv2.countNonZero(
        region_mask[y:y+h, x:x+w]
    )

    total_pixels = w * h
    gold_ratio = gold_pixels / total_pixels

    return region_mask, gold_ratio


# ==========================================
# STEP 5: Load images
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--image",
    default="test_01",
    help="Test image name, e.g. test_01"
)

args = parser.parse_args()

reference = cv2.imread(str(REFERENCE_PATH))

# Use output from Stage 3
test_path = OUTPUT / f"{args.image}_aligned.png"

test = cv2.imread(str(test_path))

if reference is None:
    raise FileNotFoundError(REFERENCE_PATH)

if test is None:
    raise FileNotFoundError(
        f"Aligned image not found: {test_path}. "
        "Run align.py first."
    )

if reference.shape[:2] != test.shape[:2]:
    raise ValueError(
        "Image dimensions differ. Check alignment."
    )

# ==========================================
# STEP 6: Mark component regions
# ==========================================

reference_marked = draw_regions(reference)
test_marked = draw_regions(test)

# ==========================================
# STEP 7: Analyze brass gear color
# ==========================================

reference_mask, reference_ratio = (
    find_brass_color(reference)
)

test_mask, test_ratio = (
    find_brass_color(test)
)

# ==========================================
# STEP 8: Save results
# ==========================================

cv2.imwrite(
    str(OUTPUT / "step4_reference_regions.png"),
    reference_marked
)

cv2.imwrite(
    str(OUTPUT / f"{args.image}_regions.png"),
    test_marked
)

cv2.imwrite(
    str(OUTPUT / f"{args.image}_brass_mask.png"),
    test_mask
)

# ==========================================
# STEP 9: Print report
# ==========================================

print("\n====================================")
print("COMPONENT LOCALIZATION")
print("====================================")

for number, name in enumerate(REGIONS, start=1):
    print(f"{number}. {name}: {REGIONS[name]}")

print("\nBrass gear color analysis:")

print(
    f"Reference gold ratio: {reference_ratio:.3f}"
)

print(
    f"Test gold ratio: {test_ratio:.3f}"
)

print("\nComponent regions marked.")
print("No OK / NOT OK decision has been made.")
print("Results saved to:", OUTPUT)
