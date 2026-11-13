
import cv2
import numpy as np
from pathlib import Path
import argparse

# ==========================================
# STEP 1: Project paths
# ==========================================

ROOT = Path(__file__).resolve().parent

REFERENCE_PATH = ROOT / "reference_good" / "good_ref_01.png"

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Select test image from command line
parser = argparse.ArgumentParser()

parser.add_argument(
    "--image",
    default="test_images/test_01.png",
    help="Path to test image"
)

args = parser.parse_args()

TEST_PATH = Path(args.image)

if not TEST_PATH.is_absolute():
    TEST_PATH = ROOT / TEST_PATH


# ==========================================
# STEP 2: Load images
# ==========================================

reference = cv2.imread(str(REFERENCE_PATH))
test = cv2.imread(str(TEST_PATH))

if reference is None:
    raise FileNotFoundError(
        f"Reference image not found: {REFERENCE_PATH}"
    )

if test is None:
    raise FileNotFoundError(
        f"Test image not found: {TEST_PATH}"
    )

print("Reference image loaded:", reference.shape)
print("Test image loaded:", test.shape)


# ==========================================
# STEP 3: Arrange four rectangle corners
# ==========================================

def order_corners(points):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    sums = points.sum(axis=1)

    differences = np.diff(
        points,
        axis=1
    ).ravel()

    top_left = points[np.argmin(sums)]
    top_right = points[np.argmin(differences)]

    bottom_right = points[np.argmax(sums)]
    bottom_left = points[np.argmax(differences)]

    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ],
        dtype=np.float32
    )


# ==========================================
# STEP 4: Detect blue base plate
# ==========================================

def detect_base_plate(image):

    # Convert BGR image into HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Blue plate HSV range
    lower_blue = np.array([92, 34, 45])
    upper_blue = np.array([118, 185, 245])

    # Create blue color mask
    mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # Close gaps in the mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (45, 45)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError(
            "Blue base plate could not be detected"
        )

    # Select largest blue region
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    contour_area = cv2.contourArea(
        largest_contour
    )

    image_area = image.shape[0] * image.shape[1]

    if contour_area < 0.20 * image_area:
        raise RuntimeError(
            "Detected base plate is too small"
        )

    # Find minimum area rotated rectangle
    rectangle = cv2.minAreaRect(
        largest_contour
    )

    # Obtain the four corners
    corners = cv2.boxPoints(
        rectangle
    )

    corners = order_corners(
        corners
    )

    return mask, corners


# ==========================================
# STEP 5: Detect plates in both images
# ==========================================

ref_mask, ref_corners = detect_base_plate(
    reference
)

test_mask, test_corners = detect_base_plate(
    test
)

print("\nReference plate corners:")
print(ref_corners)

print("\nTest plate corners:")
print(test_corners)


# ==========================================
# STEP 6: Calculate alignment
# ==========================================

# Match the first three plate corners
# between test image and reference.

transform_matrix = cv2.getAffineTransform(
    test_corners[:3],
    ref_corners[:3]
)

print("\nAffine transformation matrix:")
print(transform_matrix)


# ==========================================
# STEP 7: Apply transformation
# ==========================================

height, width = reference.shape[:2]

aligned_test = cv2.warpAffine(
    test,
    transform_matrix,
    (width, height),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(160, 160, 160)
)

print("\nImage alignment completed")


# ==========================================
# STEP 8: Create comparison overlay
# ==========================================

overlay = cv2.addWeighted(
    reference,
    0.5,
    aligned_test,
    0.5,
    0
)


# ==========================================
# STEP 9: Visualize detected plate corners
# ==========================================

detected_plate = test.copy()

cv2.polylines(
    detected_plate,
    [test_corners.astype(np.int32)],
    isClosed=True,
    color=(0, 255, 0),
    thickness=4
)

for i, point in enumerate(test_corners):

    x, y = point.astype(int)

    cv2.circle(
        detected_plate,
        (x, y),
        8,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        detected_plate,
        str(i + 1),
        (x + 12, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


# ==========================================
# STEP 10: Save results
# ==========================================

image_name = TEST_PATH.stem

cv2.imwrite(
    str(OUTPUT_DIR / f"{image_name}_plate_mask.png"),
    test_mask
)

cv2.imwrite(
    str(OUTPUT_DIR / f"{image_name}_detected_plate.png"),
    detected_plate
)

cv2.imwrite(
    str(OUTPUT_DIR / f"{image_name}_aligned.png"),
    aligned_test
)

cv2.imwrite(
    str(OUTPUT_DIR / f"{image_name}_overlay.png"),
    overlay
)


# ==========================================
# STEP 11: Final result
# ==========================================

print("\n======================================")
print("IMAGE ALIGNMENT COMPLETED SUCCESSFULLY")
print("======================================")

print("Reference:", REFERENCE_PATH.name)
print("Test image:", TEST_PATH.name)

print("\nSaved files:")

print(f"1. {image_name}_plate_mask.png")
print(f"2. {image_name}_detected_plate.png")
print(f"3. {image_name}_aligned.png")
print(f"4. {image_name}_overlay.png")

print("\nOutput folder:", OUTPUT_DIR)
