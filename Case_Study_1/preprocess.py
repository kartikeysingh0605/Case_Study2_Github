
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# STEP 1: Project paths
# ==========================================

ROOT = Path(__file__).resolve().parent

IMAGE_PATH = ROOT / "reference_good" / "good_ref_01.png"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# STEP 2: Read original image
# ==========================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

print("Original image loaded successfully")
print("Image shape:", image.shape)

# ==========================================
# STEP 3: Convert to grayscale
# ==========================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==========================================
# STEP 4: Original Canny (for comparison)
# ==========================================

old_blur = cv2.GaussianBlur(gray, (5, 5), 0)
old_edges = cv2.Canny(old_blur, 50, 150)

# ==========================================
# STEP 5: Improve local contrast using CLAHE
# ==========================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced_gray = clahe.apply(gray)

# ==========================================
# STEP 6: Apply light Gaussian blur
# ==========================================

blur = cv2.GaussianBlur(
    enhanced_gray,
    (3, 3),
    0
)

# ==========================================
# STEP 7: Improved grayscale Canny
# ==========================================

gray_edges = cv2.Canny(
    blur,
    35,
    100
)

# ==========================================
# STEP 8: Color-aware edge detection
# ==========================================

# LAB separates lightness and color information.
# This helps detect boundaries that grayscale misses.

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

l_channel, a_channel, b_channel = cv2.split(lab)

color_edges = np.zeros_like(gray)

for channel_index, channel in enumerate(
    [l_channel, a_channel, b_channel]
):
    channel_blur = cv2.GaussianBlur(
        channel, (3, 3), 0
    )

    if channel_index == 0:
        edges = cv2.Canny(channel_blur, 35, 100)
    else:
        edges = cv2.Canny(channel_blur, 20, 60)

    color_edges = cv2.bitwise_or(
        color_edges, edges
    )

# ==========================================
# STEP 9: Combine grayscale and color edges
# ==========================================

final_edges = cv2.bitwise_or(
    gray_edges,
    color_edges
)

# ==========================================
# STEP 10: Save individual images
# ==========================================

cv2.imwrite(
    str(OUTPUT_DIR / "01_grayscale.png"), gray
)

cv2.imwrite(
    str(OUTPUT_DIR / "02_enhanced_gray.png"),
    enhanced_gray
)

cv2.imwrite(
    str(OUTPUT_DIR / "03_old_canny.png"),
    old_edges
)

cv2.imwrite(
    str(OUTPUT_DIR / "04_improved_canny.png"),
    gray_edges
)

cv2.imwrite(
    str(OUTPUT_DIR / "05_color_edges.png"),
    color_edges
)

cv2.imwrite(
    str(OUTPUT_DIR / "06_final_edges.png"),
    final_edges
)

# ==========================================
# STEP 11: Display comparison in one image
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

axes[0, 0].imshow(rgb)
axes[0, 0].set_title("1. Original Image")

axes[0, 1].imshow(old_edges, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("2. Original Canny (50, 150)")

axes[1, 0].imshow(gray_edges, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title("3. Improved Grayscale Canny")

axes[1, 1].imshow(final_edges, cmap="gray", vmin=0, vmax=255)
axes[1, 1].set_title("4. Combined Color-Aware Edges")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

comparison_path = OUTPUT_DIR / "edge_comparison.png"

plt.savefig(
    str(comparison_path),
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("\n==================================")
print("IMAGE PREPROCESSING COMPLETED")
print("==================================")

print("Original Canny pixels:", np.count_nonzero(old_edges))
print("Improved Canny pixels:", np.count_nonzero(gray_edges))
print("Combined edge pixels:", np.count_nonzero(final_edges))

print("\nResults saved to:", OUTPUT_DIR)
print("Comparison:", comparison_path)
