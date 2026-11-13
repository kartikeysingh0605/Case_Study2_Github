# Vinfotech Task 1 — Stage 6: existing detectors + image/folder reporting
# Component feature checks are experimental; fail-closed decision is explicit.
# This file is intentionally named inspect.py (as required by the assignment).

# Prevent this project file from shadowing Python's standard-library inspect module.
import os
import sys

PROJECT_DIR = os.path.normcase(os.path.abspath(os.path.dirname(__file__)))
sys.path[:] = [
    entry for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or os.getcwd())) != PROJECT_DIR
]

import argparse
import json
import csv
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
PLATE_SIZE = (1132, 790)  # width, height in pixels

# Coordinates below refer to the NORMALIZED blue plate.
GEARS = {
    'Brass output gear': {
        'center': (302, 484), 'inner': 65, 'outer': 180,
        'lower': (12, 35, 70), 'upper': (34, 220, 255),
    },
    'Idler gear': {
        'center': (605, 484), 'inner': 52, 'outer': 118,
        'lower': (0, 0, 65), 'upper': (179, 55, 255),
    },
    'Compound gear': {
        'center': (713, 302), 'inner': 63, 'outer': 143,
        'lower': (0, 0, 65), 'upper': (179, 55, 255),
    },
    'Motor pinion gear': {
        'center': (916, 247), 'inner': 22, 'outer': 72,
        'lower': (0, 0, 45), 'upper': (179, 55, 112),
    },
}

BOLTS = {
    'Top-left': (53, 53),
    'Top-right': (1079, 53),
    'Bottom-left': (53, 737),
    'Bottom-right': (1079, 737),
}


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'Cannot load image: {path}')
    return image


def order_corners(points):
    """Return top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    return np.float32([
        points[np.argmin(sums)],
        points[np.argmin(differences)],
        points[np.argmax(sums)],
        points[np.argmax(differences)],
    ])


def normalize_plate(image):
    """Find blue base plate and warp it to a consistent rectangular view."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 30, 40), (125, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError('Blue base plate was not detected')
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.20 * image.shape[0] * image.shape[1]:
        raise RuntimeError('Blue base plate detection is unreliable')

    corners = order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))
    width, height = PLATE_SIZE
    destination = np.float32([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ])
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, PLATE_SIZE)


def ring_mask(shape, center, inner_radius, outer_radius):
    region = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(region, center, outer_radius, 255, -1)
    if inner_radius > 0:
        cv2.circle(region, center, inner_radius, 0, -1)
    return region


def region_fraction(binary_mask, region_mask):
    denominator = cv2.countNonZero(region_mask)
    if denominator == 0:
        raise ValueError('Empty inspection region')
    numerator = cv2.countNonZero(cv2.bitwise_and(binary_mask, region_mask))
    return numerator / denominator


def gear_score(image, spec):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    match = cv2.inRange(hsv, spec['lower'], spec['upper'])
    region = ring_mask(image.shape, spec['center'], spec['inner'], spec['outer'])
    return region_fraction(match, region)


def metal_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 50), (179, 45, 255))


def inspect_gears(test, references, annotation):
    results = []
    for number, (name, spec) in enumerate(GEARS.items(), start=1):
        reference_scores = [gear_score(ref, spec) for ref in references]
        score = gear_score(test, spec)
        threshold = 0.75 * min(reference_scores)  # Experimental only
        is_present = score >= threshold
        status = 'POSSIBLY PRESENT' if is_present else 'SUSPECT MISSING'
        color = (0, 200, 0) if is_present else (0, 0, 255)

        cv2.circle(annotation, spec['center'], spec['outer'], color, 3)
        cv2.circle(annotation, spec['center'], spec['inner'], color, 2)
        label = f'{number}. {name}: {"PRESENT?" if is_present else "SUSPECT"} {score:.3f}'
        cv2.putText(annotation, label, (145, 28 + 28 * (number - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)

        results.append({
            'component': name,
            'reference_scores': [round(v, 4) for v in reference_scores],
            'test_score': round(score, 4),
            'threshold': round(threshold, 4),
            'status': status,
        })
    return results


def bolt_head_edge_count(edge_image, center, radius=18):
    """Count short internal edges within a bolt head (not its outer washer).

    A plain filled disk can have the same color as a valid bolt, but remove
    the internal bevel/face edges. This is a *proxy* for geometry, not a
    complete hexagon recognizer.
    """
    region = ring_mask(edge_image.shape, center, 0, radius)
    return cv2.countNonZero(cv2.bitwise_and(edge_image, region))


def bolt_head_contrast(image, center):
    """Core minus outer washer brightness; insensitive to global lighting shifts.

    Correct bolt heads are darker than the surrounding washer in these
    supplied reference images. This is not a universal bolt detector.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    core = ring_mask(image.shape, center, 0, 13) > 0
    washer = ring_mask(image.shape, center, 24, 33) > 0
    return float(gray[core].mean() - gray[washer].mean())


def inspect_bolts(test, references, annotation):
    """Check color coverage and (where supported) internal bolt geometry.

    Important: not all bolts have enough visible internal detail in these
    references. For those, preserve the color-only result and record that
    geometry could NOT be verified. Do not fabricate a strong geometry pass.
    """
    results = []
    test_metal = metal_mask(test)
    reference_metals = [metal_mask(ref) for ref in references]

    # Small, unblurred edges retain the faceted bevel on the top-left and
    # bottom-left bolts. Weak corners may not have usable edges.
    test_edges = cv2.Canny(cv2.cvtColor(test, cv2.COLOR_BGR2GRAY), 20, 55)
    reference_edges = [
        cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 20, 55)
        for ref in references
    ]

    for position, center in BOLTS.items():
        checks = {
            'Bolt head': (0, 18),
            'Washer': (28, 36),
        }
        all_present = True
        for part, (inner, outer) in checks.items():
            region = ring_mask(test.shape, center, inner, outer)
            ref_scores = [region_fraction(mask, region) for mask in reference_metals]
            score = region_fraction(test_metal, region)
            threshold = 0.75 * min(ref_scores)  # Experimental only
            color_present = score >= threshold

            edge_details = None
            shape_missing = False
            if part == 'Bolt head':
                ref_edge_counts = [
                    bolt_head_edge_count(ed, center) for ed in reference_edges
                ]
                test_edge_count = bolt_head_edge_count(test_edges, center)
                # Require consistent positive evidence from ALL good refs.
                # Currently this enables structure checks for only some bolts.
                shape_supported = min(ref_edge_counts) >= 12
                edge_threshold = 0.65 * min(ref_edge_counts) if shape_supported else None
                shape_missing = shape_supported and test_edge_count < edge_threshold

                # Independent brightness geometry cue at all four corners.
                # Uniform gray replacement erases the normal dark-core /
                # brighter-washer contrast even if metal saturation is high.
                ref_contrasts = [bolt_head_contrast(ref, center) for ref in references]
                test_contrast = bolt_head_contrast(test, center)
                contrast_limit = max(ref_contrasts) + 8.0  # experimental margin
                contrast_anomaly = test_contrast > contrast_limit
                shape_missing = shape_missing or contrast_anomaly
                edge_details = {
                    'brightness_method': 'mean(core r<=13) - mean(washer 24<=r<=33)',
                    'reference_core_contrasts': [round(q, 3) for q in ref_contrasts],
                    'test_core_contrast': round(test_contrast, 3),
                    'core_contrast_limit': round(contrast_limit, 3),
                    'contrast_anomaly': contrast_anomaly,
                    'edge_method': 'Canny edge count in central 18px disk',
                    'reference_edge_counts': ref_edge_counts,
                    'test_edge_count': test_edge_count,
                    'edge_threshold': round(edge_threshold, 3) if edge_threshold else None,
                    'shape_evidence': 'ENABLED' if shape_supported else 'NOT RELIABLE IN REFERENCES',
                }

            present = color_present and not shape_missing
            all_present = all_present and present
            if shape_missing:
                status = 'SUSPECT WRONG SHAPE / MISSING'
            elif not color_present:
                status = 'SUSPECT MISSING'
            else:
                status = 'POSSIBLY PRESENT'

            item = {
                'component': f'{position} {part}',
                'reference_scores': [round(v, 4) for v in ref_scores],
                'test_score': round(score, 4),
                'threshold': round(threshold, 4),
                'status': status,
            }
            if edge_details:
                item['bolt_shape_check'] = edge_details
            results.append(item)

        color = (0, 200, 0) if all_present else (0, 0, 255)
        cv2.circle(annotation, center, 37, color, 3)
        short_name = {
            'Top-left': 'TL', 'Top-right': 'TR',
            'Bottom-left': 'BL', 'Bottom-right': 'BR'
        }[position]
        label = f'{short_name}: {"OK?" if all_present else "CHECK"}'
        x, y = center
        text_x = x + 45 if x < test.shape[1] // 2 else x - 165
        text_y = y + 105 if y < test.shape[0] // 2 else y - 55
        cv2.putText(annotation, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return results


# ==============================================
# STAGE 5D: OUTPUT NUT AND TWO CIRCLIPS
# ==============================================
# These are centers in the NORMALIZED 1132 x 790 image.
# We inspect annular regions so the result remains relatively
# stable when a gear rotates around its own axis.

SMALL_PARTS = {
    "Output gear nut": {
        "center": (302, 484),
        "inner": 10,
        "outer": 25,
        "lower": (12, 35, 70),    # Gold/yellow metal
        "upper": (34, 220, 255),
        "threshold_factor": 0.65,
    },
    "Idler circlip": {
        "center": (605, 484),
        "inner": 25,
        "outer": 35,
        "lower": (0, 25, 0),     # Dark colored C-shaped ring
        "upper": (179, 255, 134),
        "threshold_factor": 0.65,
    },
    "Compound circlip": {
        "center": (713, 302),
        "inner": 25,
        "outer": 35,
        "lower": (0, 25, 0),
        "upper": (179, 255, 134),
        "threshold_factor": 0.65,
    },
}


def small_part_score(image, spec):
    """Measure a color/brightness feature inside a circular annulus."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    matching = cv2.inRange(hsv, spec["lower"], spec["upper"])
    region = ring_mask(image.shape, spec["center"],
                       spec["inner"], spec["outer"])
    return region_fraction(matching, region)


def inspect_small_parts(test, references, annotation, gear_results):
    """Experimental checks; do not equate detection with final OK/NOT OK."""
    results = []
    idler_suspect = any(
        r["component"] == "Idler gear" and r["status"] == "SUSPECT MISSING"
        for r in gear_results
    )
    for name, spec in SMALL_PARTS.items():
        ref_scores = [small_part_score(ref, spec) for ref in references]
        score = small_part_score(test, spec)
        threshold = spec["threshold_factor"] * min(ref_scores)

        # A circlip can't be independently inspected when its entire
        # parent gear is absent. Report the dependency explicitly.
        if name == "Idler circlip" and idler_suspect:
            status = "NOT ASSESSABLE (idler gear suspected missing)"
            color = (0, 165, 255)  # orange: dependent inspection
        else:
            present = score >= threshold
            status = "POSSIBLY PRESENT" if present else "SUSPECT MISSING"
            color = (0, 200, 0) if present else (0, 0, 255)

        center = spec["center"]
        cv2.circle(annotation, center, spec["outer"], color, 2)
        cv2.circle(annotation, center, spec["inner"], color, 2)

        results.append({
            "component": name,
            "reference_scores": [round(v, 4) for v in ref_scores],
            "test_score": round(score, 4),
            "threshold": round(threshold, 4),
            "status": status,
        })

    # Place text in a bottom report strip, not over the gear teeth.
    strip = np.zeros((110, annotation.shape[1], 3), dtype=np.uint8)
    strip[:] = (32, 32, 32)
    for i, r in enumerate(results):
        status = r["status"]
        color = ((0, 200, 0) if status == "POSSIBLY PRESENT"
                 else (0, 165, 255) if status.startswith("NOT ASSESSABLE")
                 else (0, 0, 255))
        label = f"{r['component']}: {status} ({r['test_score']:.3f})"
        cv2.putText(strip, label, (18, 28 + 32 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return results, strip



# ==============================================
# STAGE 5E: BEARING SUPPORTS, SCREWS, MOTOR PLATE,
# PLUG AND AN EXPERIMENTAL OUTPUT-WASHER CHECK
# ==============================================
# Fixed ROIs are for the 1132 x 790 *normalized* plate, not
# pixel locations in the original 1600 x 1200 image.
# Every template below is taken from correct reference images.
# Some labels refer to exposed visual proxies, not full part verification.
# Template matching is insensitive to small shifts, but not to severe
# occlusion, large rotation, or hardware behind other components.

STATIC_PARTS = {
    'Idler bearing block (visible support)': (706, 626, 19),
    'Compound bearing block (visible support)': (592, 146, 19),
    'Idler bearing screw A': (535, 355, 19),
    'Idler bearing screw B': (680, 603, 19),
    'Compound bearing screw A': (625, 172, 19),
    'Compound bearing screw B': (789, 455, 19),
    'Motor mounting plate (visible corner)': (1005, 333, 15),
    'Motor mounting screw A (tentative ROI)': (805, 194, 16),
    'Motor mounting screw B (tentative ROI)': (988, 314, 15),
    'Cable connector (plug position)': (921, 146, 24),
}


def local_template_score(reference, test, center_x, center_y, radius, search=7):
    """Best local NCC between one good patch and a test neighborhood.

    Values close to 1 suggest similar image structure. We search ±7 pixels
    to tolerate small alignment errors. This is not object identification.
    """
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    x, y, r = center_x, center_y, radius
    template = reference_gray[y-r:y+r+1, x-r:x+r+1]
    search_roi = test_gray[
        y-r-search:y+r+search+1,
        x-r-search:x+r+search+1
    ]
    if (template.size == 0 or search_roi.size == 0
            or search_roi.shape[0] < template.shape[0]
            or search_roi.shape[1] < template.shape[1]):
        raise ValueError(f'Invalid template position: {(x, y, r)}')
    response = cv2.matchTemplate(
        search_roi, template, cv2.TM_CCOEFF_NORMED
    )
    return float(np.nanmax(response))


def template_consistency(references, center):
    """Leave-one-reference-out scores; no test images used for calibration."""
    x, y, r = center
    scores = []
    for i in range(len(references)):
        scores.append(max(
            local_template_score(references[j], references[i], x, y, r)
            for j in range(len(references)) if j != i
        ))
    return scores


def inspect_static_parts(test, references, annotation):
    results = []
    for i, (name, center) in enumerate(STATIC_PARTS.items(), 1):
        x, y, r = center
        ref_consistency = template_consistency(references, center)
        score = max(
            local_template_score(ref, test, x, y, r)
            for ref in references
        )

        # Experimental: threshold calibrated on good-good agreement only.
        # Lower bound prevents a near-zero threshold for uncertain ROIs.
        threshold = max(0.45, 0.60 * min(ref_consistency))
        poor_reference_agreement = min(ref_consistency) < 0.50
        if score < threshold:
            status = 'SUSPECT MISSING / MISPLACED'
        elif poor_reference_agreement:
            status = 'REVIEW - references inconsistent'
        else:
            status = 'POSSIBLY PRESENT' 

        color = ((0, 200, 0) if status == 'POSSIBLY PRESENT'
                 else (0, 165, 255) if status.startswith('REVIEW')
                 else (0, 0, 255))
        cv2.rectangle(annotation, (x-r, y-r), (x+r, y+r), color, 2)
        cv2.putText(annotation, str(i), (x-r, max(12, y-r-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        results.append({
            'component': name,
            'method': 'local template NCC (visual proxy)',
            'test_score': round(score, 4),
            'threshold': round(threshold, 4),
            'good_reference_consistency': [round(s, 4) for s in ref_consistency],
            'status': status,
        })
    return results


def output_washer_score(image):
    """Proportion of low-saturation metal inside an outer fastener ring."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 80), (179, 55, 255))
    ring = ring_mask(image.shape, (302, 484), 35, 48)
    return region_fraction(mask, ring)


def inspect_output_washer(test, references, annotation, small_results):
    """Washer region is often occluded by the nut; don't infer both independently."""
    nut = next(r for r in small_results if r['component'] == 'Output gear nut')
    scores = [output_washer_score(r) for r in references]
    score = output_washer_score(test)
    threshold = 0.65 * min(scores)
    if nut['status'] != 'POSSIBLY PRESENT':
        status = 'NOT ASSESSABLE (output nut suspected missing)'
    else:
        status = ('POSSIBLY PRESENT' if score >= threshold
                  else 'SUSPECT MISSING / MISPLACED')
    color = ((0, 200, 0) if status == 'POSSIBLY PRESENT'
             else (0, 165, 255) if status.startswith('NOT ASSESSABLE')
             else (0, 0, 255))
    cv2.circle(annotation, (302, 484), 48, color, 2)
    return {
        'component': 'Output gear washer (outer visible band)',
        'method': 'metal pixels in an annulus, nut-dependent',
        'test_score': round(score, 4),
        'threshold': round(threshold, 4),
        'reference_scores': [round(s, 4) for s in scores],
        'status': status,
    }


def stage5e_report_strip(results, width):
    """Two-column report so labels do not hide the gear assembly."""
    rows = (len(results) + 1) // 2
    height = 33 + rows * 31
    strip = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(strip, 'STAGE 5E - preliminary checks (1-10 match numbered ROIs)',
                (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (235, 235, 235), 1, cv2.LINE_AA)
    col_width = width // 2
    for i, item in enumerate(results):
        col, row = divmod(i, rows)
        status = item['status']
        color = ((0, 200, 0) if status == 'POSSIBLY PRESENT'
                 else (0, 165, 255) if status.startswith(('REVIEW', 'NOT ASSESSABLE'))
                 else (0, 0, 255))
        display_name = item['component'].replace('(visible support)', '')
        display_name = display_name.replace('(tentative ROI)', '').replace('(plug position)', '')
        display_name = display_name.replace('(visible corner)', '')
        display_name = display_name.replace('(outer visible band)', '')
        label = f'{i+1}. {display_name[:31]}: {"?" if status != "POSSIBLY PRESENT" else "P"}'
        cv2.putText(strip, label, (10 + col*col_width, 52 + row*31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
    return strip


def resolve_test_image(argument):
    given = Path(argument)
    if given.is_absolute():
        return given
    if given.suffix.lower() in SUPPORTED_EXTENSIONS:
        direct = ROOT / given
        # Fall back to test_images only for bare names such as test_04.png;
        # preserve explicit paths (e.g. test_images/test_04.png) if missing.
        if direct.exists() or given.parent != Path('.'):
            return direct
        return ROOT / 'test_images' / given.name
    return ROOT / 'test_images' / f'{argument}.png'



# ==============================================
# STAGE 6 + 7C: REPORTING AND GEOMETRY DIAGNOSTIC
# ==============================================
# The current detectors have NOT been validated on unseen missing/misplaced
# parts. A visually correct assembly must not receive a certified "OK" yet.
DETECTORS_VALIDATED = False
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
# The visible-proxy option is an explicitly assumption-based assessment of
# the overhead image, NOT a claim that the hidden motor body was seen.
# Keep strict mode as the default for users who require direct verification.
MOTOR_BODY_NAME = 'Motor body (behind mount/pinion)'
MOTOR_VISIBLE_PROXIES = (
    'Motor pinion gear',
    'Motor mounting plate (visible corner)',
    'Motor mounting screw A (tentative ROI)',
    'Motor mounting screw B (tentative ROI)',
    'Cable connector (plug position)',
)
DECISION_POLICIES = ('strict', 'visible-proxy')
CSV_COLUMNS = [
    'image', 'verdict', 'binary_fallback', 'reason', 'decision_policy',
    'conditional_ok', 'assumptions_used', 'suspected_count', 'review_count',
    'suspected_parts', 'review_parts', 'seconds',
    'annotated_path', 'json_path', 'error'
]


def classify_results(results, policy='strict'):
    """Choose an inspection verdict without hiding unassessable components.

    Strict mode: existing fail-closed behavior. A hidden part prevents OK.
    Visible-proxy mode: only the unseen motor BODY can be waived, and only
    under the stated assumption that all five visible motor proxies pass.
    No other REVIEW or NOT ASSESSABLE component can be waived.

    In either mode this is experimental image analysis, not a certified
    guarantee about the physical assembly.
    """
    if policy not in DECISION_POLICIES:
        raise ValueError(f'Unknown decision policy: {policy}')

    suspects = [r['component'] for r in results
                if r['status'].startswith('SUSPECT')]
    unresolved = [r['component'] for r in results
                  if r['status'].startswith(('REVIEW', 'NOT ASSESSABLE'))]

    assumptions_used = []
    waived_checks = []
    by_name = {r['component']: r for r in results}
    if policy == 'visible-proxy' and MOTOR_BODY_NAME in unresolved:
        proxies_are_present = all(
            by_name.get(name, {}).get('status') == 'POSSIBLY PRESENT'
            for name in MOTOR_VISIBLE_PROXIES
        )
        if proxies_are_present:
            unresolved.remove(MOTOR_BODY_NAME)
            waived_checks.append(MOTOR_BODY_NAME)
            assumptions_used.append(
                'Assume the hidden motor body is installed when the visible '
                'pinion, mounting plate, both mounting screw regions and '
                'plug all pass their appearance checks; NOT directly visible.'
            )

    conditional_ok = False
    if suspects:
        verdict = 'NOT OK'
        reason = 'Suspected missing or misplaced visible required component(s)'
    elif unresolved:
        verdict = 'REVIEW REQUIRED'
        reason = 'One or more required checks remain unresolved'
    elif policy == 'visible-proxy' and waived_checks:
        verdict = 'OK'
        conditional_ok = True
        reason = 'Conditional OK for visible inspection under documented hidden-motor assumption'
    elif not DETECTORS_VALIDATED:
        verdict = 'REVIEW REQUIRED'
        reason = 'Experimental detectors have not been validated on independent faults'
    else:
        verdict = 'OK'
        reason = 'All required checks passed independently validated acceptance criteria'

    return {
        'verdict': verdict,
        'binary_fallback': 'OK' if verdict == 'OK' else 'NOT OK',
        'reason': reason,
        'suspected_parts': suspects,
        'review_parts': unresolved,
        'waived_checks': waived_checks,
        'assumptions_used': assumptions_used,
        'decision_policy': policy,
        'conditional_ok': conditional_ok,
        'detectors_validated': DETECTORS_VALIDATED,
    }


def add_final_strip(annotation, decision):
    """Keep the verdict below the image; do not conceal mechanical details."""
    width = annotation.shape[1]
    strip = np.full((120, width, 3), (26, 26, 26), dtype=np.uint8)
    verdict = decision['verdict']
    color = ((0, 0, 255) if verdict == 'NOT OK'
             else (0, 165, 255) if verdict == 'REVIEW REQUIRED'
             else (0, 200, 0))
    cv2.putText(strip, 'STAGE 7E  |  INSPECTION SUMMARY',
                (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(strip, f'VERDICT: {verdict}',
                (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                color, 2, cv2.LINE_AA)
    counts = (f"Suspected: {len(decision['suspected_parts'])}"
              f"    Review / not assessable: {len(decision['review_parts'])}"
              f"    Binary fail-closed: {decision['binary_fallback']}")
    cv2.putText(strip, counts, (14, 94), cv2.FONT_HERSHEY_SIMPLEX,
                0.53, (225, 225, 225), 1, cv2.LINE_AA)
    footer = ('CONDITIONAL OK: hidden motor assumed from visible parts'
              if decision['conditional_ok'] else
              'Experimental detectors: not a certified acceptance')
    cv2.putText(strip, footer,
                (14, 113), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (190, 190, 190), 1, cv2.LINE_AA)
    return strip


def resolve_output_path(argument):
    target = Path(argument)
    return target if target.is_absolute() else ROOT / target


def load_references():
    """Load the good reference images only once per CLI invocation."""
    references = []
    for i in range(1, 4):
        image_path = ROOT / 'reference_good' / f'good_ref_{i:02d}.png'
        references.append(normalize_plate(load_image(image_path)))
    return references


def run_inspection(image_path, references, output_dir, policy='strict'):
    """Run unchanged Stages 5B-5E and save Stage 6 artifacts."""
    start = time.perf_counter()
    original = load_image(image_path)
    test = normalize_plate(original)
    annotation = test.copy()

    gear_results = inspect_gears(test, references, annotation)
    bolt_results = inspect_bolts(test, references, annotation)
    small_results, strip_5d = inspect_small_parts(
        test, references, annotation, gear_results
    )
    static_results = inspect_static_parts(test, references, annotation)
    washer_result = inspect_output_washer(
        test, references, annotation, small_results
    )
    motor_result = {
        'component': 'Motor body (behind mount/pinion)',
        'method': 'not independently visible',
        'test_score': None,
        'threshold': None,
        'status': 'NOT ASSESSABLE (motor body not independently visible)',
    }
    stage5e_results = static_results + [washer_result, motor_result]
    results = gear_results + bolt_results + small_results + stage5e_results
    decision = classify_results(results, policy=policy)
    strip_5e = stage5e_report_strip(stage5e_results, annotation.shape[1])
    final_strip = add_final_strip(annotation, decision)
    annotated = np.vstack([annotation, strip_5d, strip_5e, final_strip])

    name = image_path.stem
    annotated_path = output_dir / f'{name}_inspection_stage6.png'
    normalized_path = output_dir / f'{name}_normalized.png'
    json_path = output_dir / f'{name}_inspection_stage6.json'
    for path, img in ((annotated_path, annotated), (normalized_path, test)):
        if not cv2.imwrite(str(path), img):
            raise OSError(f'Could not save image to {path}')

    seconds = round(time.perf_counter() - start, 3)
    report = {
        'image': str(image_path.resolve()),
        'stage': '7E (experimental Stages 5B-5E + explicit decision policy)',
        'verdict': decision['verdict'],
        'binary_fallback': decision['binary_fallback'],
        'reason': decision['reason'],
        'detectors_validated': DETECTORS_VALIDATED,
        'decision_policy': policy,
        'conditional_ok': decision['conditional_ok'],
        'assumptions_used': decision['assumptions_used'],
        'waived_checks': decision['waived_checks'],
        'suspected_parts': decision['suspected_parts'],
        'review_parts': decision['review_parts'],
        'seconds': seconds,
        'component_results': results,
        'warnings': [
            'Presence scores do not prove correct component geometry or placement.',
            'Hidden motor body is not independently visible. Visible-proxy policy assumes its presence from five visible motor-related checks.',
            'Binary fallback NOT OK for review cases does NOT mean a verified physical fault.',
        ],
    }
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)

    row = {
        'image': image_path.name,
        'verdict': decision['verdict'],
        'binary_fallback': decision['binary_fallback'],
        'reason': decision['reason'],
        'decision_policy': policy,
        'conditional_ok': decision['conditional_ok'],
        'assumptions_used': '; '.join(decision['assumptions_used']),
        'suspected_count': len(decision['suspected_parts']),
        'review_count': len(decision['review_parts']),
        'suspected_parts': '; '.join(decision['suspected_parts']),
        'review_parts': '; '.join(decision['review_parts']),
        'seconds': seconds,
        'annotated_path': str(annotated_path),
        'json_path': str(json_path),
        'error': '',
    }
    return row


def print_summary(rows):
    print('\n' + '=' * 88)
    print('STAGE 7E  |  GEAR ASSEMBLY INSPECTION SUMMARY')
    print('=' * 88)
    print(f"{'IMAGE':<20} {'VERDICT':<17} {'SUSPECT':>7} {'REVIEW':>7} {'SEC':>8}")
    print('-' * 88)
    for item in rows:
        seconds = item['seconds']
        duration = str(seconds) if seconds != '' else '-'
        print(f"{item['image']:<20} {item['verdict']:<17} "
              f"{str(item['suspected_count']):>7} "
              f"{str(item['review_count']):>7} {duration:>8}")
    print('-' * 88)
    for item in rows:
        if item['suspected_parts']:
            print(f"{item['image']}: SUSPECTED: {item['suspected_parts']}")
        if item['error']:
            print(f"{item['image']}: ERROR: {item['error']}")
    print('NOTE: REVIEW REQUIRED uses NOT OK binary fallback (fail-closed).')
    print('NOTE: Visible-proxy OK is conditional on documented hidden-motor assumption, NOT direct motor inspection.')
    print('NOTE: Detection thresholds are experimental; unseen physical fault accuracy is unmeasured.')


def save_summary_csv(rows, output_dir):
    target = output_dir / 'inspection_summary.csv'
    with target.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return target



def _metric(value):
    """Format a detector parameter for the terminal (not a confidence score)."""
    if value is None:
        return '-'
    if isinstance(value, (int, float)):
        return f'{value:.4f}'
    return str(value)


def print_validation_parameters(json_path):
    """Read the saved JSON report, so displayed metrics match the output file."""
    with Path(json_path).open('r', encoding='utf-8') as handle:
        report = json.load(handle)

    print('\n' + '=' * 105)
    print('COMPONENT VALIDATION PARAMETERS (experimental image features, not model accuracy)')
    print('=' * 105)
    print(f"Image: {Path(report['image']).name}")
    print(f"Verdict: {report['verdict']}  |  Policy: {report['decision_policy']}"
          f"  |  Conditional OK: {report['conditional_ok']}")
    print('Blue-plate alignment: completed (component layout similarity NOT independently verified)')
    print(f"{'COMPONENT':<44} {'SCORE':>8} {'LIMIT':>8} {'REF MIN':>8}   STATUS")
    print('-' * 105)
    counts = {'passed': 0, 'suspect': 0, 'review': 0}
    for result in report['component_results']:
        status = result['status']
        if status == 'POSSIBLY PRESENT':
            counts['passed'] += 1
        elif status.startswith('SUSPECT'):
            counts['suspect'] += 1
        else:
            counts['review'] += 1
        ref_values = result.get('reference_scores')
        if not ref_values:
            ref_values = result.get('good_reference_consistency')
        ref_min = min(ref_values) if ref_values else None
        component = result['component'][:44]
        print(f"{component:<44} {_metric(result.get('test_score')):>8} "
              f"{_metric(result.get('threshold')):>8} {_metric(ref_min):>8}   {status}")

        # Bolts have additional shape evidence beyond metal-pixel coverage.
        shape = result.get('bolt_shape_check')
        if shape:
            print('  Bolt shape: '
                  f"edges={_metric(shape.get('test_edge_count'))} "
                  f"(limit={_metric(shape.get('edge_threshold'))}), "
                  f"contrast={_metric(shape.get('test_core_contrast'))} "
                  f"(max={_metric(shape.get('core_contrast_limit'))}), "
                  f"shape evidence={shape.get('shape_evidence', '-')}")
    print('-' * 105)
    print(f"Component checks: {counts['passed']} possibly present, "
          f"{counts['suspect']} suspect, {counts['review']} unresolved")
    if report.get('suspected_parts'):
        print('Suspected: ' + '; '.join(report['suspected_parts']))
    if report.get('review_parts'):
        print('Review: ' + '; '.join(report['review_parts']))
    if report.get('assumptions_used'):
        print('Assumption: ' + '; '.join(report['assumptions_used']))
    print('These scores are raw feature measurements and experimental thresholds, NOT accuracy or probability.')
    print(f'Report JSON: {json_path}')


def open_inspection_result(path):
    """Open the saved annotated result with the operating system's default app."""
    path = Path(path).resolve()
    if not path.exists():
        print(f'Cannot open result; file is missing: {path}')
        return False
    try:
        if os.name == 'nt':
            os.startfile(str(path))  # Windows 11: default image viewer / File Explorer
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
                print('No desktop display detected. Open this file manually:')
                print(f'  {path}')
                return False
            subprocess.Popen(['xdg-open', str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, FileNotFoundError) as exc:
        print(f'Automatic opening failed: {exc}')
        print(f'Open this file manually: {path}')
        return False
    print(f'Opening: {path}')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 7E inspection: strict or assumption-based visible-proxy verdict'
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image', type=str,
                      help='e.g. test_images/test_06.png or test_06')
    mode.add_argument('--folder', type=str,
                      help='e.g. test_images; processes supported images in this folder')
    parser.add_argument('--output', default='output',
                        help='Output directory; defaults to project/output')
    parser.add_argument('--policy', choices=DECISION_POLICIES, default='strict',
                        help='strict = fail closed (default); visible-proxy = conditional OK if all visible motor-related parts pass')
    parser.add_argument('--show', action='store_true',
                        help='Print component validation parameters and open the annotated image. '
                             'For --folder, opens the output folder.')
    parser.add_argument('--details', action='store_true',
                        help='Print component validation parameters without opening images.')
    parser.add_argument('--client-report', action='store_true',
                        help='Also save a clean numbered client report PNG and a separate layout-risk note. '
                             'With --show, open the client report rather than the debug annotation.')
    args = parser.parse_args()

    output_dir = resolve_output_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = load_references()

    if args.image is not None:
        images = [resolve_test_image(args.image)]
    else:
        folder = Path(args.folder)
        folder = folder if folder.is_absolute() else ROOT / folder
        if not folder.is_dir():
            parser.error(f'Input folder does not exist: {folder}')
        images = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not images:
            parser.error(f'No supported images in {folder}')

    rows = []
    client_reports = {}
    for image_path in images:
        print(f'Inspecting {image_path.name} ...')
        try:
            row = run_inspection(image_path, references, output_dir, policy=args.policy)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            row = {
                'image': image_path.name,
                'verdict': 'ERROR',
                'binary_fallback': 'NOT OK',
                'reason': 'Inspection could not be completed',
                'decision_policy': args.policy,
                'conditional_ok': False,
                'assumptions_used': '',
                'suspected_count': '',
                'review_count': '',
                'suspected_parts': '',
                'review_parts': '',
                'seconds': '',
                'annotated_path': '',
                'json_path': '',
                'error': str(exc),
            }
            print(f'  ERROR: {exc}')
        else:
            print(f"  VERDICT: {row['verdict']}  |  "
                  f"Binary fallback: {row['binary_fallback']}")
            if row['suspected_parts']:
                print(f"  Suspected: {row['suspected_parts']}")
            if row['review_parts']:
                print(f"  Review: {row['review_parts']}")
            if row['conditional_ok']:
                print('  CONDITIONAL OK: hidden motor inferred from visible proxies; not directly observed')
            print(f"  Time: {row['seconds']} seconds")
            if args.show or args.details:
                print_validation_parameters(row['json_path'])
            if args.client_report:
                # Presentation-only extension; the detection decisions and the
                # original PNG/JSON/CSV remain exactly as they were.
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        'gear_client_report', str(ROOT / 'client_report.py'))
                    if spec is None or spec.loader is None:
                        raise RuntimeError('Unable to load client_report.py')
                    client_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(client_module)
                    positions = {}
                    for part, item in GEARS.items():
                        positions[part] = ('circle', *item['center'], item['outer'])
                    for corner, center in BOLTS.items():
                        positions[f'{corner} Bolt head'] = ('circle', *center, 18)
                        positions[f'{corner} Washer'] = ('circle', *center, 36)
                    for part, item in SMALL_PARTS.items():
                        positions[part] = ('circle', *item['center'], item['outer'])
                    for part, (x, y, radius) in STATIC_PARTS.items():
                        positions[part] = ('rect', x, y, radius)
                    positions['Output gear washer (outer visible band)'] = (
                        'circle', 302, 484, 48)
                    stem = Path(row['json_path']).name.replace('_inspection_stage6.json', '')
                    normal = output_dir / f'{stem}_normalized.png'
                    report_path, summary = client_module.create_report(
                        row['json_path'], normal, positions)
                    client_reports[row['image']] = str(report_path)
                    print(f'  Client report: {report_path}')
                    print(f'  Layout review: {summary["layout_screening"]["status"]}')
                except Exception as exc:
                    # A presentation failure must not erase a valid detector report.
                    print(f'  WARNING: Client report not generated: {exc}')
        rows.append(row)

    print_summary(rows)
    csv_path = save_summary_csv(rows, output_dir)
    print(f'CSV summary: {csv_path}')
    print(f'Annotated images and JSON reports: {output_dir}')
    if args.show:
        # Single image: automatically open the exact annotated result, never
        # an unrelated normalized image or a result left by a prior run.
        if args.image is not None:
            if rows and rows[0]['annotated_path']:
                if args.client_report and client_reports.get(rows[0]['image']):
                    open_inspection_result(client_reports[rows[0]['image']])
                else:
                    open_inspection_result(rows[0]['annotated_path'])
        else:
            # Don't launch dozens of image-viewer windows for a batch.
            open_inspection_result(output_dir)
    if any(row['verdict'] == 'ERROR' for row in rows):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
