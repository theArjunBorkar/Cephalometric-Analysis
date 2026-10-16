"""
Loads Aariz's 29 cephalometric landmarks and computes a crop box around the
C2-C4 cervical vertebrae region, anchored on the Articulare landmark.

Why Articulare: Aariz does not ship dedicated cervical-vertebra landmarks, but
Articulare (the intersection of the mandibular ramus and the cranial base) is
the closest available skeletal point to where C2-C4 sit in a lateral
cephalogram -- the vertebrae are visible just below and slightly posterior to
it. This is a geometric heuristic, not a learned or validated detector, so
ALWAYS sanity-check it visually (see visualize_crop.py) before trusting it.

Landmark order (0-indexed), from the original Aariz paper's Table 3:
    0: A-point   1: ANS        2: B-point     3: Menton      4: Nasion
    5: Orbitale  6: Pogonion   7: PNS         8: Ramus       9: Sella
    10: Articulare  11: Condylion  12: Gnathion  13: Gonion  14: Porion
    15-22: dental points (not used here)
    23-28: soft-tissue points (not used here)
"""
import os
import json
import numpy as np

IDX_SELLA = 9
IDX_NASION = 4
IDX_ARTICULARE = 10

LANDMARKS_DIRNAME = os.path.join("Annotations", "Cephalometric Landmarks")
SENIOR_SUBDIR = "Senior Orthodontists"
JUNIOR_SUBDIR = "Junior Orthodontists"


def _load_landmark_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    pts = [[lm["value"]["x"], lm["value"]["y"]] for lm in data["landmarks"]]
    return np.array(pts, dtype=np.float32)  # (29, 2)


def load_landmarks(dataset_root, split_name, image_filename, average_annotators=True):
    """
    Loads the 29 landmarks for one image, averaging senior+junior annotations
    by default (same convention used in Phase 1 for the ISBI dataset).
    """
    stem = os.path.splitext(image_filename)[0]
    senior_path = os.path.join(dataset_root, split_name, LANDMARKS_DIRNAME, SENIOR_SUBDIR, stem + ".json")
    junior_path = os.path.join(dataset_root, split_name, LANDMARKS_DIRNAME, JUNIOR_SUBDIR, stem + ".json")

    if average_annotators and os.path.exists(senior_path) and os.path.exists(junior_path):
        return (_load_landmark_json(senior_path) + _load_landmark_json(junior_path)) / 2.0
    elif os.path.exists(senior_path):
        return _load_landmark_json(senior_path)
    elif os.path.exists(junior_path):
        return _load_landmark_json(junior_path)
    else:
        raise FileNotFoundError(
            f"No landmark annotation found for {image_filename} under "
            f"{os.path.join(dataset_root, split_name, LANDMARKS_DIRNAME)}."
        )


def compute_cervical_crop_box(landmarks, width_scale=1.4, height_scale=2.2,
                               x_offset_scale=-0.35, y_offset_scale=0.9):
    """
    Computes a crop box (x0, y0, x1, y1) in original image pixel coordinates,
    anchored on Articulare and sized relative to the Sella-Nasion distance
    (a standard cephalometric scale reference, roughly proportional to skull
    size -- this is what lets one crop formula work across Aariz's 7 different
    imaging devices with very different pixel spacings).

    Parameters (all in units of the S-N distance) are tunable -- if your visual
    check (visualize_crop.py) shows the box missing the vertebrae, adjust these:
      width_scale, height_scale : size of the crop box
      x_offset_scale            : shift from Articulare along the S->N direction
                                   (negative = posterior/behind Articulare, which
                                   is where the vertebrae sit relative to the jaw)
      y_offset_scale            : shift downward (in image pixel Y, i.e. toward
                                   the neck) from Articulare
    """
    sella = landmarks[IDX_SELLA]
    nasion = landmarks[IDX_NASION]
    articulare = landmarks[IDX_ARTICULARE]

    sn_vec = nasion - sella
    sn_dist = np.linalg.norm(sn_vec)
    if sn_dist < 1e-3:
        raise ValueError("Degenerate Sella-Nasion distance -- check landmark annotations.")
    sn_unit = sn_vec / sn_dist  # unit vector pointing from Sella toward Nasion (roughly "forward")

    # Center of the crop: start at Articulare, shift backward (away from Nasion,
    # i.e. posterior) and downward toward the neck.
    center_x = articulare[0] + x_offset_scale * sn_dist * sn_unit[0]
    center_y = articulare[1] + y_offset_scale * sn_dist  # downward in image Y

    half_w = 0.5 * width_scale * sn_dist
    half_h = 0.5 * height_scale * sn_dist

    x0, x1 = center_x - half_w, center_x + half_w
    y0, y1 = center_y - half_h, center_y + half_h
    return x0, y0, x1, y1


def crop_image_to_box(image, box):
    """image: HxW(xC) numpy array. box: (x0,y0,x1,y1) in pixel coords (float, may be out of bounds)."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop box after clamping to image bounds: {(x0, y0, x1, y1)} for image size {(h, w)}")
    return image[y0:y1, x0:x1]
