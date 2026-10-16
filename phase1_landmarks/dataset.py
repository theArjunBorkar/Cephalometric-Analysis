"""
ISBI 2015 cephalometric dataset loader.

Parses raw images + the two annotator folders, builds Gaussian-heatmap
targets for heatmap-regression training.
"""
import os
import glob
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


NUM_LANDMARKS = 19
AVERAGE_ANNOTATORS = True  # average junior+senior; set False to use one only


def _read_annotation(txt_path, num_points=NUM_LANDMARKS):
    """Reads the first `num_points` lines of an ISBI annotation file as (x, y) pixel coords."""
    coords = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(coords) >= num_points:
                break
            parts = line.split(",")
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            coords.append((x, y))
    if len(coords) != num_points:
        raise ValueError(f"Expected {num_points} points in {txt_path}, got {len(coords)}")
    return np.array(coords, dtype=np.float32)  # (19, 2) in original pixel space


def _find_image_for_id(image_dirs, image_id):
    """image_id is typically a stem like '001'. Searches given directories for a matching file."""
    for d in image_dirs:
        for ext in (".bmp", ".jpg", ".jpeg", ".png"):
            candidate = os.path.join(d, image_id + ext)
            if os.path.exists(candidate):
                return candidate
    return None


def build_file_list(data_root, image_subdirs=None):
    """
    Builds a list of (image_path, annotation_path) pairs.

    Expects:
      data_root/RawImage/*/*.bmp   (any subfolders, searched recursively)
      data_root/AnnotationsByMD/400_junior/*.txt
      data_root/AnnotationsByMD/400_senior/*.txt

    image_subdirs: optional list of folder names (e.g. ["Test1Data", "Test2Data"])
      to restrict the search to specific RawImage subfolders. Use this to evaluate
      on the ISBI challenge's official held-out test sets instead of a random split
      of everything. Leave as None to search all RawImage subfolders (training-time
      behavior).
    """
    if image_subdirs:
        image_dirs = [os.path.join(data_root, "RawImage", d) for d in image_subdirs]
        image_dirs = [d for d in image_dirs if os.path.isdir(d)]
        if not image_dirs:
            raise FileNotFoundError(
                f"None of the requested subfolders {image_subdirs} were found under "
                f"{os.path.join(data_root, 'RawImage')}. Check the exact folder names in "
                "your download (case/spelling can vary)."
            )
    else:
        image_dirs = glob.glob(os.path.join(data_root, "RawImage", "*"))
        if not image_dirs:
            image_dirs = [os.path.join(data_root, "RawImage")]

    junior_dir = os.path.join(data_root, "AnnotationsByMD", "400_junior")
    senior_dir = os.path.join(data_root, "AnnotationsByMD", "400_senior")

    junior_files = sorted(glob.glob(os.path.join(junior_dir, "*.txt")))
    if not junior_files:
        raise FileNotFoundError(
            f"No annotation .txt files found under {junior_dir}. "
            "Check --data_root and folder names (edit dataset.py if yours differ)."
        )

    entries = []
    for jf in junior_files:
        image_id = os.path.splitext(os.path.basename(jf))[0]
        img_path = _find_image_for_id(image_dirs, image_id)
        if img_path is None:
            continue  # skip if image missing

        if AVERAGE_ANNOTATORS:
            sf = os.path.join(senior_dir, os.path.basename(jf))
            if os.path.exists(sf):
                pts_j = _read_annotation(jf)
                pts_s = _read_annotation(sf)
                pts = (pts_j + pts_s) / 2.0
            else:
                pts = _read_annotation(jf)
        else:
            pts = _read_annotation(jf)

        entries.append((img_path, pts))

    if not entries:
        raise RuntimeError(
            "Found annotation files but couldn't match any to images. "
            "Check that image filenames (minus extension) match annotation filenames."
        )
    return entries


def _make_heatmap(size_hw, point_xy, sigma=3.0):
    """Single Gaussian heatmap for one landmark, in output (heatmap-resolution) coordinates."""
    h, w = size_hw
    x, y = point_xy
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


class CephDataset(Dataset):
    """
    Returns:
      image: FloatTensor (1, H, W), normalized to [0, 1]
      heatmaps: FloatTensor (19, H_out, W_out)
      meta: dict with original image size and landmark coords (for eval)
    """

    def __init__(self, entries, image_size=512, heatmap_size=256, sigma=3.0, augment=False):
        self.entries = entries
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, pts = self.entries[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        orig_h, orig_w = img.shape

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        img_resized = cv2.resize(img, (self.image_size, self.image_size))

        pts_resized = pts.copy()
        pts_resized[:, 0] *= scale_x
        pts_resized[:, 1] *= scale_y

        if self.augment:
            img_resized, pts_resized = self._augment(img_resized, pts_resized)

        # scale landmark coords into heatmap resolution
        hm_scale = self.heatmap_size / self.image_size
        pts_hm = pts_resized * hm_scale

        heatmaps = np.stack(
            [_make_heatmap((self.heatmap_size, self.heatmap_size), p, self.sigma) for p in pts_hm],
            axis=0,
        )

        image_t = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0)
        heatmaps_t = torch.from_numpy(heatmaps)

        meta = {
            "orig_size": (orig_h, orig_w),
            "landmarks_orig_px": pts,  # original-pixel-space ground truth, for mm error eval
            "image_path": img_path,
            "scale_to_input": (scale_x, scale_y),
        }
        return image_t, heatmaps_t, meta

    def _augment(self, img, pts):
        # mild rotation (+/- 8 deg) and brightness/contrast jitter.
        # No horizontal flip: left/right is anatomically meaningful on a lateral skull view.
        h, w = img.shape
        angle = np.random.uniform(-8, 8)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=1)
        pts = (M @ pts_h.T).T

        alpha = np.random.uniform(0.85, 1.15)  # contrast
        beta = np.random.uniform(-15, 15)  # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return img, pts


def make_train_val_split(entries, val_fraction=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(entries))
    rng.shuffle(idx)
    n_val = int(len(entries) * val_fraction)
    val_idx = set(idx[:n_val].tolist())
    train_entries = [e for i, e in enumerate(entries) if i not in val_idx]
    val_entries = [e for i, e in enumerate(entries) if i in val_idx]
    return train_entries, val_entries