"""
Loads Aariz cephalograms + CVM stage labels for classification.

Structure (matches the official Aariz dataset.py / README):
    <dataset_root>/<Split>/Cephalograms/<file>.jpg
    <dataset_root>/<Split>/Annotations/CVM Stages/<file>.json
        -> {"cvm_stage": {"value": 1..6}}

If your local copy uses different folder names, adjust the constants below.
"""
import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

NUM_CVM_STAGES = 6
CEPHALOGRAMS_DIRNAME = "Cephalograms"
CVM_STAGES_DIRNAME = os.path.join("Annotations", "CVM Stages")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Standard ImageNet normalization stats -- used since the model backbone
# (ResNet18) is pretrained on ImageNet.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_clahe(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast-Limited Adaptive Histogram Equalization, applied on the L
    (lightness) channel in LAB space -- the standard approach for color
    images, since applying CLAHE independently per RGB channel can shift
    color balance. X-ray images are effectively grayscale already, but
    routing through LAB keeps this correct in general and matches Aariz's
    own suggested preprocessing (their repo lists CLAHE as one of the
    provided transforms).

    This locally boosts contrast, which should make low-contrast vertebral
    edges easier for the network to pick up -- especially helpful here since
    the crop region is small and Aariz spans 7 devices with different
    exposure/contrast characteristics.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def _resolve_split_dir(dataset_root, split):
    """
    Finds the split folder on disk regardless of casing. Aariz's own published
    loader code assumes capitalized names (Train/Valid/Test), but some
    downloads of the dataset use lowercase (train/valid/test) instead -- so
    we check what's actually there rather than assuming either.
    """
    split = split.strip().lower()
    if split not in ("train", "valid", "test"):
        raise ValueError(f"split must be 'train', 'valid', or 'test', got '{split}'")

    candidates = [split, split.capitalize(), split.upper()]
    for name in candidates:
        candidate_path = os.path.join(dataset_root, name)
        if os.path.isdir(candidate_path):
            return name

    raise FileNotFoundError(
        f"Couldn't find a '{split}' split folder (tried {candidates}) under "
        f"{dataset_root}. Check --data_root points at the folder that directly "
        f"contains your train/valid/test folders."
    )


class AarizCVMDataset(Dataset):
    def __init__(self, dataset_root, split, image_size=384, augment=False,
                 crop_to_cervical=False, crop_params=None, use_clahe=False):
        """
        crop_to_cervical: if True, crops each image to the C2-C4 cervical region
            (anchored on the Articulare landmark, see crop_utils.py) before
            resizing -- isolating the vertebrae from the rest of the skull.
            ALWAYS sanity-check this with visualize_crop.py first; it's a
            geometric heuristic, not a validated detector.
        crop_params: optional dict of overrides for compute_cervical_crop_box's
            width_scale/height_scale/x_offset_scale/y_offset_scale, if you've
            tuned them via visualize_crop.py.
        use_clahe: if True, applies CLAHE contrast enhancement (see
            apply_clahe() above) after cropping, before resizing.
        """
        self.dataset_root = dataset_root
        self.split_name = _resolve_split_dir(dataset_root, split)
        self.images_root = os.path.join(dataset_root, self.split_name, CEPHALOGRAMS_DIRNAME)
        self.cvm_root = os.path.join(dataset_root, self.split_name, CVM_STAGES_DIRNAME)
        self.crop_to_cervical = crop_to_cervical
        self.crop_params = crop_params or {}
        self.use_clahe = use_clahe

        if not os.path.isdir(self.images_root):
            raise FileNotFoundError(
                f"Couldn't find {self.images_root}. Check --data_root and that the "
                "Aariz folder structure matches (Train/Valid/Test, each with a "
                "'Cephalograms' subfolder)."
            )
        if not os.path.isdir(self.cvm_root):
            raise FileNotFoundError(
                f"Couldn't find {self.cvm_root}. Check the 'Annotations/CVM Stages' "
                "path -- adjust CVM_STAGES_DIRNAME at the top of dataset.py if your "
                "download uses different folder names."
            )

        all_files = sorted(os.listdir(self.images_root))
        self.image_files = [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_root}.")

        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def _label_path(self, image_filename):
        stem = os.path.splitext(image_filename)[0]
        return os.path.join(self.cvm_root, stem + ".json")

    def _load_cvm_stage(self, image_filename):
        label_path = self._label_path(image_filename)
        with open(label_path, "r") as f:
            data = json.load(f)
        stage_value = int(data["cvm_stage"]["value"])  # 1..6
        return stage_value - 1  # -> 0..5 for CrossEntropyLoss

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.images_root, fname)

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.crop_to_cervical:
            from crop_utils import load_landmarks, compute_cervical_crop_box, crop_image_to_box
            landmarks = load_landmarks(self.dataset_root, self.split_name, fname)
            box = compute_cervical_crop_box(landmarks, **self.crop_params)
            img = crop_image_to_box(img, box)

        if self.use_clahe:
            img = apply_clahe(img)

        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.augment:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        label = self._load_cvm_stage(fname)

        image_t = torch.from_numpy(img.copy())
        label_t = torch.tensor(label, dtype=torch.long)

        return image_t, label_t, fname

    def _augment(self, img):
        # Mild rotation + brightness/contrast jitter. No horizontal flip --
        # left/right orientation is anatomically meaningful on a lateral skull view.
        h, w = img.shape[:2]
        angle = np.random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        alpha = np.random.uniform(0.85, 1.15)
        beta = np.random.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img

    def class_distribution(self):
        """Returns a dict {stage_1_to_6: count} -- useful to check for class imbalance."""
        counts = {i: 0 for i in range(1, NUM_CVM_STAGES + 1)}
        for fname in self.image_files:
            stage0 = self._load_cvm_stage(fname)
            counts[stage0 + 1] += 1
        return counts