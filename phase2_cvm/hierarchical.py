"""
Defines the CVM phase grouping used for hierarchical classification, and a
thin Dataset wrapper that remaps AarizCVMDataset's 6-way stage labels into
either 3-way phase labels or 2-way within-phase labels, without duplicating
any of the image loading/cropping/CLAHE logic already in dataset.py.

Phase grouping follows the standard clinical convention (Baccetti et al.):
  Early phase (pre-pubertal, growth not yet started):  CS1, CS2
  Peak phase  (active pubertal growth spurt):           CS3, CS4
  Late phase  (post-pubertal, growth completed):        CS5, CS6

Why this grouping specifically: it's clinically meaningful (these are the
groupings orthodontists actually reason in when deciding treatment timing),
and it splits your confusion matrix's error pattern in a useful way -- your
worst confusions (CS4/CS5/CS6) fall across the Peak/Late boundary and within
Late, so the binary specialist for Late (CS5 vs CS6) is doing the most
important job here.
"""
import torch
from torch.utils.data import Dataset

PHASE_NAMES = ["Early", "Peak", "Late"]

# 0-indexed CVM stage (0=CS1 .. 5=CS6) -> phase index (0=Early, 1=Peak, 2=Late)
PHASE_OF_STAGE = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}

# phase index -> the two 0-indexed stages within it, in a fixed order
# (this order defines what binary label 0 vs 1 means for that phase's specialist)
STAGES_IN_PHASE = {0: [0, 1], 1: [2, 3], 2: [4, 5]}


class PhaseDataset(Dataset):
    """
    Wraps an existing AarizCVMDataset instance and remaps its labels.

    mode="phase":  returns 3-way phase labels (0=Early, 1=Peak, 2=Late) for
                   every image in the base dataset.
    mode="binary": returns 2-way labels, but ONLY for images whose true stage
                   falls within `phase` -- use this to train/evaluate one
                   phase's binary specialist in isolation.
    """

    def __init__(self, base_dataset, mode="phase", phase=None):
        assert mode in ("phase", "binary")
        if mode == "binary" and phase is None:
            raise ValueError("phase must be given when mode='binary'")

        self.base = base_dataset
        self.mode = mode
        self.phase = phase

        if mode == "binary":
            allowed_stages = set(STAGES_IN_PHASE[phase])
            self.indices = [
                i for i, fname in enumerate(self.base.image_files)
                if self.base._load_cvm_stage(fname) in allowed_stages
            ]
            if not self.indices:
                raise RuntimeError(
                    f"No images found for phase={phase} ({PHASE_NAMES[phase]}, "
                    f"stages {STAGES_IN_PHASE[phase]}) in this split."
                )
        else:
            self.indices = list(range(len(self.base)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, stage_label, fname = self.base[base_idx]
        stage0 = int(stage_label.item())

        if self.mode == "phase":
            label = PHASE_OF_STAGE[stage0]
        else:  # binary
            allowed = STAGES_IN_PHASE[self.phase]
            label = allowed.index(stage0)  # 0 or 1, order fixed by STAGES_IN_PHASE

        return image, torch.tensor(label, dtype=torch.long), fname

    def class_distribution(self):
        """Counts for whatever label space this wrapper currently exposes.
        Reads labels directly (no image decoding) for efficiency."""
        counts = {}
        for base_idx in self.indices:
            fname = self.base.image_files[base_idx]
            stage0 = self.base._load_cvm_stage(fname)
            if self.mode == "phase":
                key = PHASE_OF_STAGE[stage0]
            else:
                key = STAGES_IN_PHASE[self.phase].index(stage0)
            counts[key] = counts.get(key, 0) + 1
        return counts
