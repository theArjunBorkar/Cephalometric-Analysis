# Phase 2 — CVM Stage Classification (Aariz dataset)

Classifies each lateral cephalogram into one of 6 CVM stages (CS1-CS6),
using the Aariz benchmark dataset (Khalid et al., Nature Scientific Data 2025).

## Why this is a different kind of problem than Phase 1

Phase 1 (landmark detection) was per-pixel heatmap regression. CVM staging is
a **per-image classification** problem: one label (1-6) per cephalogram,
based on the visual shape of the C2-C4 cervical vertebrae. Aariz does not
ship dedicated cervical-vertebra landmark points, so this pipeline classifies
directly from the (resized) full image rather than a landmark-cropped region.
Cropping to just the vertebral region is a natural improvement to try later
(see "Possible improvements" below) once this baseline is working.

## Expected dataset layout (from Aariz's own dataset.py / README)

```
<dataset_root>/
  Train/
    Cephalograms/                       # image files
    Annotations/
      Cephalometric Landmarks/
        Senior Orthodontists/*.json
        Junior Orthodontists/*.json
      CVM Stages/*.json                 # {"cvm_stage": {"value": 1-6}}
  Valid/
    ... same structure ...
  Test/
    ... same structure ...
```

Each CVM Stages JSON filename matches its image filename (extension swapped
to `.json`), e.g. `Cephalograms/0001.jpg` <-> `CVM Stages/0001.json`.

If your downloaded folder names differ slightly (capitalization, spacing),
adjust the constants at the top of `dataset.py`.

## Approach

- **Backbone**: ResNet18, pretrained on ImageNet, final layer replaced with
  a 6-way classification head. With only ~1000 images, training a CNN from
  scratch would likely underfit or overfit badly -- transfer learning from
  ImageNet features (edges, textures, shapes) gives a much better starting
  point, even though X-rays look nothing like natural photos.
- **Loss**: standard cross-entropy over 6 classes. (CVM stages are ordinal --
  see "Possible improvements" for a more clinically-aware alternative.)
- **Augmentation**: mild rotation + brightness/contrast jitter, no horizontal
  flip (a lateral skull view has a meaningful left/right orientation).

## Files

- `dataset.py` — `AarizCVMDataset`, adapted from Aariz's own loader.
- `model.py` — ResNet18-based classifier.
- `train.py` — training loop using the official Train/Valid split.
- `evaluate.py` — evaluates on the official Test split: overall accuracy,
  per-class precision/recall, confusion matrix, and **adjacent-stage
  accuracy** (prediction within +/-1 stage of ground truth) -- this last one
  matters clinically, since CVM staging has real inter-annotator disagreement
  even among human orthodontists, and being one stage off is a much smaller
  error than being three stages off.

## Quickstart

```bash
pip install torch torchvision scikit-learn matplotlib tqdm opencv-python
python train.py --data_root /path/to/Aariz --epochs 30 --batch_size 16
python evaluate.py --data_root /path/to/Aariz --checkpoint checkpoints/best.pt
```

## Possible improvements (once the baseline works)

1. **Crop to the cervical region.** ✅ Implemented — see `crop_utils.py` and
   `visualize_crop.py`. Use `--crop_to_cervical` in train/evaluate.
2. **CLAHE contrast enhancement.** ✅ Implemented — see `apply_clahe()` in
   `dataset.py`. Use `--use_clahe`. Note: in testing on this dataset, CLAHE
   actually *hurt* macro-F1 (the model collapsed toward over-predicting the
   majority class) even though it slightly raised raw accuracy -- worth
   re-checking on your own data with the per-class report, not just accuracy.
3. **Hierarchical phase-classifier cascade.** ✅ Implemented — see
   `hierarchical.py`, `train_phase.py`, `train_binary.py`,
   `evaluate_hierarchical.py` below.
4. **Ordinal loss instead of plain cross-entropy** (e.g. CORN, or simply
   weighting the loss by |predicted stage - true stage|) -- not yet
   implemented. Since stages are ordered, penalizing a CS2-vs-CS6 mistake the
   same as a CS2-vs-CS3 mistake throws away information the task actually has.

## Hierarchical cascade (Early/Peak/Late + binary specialists)

Mirrors the approach used by the strongest published results on this exact
benchmark: first classify into 3 broad phases, then a phase-specific binary
classifier picks the exact stage. This directly targets adjacent-stage
confusion (e.g. CS4 vs CS5) rather than asking one model to resolve all 6
classes at once.

Phase grouping (see `hierarchical.py` for rationale):
  Early = CS1, CS2   |   Peak = CS3, CS4   |   Late = CS5, CS6

```bash
# 1. Train the coarse phase classifier
python3 train_phase.py --data_root ./Aariz --crop_to_cervical

# 2. Train each phase's binary specialist (run all three)
python3 train_binary.py --data_root ./Aariz --crop_to_cervical --phase 0
python3 train_binary.py --data_root ./Aariz --crop_to_cervical --phase 1
python3 train_binary.py --data_root ./Aariz --crop_to_cervical --phase 2

# 3. Evaluate the full cascade (chains all four models together)
python3 evaluate_hierarchical.py --data_root ./Aariz --crop_to_cervical
```

Pass the same `--crop_to_cervical` / `--use_clahe` flags consistently across
all four commands -- mismatched preprocessing between the phase classifier,
the specialists, and evaluation will silently give meaningless results.

`evaluate_hierarchical.py` also reports the phase classifier's own accuracy
separately, since phase-classification errors propagate downstream: a wrong
Early/Peak/Late call sends an image to the wrong specialist, which can never
recover the correct final stage. If overall cascade accuracy disappoints,
check this number first to see whether the bottleneck is phase confusion or
within-phase confusion.