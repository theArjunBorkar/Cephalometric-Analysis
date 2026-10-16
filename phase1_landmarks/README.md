# Cephalometric Landmark Detection — Phase 1

Heatmap-regression model for detecting the 19 standard craniofacial landmarks
on lateral cephalometric X-rays (ISBI 2015 Grand Challenge format), as a
foundation for a later CVM-staging extension.

## Expected dataset layout

The ISBI 2015 challenge data is usually distributed like this:

```
data/
  RawImage/
    TrainingData/         # 150 images, ISBI-style .bmp
    Test1Data/             # 150 images
    Test2Data/             # 100 images
  AnnotationsByMD/
    400_junior/            # .txt files, one per image, 19 lines of "x,y"
    400_senior/
```

If your download's folder names differ, just edit the paths at the top of
`train.py` — nothing else needs to change.

Each annotation `.txt` file has 19 lines of `x,y` pixel coordinates
(one line per landmark, in a fixed known order), followed by two lines with
the physical pixel spacing ratio (used to convert pixel error to mm) — the
loader in `dataset.py` reads the first 19 lines and ignores the rest.

We average the junior/senior annotations by default (this is standard
practice in the literature, and reduces annotator bias/noise) — see
`dataset.py: AVERAGE_ANNOTATORS`.

## Setup

```bash
pip install -r requirements.txt
```

Works fine in Google Colab — just `!pip install -r requirements.txt` after
uploading/mounting your dataset, or `!unzip your_isbi_dataset.zip -d data/`.

## Files

- `dataset.py` — parses ISBI images + annotations, builds Gaussian heatmap
  targets for each landmark, standard augmentation (flip is NOT used since
  left/right matters on a lateral skull view; we use mild rotation/
  brightness/contrast jitter instead).
- `model.py` — a compact U-Net that takes a grayscale ceph image and outputs
  19 heatmap channels (one per landmark).
- `train.py` — training loop with checkpointing and validation.
- `evaluate.py` — computes per-landmark mean radial error (MRE) in mm and
  Success Detection Rate (SDR) at 2mm/2.5mm/3mm/4mm thresholds — the exact
  metrics used in the ISBI challenge papers, so you can compare your results
  directly against published numbers.
- `utils.py` — shared helpers (heatmap→coordinate decoding, visualization).

## Quickstart

```bash
python train.py --data_root ./data --epochs 50 --batch_size 8
python evaluate.py --data_root ./data --checkpoint checkpoints/best.pt
```

## Notes for Phase 2 (CVM staging)

Once this baseline works, the natural extension is:
1. Manually label CVM stage (CS1–CS6) for a subset of images you have access
   to (yours + any you can source), using Baccetti's criteria on C2–C4 shape.
2. Either (a) train a small classifier on a cropped C2–C4 region (crop
   using this model's landmark outputs as an anchor, if any of the 19 points
   are close to the vertebrae — otherwise crop manually), or (b) add new
   cervical landmark labels yourself and retrain this same architecture with
   more output channels.
3. Given how small your labeled set will likely be, look into transfer
   learning from this Phase 1 model's encoder weights, or heavy augmentation,
   rather than training a CVM classifier from scratch.