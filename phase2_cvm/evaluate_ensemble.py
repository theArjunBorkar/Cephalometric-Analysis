"""
Evaluates an ensemble of models trained with different random seeds
(via train.py --seed N), averaging their softmax probabilities before taking
the final predicted class. This is the one lever from the published
state-of-the-art cascade (Helal et al.) we hadn't tried yet -- averaging
out individual-model noise, rather than changing the architecture.

Usage:
  python train.py --data_root ./Aariz --crop_to_cervical --seed 0
  python train.py --data_root ./Aariz --crop_to_cervical --seed 1
  python train.py --data_root ./Aariz --crop_to_cervical --seed 2
  python evaluate_ensemble.py --data_root ./Aariz --crop_to_cervical --seeds 0,1,2
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset import AarizCVMDataset, NUM_CVM_STAGES
from model import build_cvm_classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--seeds", type=str, required=True,
                    help="Comma-separated seeds to ensemble, e.g. '0,1,2'. Each must have "
                         "a corresponding checkpoints/best_seed<N>.pt from train.py --seed N.")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--crop_to_cervical", action="store_true",
                    help="Must match whatever all ensemble member checkpoints were trained with.")
    p.add_argument("--use_clahe", action="store_true",
                    help="Must match whatever all ensemble member checkpoints were trained with.")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    test_ds = AarizCVMDataset(args.data_root, "test", args.image_size, augment=False,
                               crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)
    print(f"Test set: {len(test_ds)} images")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    models = []
    for seed in seeds:
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_seed{seed}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Missing {ckpt_path} -- train it first with: "
                f"python train.py --data_root {args.data_root} "
                f"{'--crop_to_cervical ' if args.crop_to_cervical else ''}"
                f"{'--use_clahe ' if args.use_clahe else ''}--seed {seed}"
            )
        m = build_cvm_classifier(num_classes=NUM_CVM_STAGES, pretrained=False).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        models.append(m)
    print(f"Ensembling {len(models)} models (seeds {seeds})")

    all_preds, all_labels = [], []
    # Also track single-model (first seed) predictions, so we can report the
    # ensemble's improvement over a single run using the exact same test batches.
    all_single_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)

            probs_sum = None
            single_probs = None
            for i, m in enumerate(models):
                logits = m(images)
                probs = F.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
                if i == 0:
                    single_probs = probs

            avg_probs = probs_sum / len(models)
            preds = avg_probs.argmax(dim=1).cpu().numpy()
            single_preds = single_probs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_single_preds.extend(single_preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_single_preds = np.array(all_single_preds)
    all_labels = np.array(all_labels)

    single_exact_acc = (all_single_preds == all_labels).mean()
    ensemble_exact_acc = (all_preds == all_labels).mean()
    ensemble_adjacent_acc = (np.abs(all_preds - all_labels) <= 1).mean()

    print(f"\nSingle model (seed {seeds[0]}) exact accuracy: {single_exact_acc*100:.2f}%  "
          f"(for reference, same test batches)")
    print(f"Ensemble exact-stage accuracy:    {ensemble_exact_acc*100:.2f}%")
    print(f"Ensemble adjacent-stage accuracy: {ensemble_adjacent_acc*100:.2f}%  (within +/-1 stage)")

    stage_names = [f"CS{i+1}" for i in range(NUM_CVM_STAGES)]
    print("\nPer-class report (ensemble):")
    print(classification_report(all_labels, all_preds, target_names=stage_names, zero_division=0))

    print("Confusion matrix (rows = true stage, columns = predicted stage):")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CVM_STAGES)))
    header = "        " + "  ".join(f"{n:>5}" for n in stage_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:5d}" for v in row)
        print(f"{stage_names[i]:>6}  {row_str}")


if __name__ == "__main__":
    main()