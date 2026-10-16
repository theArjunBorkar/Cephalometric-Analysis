"""
Evaluates the full hierarchical cascade on the official Aariz Test split:
  1. Phase classifier predicts Early/Peak/Late.
  2. That phase's binary specialist picks the exact stage within it.
Reports the same metrics as evaluate.py (exact accuracy, adjacent accuracy,
per-class report, confusion matrix) so you can directly compare against the
flat single-model results.

Requires all four checkpoints to already exist:
  checkpoints/phase_classifier_best.pt
  checkpoints/binary_phase0_best.pt  (Early: CS1 vs CS2)
  checkpoints/binary_phase1_best.pt  (Peak: CS3 vs CS4)
  checkpoints/binary_phase2_best.pt  (Late: CS5 vs CS6)
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset import AarizCVMDataset, NUM_CVM_STAGES
from hierarchical import PHASE_NAMES, STAGES_IN_PHASE
from model import build_cvm_classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--crop_to_cervical", action="store_true",
                    help="Must match whatever all four checkpoints were trained with.")
    p.add_argument("--use_clahe", action="store_true",
                    help="Must match whatever all four checkpoints were trained with.")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = AarizCVMDataset(args.data_root, "test", args.image_size, augment=False,
                               crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)
    print(f"Test set: {len(test_ds)} images")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    phase_model = build_cvm_classifier(num_classes=3, pretrained=False).to(device)
    phase_model.load_state_dict(torch.load(
        os.path.join(args.checkpoint_dir, "phase_classifier_best.pt"), map_location=device))
    phase_model.eval()

    binary_models = {}
    for phase in range(3):
        m = build_cvm_classifier(num_classes=2, pretrained=False).to(device)
        ckpt_path = os.path.join(args.checkpoint_dir, f"binary_phase{phase}_best.pt")
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        binary_models[phase] = m

    all_preds, all_labels = [], []
    correct_phase_count = 0

    with torch.no_grad():
        for images, stage_labels, _ in loader:
            images = images.to(device)
            batch_size = images.size(0)

            phase_logits = phase_model(images)
            pred_phases = phase_logits.argmax(dim=1).cpu().numpy()

            true_phases = np.array([
                0 if s.item() in STAGES_IN_PHASE[0] else (1 if s.item() in STAGES_IN_PHASE[1] else 2)
                for s in stage_labels
            ])
            correct_phase_count += (pred_phases == true_phases).sum()

            # Run each sample through ITS PREDICTED phase's specialist -- this
            # is the real cascade behavior (errors in phase prediction propagate),
            # not an oracle shortcut using the true phase.
            final_stage_preds = np.zeros(batch_size, dtype=np.int64)
            for phase in range(3):
                mask = pred_phases == phase
                if not mask.any():
                    continue
                idxs = np.where(mask)[0]
                sub_images = images[idxs]
                binary_logits = binary_models[phase](sub_images)
                binary_preds = binary_logits.argmax(dim=1).cpu().numpy()
                stage_options = STAGES_IN_PHASE[phase]
                for local_i, global_i in enumerate(idxs):
                    final_stage_preds[global_i] = stage_options[binary_preds[local_i]]

            all_preds.extend(final_stage_preds.tolist())
            all_labels.extend(stage_labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    phase_acc = correct_phase_count / len(test_ds)
    print(f"\nStage-1 phase classifier accuracy: {phase_acc*100:.2f}%  "
          f"(errors here propagate into the final stage prediction)")

    exact_acc = (all_preds == all_labels).mean()
    adjacent_acc = (np.abs(all_preds - all_labels) <= 1).mean()

    print(f"\nFinal cascade exact-stage accuracy:    {exact_acc*100:.2f}%")
    print(f"Final cascade adjacent-stage accuracy: {adjacent_acc*100:.2f}%  (within +/-1 stage)")

    stage_names = [f"CS{i+1}" for i in range(NUM_CVM_STAGES)]
    print("\nPer-class report:")
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
