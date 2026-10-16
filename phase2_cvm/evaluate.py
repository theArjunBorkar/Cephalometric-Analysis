"""
Evaluates a trained CVM classifier on the official Aariz Test split.

Reports:
  - Overall (exact-stage) accuracy
  - Per-class precision/recall/F1
  - Confusion matrix
  - Adjacent-stage accuracy (prediction within +/-1 stage of ground truth) --
    clinically relevant since CVM staging has real inter-orthodontist
    disagreement, and being one stage off is a much smaller error than being
    three stages off.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset import AarizCVMDataset, NUM_CVM_STAGES
from model import build_cvm_classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--crop_to_cervical", action="store_true",
                    help="Must match whatever the checkpoint was trained with.")
    p.add_argument("--use_clahe", action="store_true",
                    help="Must match whatever the checkpoint was trained with.")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = AarizCVMDataset(args.data_root, "test", args.image_size, augment=False,
                               crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)
    print(f"Test set: {len(test_ds)} images")
    print("Test class distribution (stage -> count):", test_ds.class_distribution())

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_cvm_classifier(num_classes=NUM_CVM_STAGES, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    exact_acc = (all_preds == all_labels).mean()
    adjacent_acc = (np.abs(all_preds - all_labels) <= 1).mean()

    print(f"\nExact-stage accuracy:    {exact_acc*100:.2f}%")
    print(f"Adjacent-stage accuracy: {adjacent_acc*100:.2f}%  (prediction within +/-1 stage)")

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