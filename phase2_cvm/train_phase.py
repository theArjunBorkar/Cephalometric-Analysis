"""
Trains the coarse 3-way phase classifier (Early=CS1-2, Peak=CS3-4, Late=CS5-6).
This is the first stage of the hierarchical cascade -- see hierarchical.py
for the phase grouping definition, and evaluate_hierarchical.py for how this
combines with the binary specialists at inference time.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AarizCVMDataset
from hierarchical import PhaseDataset, PHASE_NAMES
from model import build_cvm_classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--crop_to_cervical", action="store_true")
    p.add_argument("--use_clahe", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--disable_class_weights", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def run_epoch(model, loader, device, optimizer=None, criterion=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, labels, _ in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_train = AarizCVMDataset(args.data_root, "train", args.image_size, augment=True,
                                  crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)
    base_valid = AarizCVMDataset(args.data_root, "valid", args.image_size, augment=False,
                                  crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)

    train_ds = PhaseDataset(base_train, mode="phase")
    valid_ds = PhaseDataset(base_valid, mode="phase")

    print(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}")
    train_dist = train_ds.class_distribution()
    print("Train phase distribution:", {PHASE_NAMES[k]: v for k, v in sorted(train_dist.items())})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    model = build_cvm_classifier(num_classes=3, pretrained=True,
                                  freeze_backbone=args.freeze_backbone).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        counts = [train_dist.get(i, 0) for i in range(3)]
        total = sum(counts)
        weights = [total / (3 * c) if c > 0 else 0.0 for c in counts]
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        print("Phase class weights:", {PHASE_NAMES[i]: f"{w:.2f}" for i, w in enumerate(weights)})
        criterion = nn.CrossEntropyLoss(weight=weights_t)

    best_val_acc = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, "phase_classifier_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = run_epoch(model, valid_loader, device, criterion=criterion)
        scheduler.step(val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

    print(f"\nPhase classifier training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
