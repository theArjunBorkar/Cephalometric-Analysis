"""
Trains a binary specialist for ONE phase -- e.g. for phase=2 (Late), this
learns to distinguish CS5 vs CS6 among images already known to be in the
Late phase. Run this three times (--phase 0, 1, 2) to get all three
specialists needed for the full cascade.

Given your confusion matrix so far (errors concentrated at CS4/CS5/CS6), the
Late-phase specialist (--phase 2, CS5 vs CS6) is doing the most important
job in the cascade -- worth checking its standalone accuracy first if you
want to gauge whether the hierarchical approach is paying off before
training all three.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AarizCVMDataset
from hierarchical import PhaseDataset, PHASE_NAMES, STAGES_IN_PHASE
from model import build_cvm_classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--phase", type=int, required=True, choices=[0, 1, 2],
                    help="Which phase's binary specialist to train: 0=Early (CS1 vs CS2), "
                         "1=Peak (CS3 vs CS4), 2=Late (CS5 vs CS6).")
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

    stage_names = [f"CS{s+1}" for s in STAGES_IN_PHASE[args.phase]]
    print(f"Training binary specialist for phase={args.phase} "
          f"({PHASE_NAMES[args.phase]}: {stage_names[0]} vs {stage_names[1]})")
    print(f"Using device: {device}")

    base_train = AarizCVMDataset(args.data_root, "train", args.image_size, augment=True,
                                  crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)
    base_valid = AarizCVMDataset(args.data_root, "valid", args.image_size, augment=False,
                                  crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)

    train_ds = PhaseDataset(base_train, mode="binary", phase=args.phase)
    valid_ds = PhaseDataset(base_valid, mode="binary", phase=args.phase)

    print(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}")
    train_dist = train_ds.class_distribution()
    print(f"Train distribution ({stage_names[0]}=0, {stage_names[1]}=1):", train_dist)

    if len(valid_ds) == 0:
        print("WARNING: validation set for this phase is empty (too few images in this "
              "phase's valid split) -- checkpointing on train accuracy instead.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True) if len(valid_ds) > 0 else None

    model = build_cvm_classifier(num_classes=2, pretrained=True,
                                  freeze_backbone=args.freeze_backbone).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        counts = [train_dist.get(i, 0) for i in range(2)]
        total = sum(counts)
        weights = [total / (2 * c) if c > 0 else 0.0 for c in counts]
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Class weights ({stage_names[0]}, {stage_names[1]}):", [f"{w:.2f}" for w in weights])
        criterion = nn.CrossEntropyLoss(weight=weights_t)

    best_val_acc = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, f"binary_phase{args.phase}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, criterion)
        if valid_loader is not None:
            val_loss, val_acc = run_epoch(model, valid_loader, device, criterion=criterion)
        else:
            val_loss, val_acc = train_loss, train_acc
        scheduler.step(val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

    print(f"\nBinary specialist (phase={args.phase}) training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
