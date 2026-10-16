import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AarizCVMDataset, NUM_CVM_STAGES
from model import build_cvm_classifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--seed", type=int, default=0,
                    help="Random seed for model init and data shuffling. Train several "
                         "runs with different seeds to build an ensemble -- checkpoints "
                         "are saved per-seed so they don't overwrite each other.")
    p.add_argument("--crop_to_cervical", action="store_true",
                    help="Crop each image to the C2-C4 cervical region (anchored on the "
                         "Articulare landmark) before classifying, instead of using the "
                         "whole cephalogram. Sanity-check with visualize_crop.py first.")
    p.add_argument("--use_clahe", action="store_true",
                    help="Apply CLAHE contrast enhancement after cropping, before resizing. "
                         "Cheap preprocessing step that can make low-contrast vertebral "
                         "edges easier for the network to pick up.")
    p.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze the pretrained backbone and only train the final layer. "
                         "Faster and less prone to overfitting on a small dataset, at the "
                         "cost of some accuracy -- worth trying if the full-finetune run overfits.")
    p.add_argument("--disable_class_weights", action="store_true",
                    help="Disable class-weighted loss and use plain cross-entropy instead. "
                         "Useful for comparing against the class-weighted run to see how much "
                         "the weighting actually helped on minority stages (CS1-CS3).")
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
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  seed: {args.seed}")

    train_ds = AarizCVMDataset(args.data_root, "train", args.image_size, augment=True,
                                crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)
    valid_ds = AarizCVMDataset(args.data_root, "valid", args.image_size, augment=False,
                                crop_to_cervical=args.crop_to_cervical, use_clahe=args.use_clahe)

    print(f"Train: {len(train_ds)} images  Valid: {len(valid_ds)} images")
    print("Train class distribution (stage -> count):", train_ds.class_distribution())
    print("Valid class distribution (stage -> count):", valid_ds.class_distribution())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    model = build_cvm_classifier(num_classes=NUM_CVM_STAGES,
                                  pretrained=True,
                                  freeze_backbone=args.freeze_backbone).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("Class weighting disabled -- using plain cross-entropy.")
    else:
        train_counts = train_ds.class_distribution()  # {1: n1, ..., 6: n6}
        counts_by_class = [train_counts[i] for i in range(1, NUM_CVM_STAGES + 1)]
        total = sum(counts_by_class)
        # Inverse-frequency weighting (same scheme as sklearn's class_weight="balanced"):
        # rarer classes (CS1-CS3 here) get proportionally larger weight in the loss,
        # so the model can't just ignore them and still get a low average loss.
        class_weights = [total / (NUM_CVM_STAGES * c) for c in counts_by_class]
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print("Class weights (CS1..CS6):", [f"{w:.2f}" for w in class_weights])
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = run_epoch(model, valid_loader, device, criterion=criterion)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"last_seed{args.seed}.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_seed{args.seed}.pt"))
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()