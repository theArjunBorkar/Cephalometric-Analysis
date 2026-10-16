import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1_landmarks.dataset import CephDataset, build_file_list, make_train_val_split
from phase1_landmarks.model import UNetLandmark


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--heatmap_size", type=int, default=256)
    p.add_argument("--sigma", type=float, default=6.0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    entries = build_file_list(args.data_root)
    print(f"Found {len(entries)} labeled images.")
    train_entries, val_entries = make_train_val_split(entries, args.val_fraction)
    print(f"Train: {len(train_entries)}  Val: {len(val_entries)}")

    train_ds = CephDataset(train_entries, args.image_size, args.heatmap_size, args.sigma, augment=True)
    val_ds = CephDataset(val_entries, args.image_size, args.heatmap_size, args.sigma, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = UNetLandmark(num_landmarks=19).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, heatmaps, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images, heatmaps = images.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, heatmaps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps, _ in val_loader:
                images, heatmaps = images.to(device), heatmaps.to(device)
                preds = model(images)
                loss = criterion(preds, heatmaps)
                val_loss += loss.item() * images.size(0)
        val_loss /= max(len(val_ds), 1)

        scheduler.step(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.6f})")

    print("Training complete.")


if __name__ == "__main__":
    main()