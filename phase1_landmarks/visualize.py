"""
Plots predicted vs ground-truth landmarks on a handful of sample images,
so you can visually sanity-check the model rather than just trusting metrics.

Usage:
  python visualize.py --data_root ./data --checkpoint checkpoints/best.pt \
      --test_dirs Test1Data,Test2Data --num_samples 6
"""
import os
import argparse
import random
import torch

from phase1_landmarks.dataset import CephDataset, build_file_list, make_train_val_split
from phase1_landmarks.model import UNetLandmark
from phase1_landmarks.utils import heatmaps_to_coords, visualize_prediction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    p.add_argument("--test_dirs", type=str, default=None,
                    help="Comma-separated RawImage subfolders (e.g. 'Test1Data,Test2Data'). "
                         "If omitted, samples from a random val split instead.")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--heatmap_size", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--output_dir", type=str, default="./viz_output")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_dirs:
        subdirs = [s.strip() for s in args.test_dirs.split(",") if s.strip()]
        entries = build_file_list(args.data_root, image_subdirs=subdirs)
    else:
        all_entries = build_file_list(args.data_root)
        _, entries = make_train_val_split(all_entries, args.val_fraction)

    random.seed(args.seed)
    sample_entries = random.sample(entries, min(args.num_samples, len(entries)))

    ds = CephDataset(sample_entries, args.image_size, args.heatmap_size, augment=False)

    model = UNetLandmark(num_landmarks=19).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    hm_to_input_scale = args.image_size / args.heatmap_size

    with torch.no_grad():
        for i in range(len(ds)):
            image, gt_heatmaps, meta = ds[i]
            image_b = image.unsqueeze(0).to(device)
            pred_heatmaps = model(image_b).cpu()

            pred_coords = heatmaps_to_coords(pred_heatmaps)[0] * hm_to_input_scale
            gt_coords = heatmaps_to_coords(gt_heatmaps.unsqueeze(0))[0] * hm_to_input_scale

            fname = os.path.basename(meta["image_path"])
            save_path = os.path.join(args.output_dir, f"pred_{i:02d}_{fname}.png")
            visualize_prediction(image, pred_coords, gt_coords, save_path=save_path)
            print(f"Saved {save_path}  (source: {fname})")

    print(f"\nDone. {len(ds)} visualizations written to {args.output_dir}/")
    print("Red = predicted, green X = ground truth. Look for landmarks where the "
          "red/green markers are far apart, or where predictions land on clearly "
          "wrong anatomy (e.g. off the skull entirely) -- that indicates a real "
          "failure mode rather than just normal sub-mm noise.")


if __name__ == "__main__":
    main()