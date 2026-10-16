"""
Evaluates a trained model using the same metrics as the ISBI 2015 challenge
papers, so you can compare directly against published numbers:

  - MRE (Mean Radial Error), in mm
  - SDR (Successful Detection Rate) at 2.0mm / 2.5mm / 3.0mm / 4.0mm

Note: the standard ISBI ceph images are captured at ~0.1mm/pixel. If your
copy of the dataset has different image dimensions, double check this value
(some redistributions resize the raw images) -- pass --mm_per_pixel to override.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from phase1_landmarks.dataset import CephDataset, build_file_list, make_train_val_split
from phase1_landmarks.model import UNetLandmark
from phase1_landmarks.utils import heatmaps_to_coords


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--heatmap_size", type=int, default=256)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--test_dirs", type=str, default=None,
                    help="Comma-separated RawImage subfolder names to evaluate on directly "
                         "(e.g. 'Test1Data,Test2Data' for the official ISBI held-out sets). "
                         "When set, uses ALL images in these folders as the eval set and "
                         "ignores --val_fraction.")
    p.add_argument("--mm_per_pixel", type=float, default=0.1,
                    help="Physical pixel spacing of the RAW (original) images. "
                         "Standard ISBI ceph scans are 0.1mm/pixel.")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_dirs:
        subdirs = [s.strip() for s in args.test_dirs.split(",") if s.strip()]
        eval_entries = build_file_list(args.data_root, image_subdirs=subdirs)
        print(f"Evaluating on official test folder(s) {subdirs}: {len(eval_entries)} images.")
    else:
        entries = build_file_list(args.data_root)
        _, eval_entries = make_train_val_split(entries, args.val_fraction)
        print(f"Evaluating on random val split: {len(eval_entries)} images.")

    ds = CephDataset(eval_entries, args.image_size, args.heatmap_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNetLandmark(num_landmarks=19).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_errors_mm = []  # will hold per-landmark errors, shape accumulated as (N, 19)

    hm_to_input_scale = args.image_size / args.heatmap_size

    with torch.no_grad():
        for images, heatmaps, metas in loader:
            images = images.to(device)
            preds = model(images)
            pred_coords_hm = heatmaps_to_coords(preds)  # (B, 19, 2) in heatmap space

            for i in range(images.size(0)):
                scale_x, scale_y = metas["scale_to_input"][0][i].item(), metas["scale_to_input"][1][i].item()
                gt_orig = metas["landmarks_orig_px"][i].numpy()  # (19, 2) original pixel space

                pred_input = pred_coords_hm[i] * hm_to_input_scale  # back to 512x512 space
                pred_orig = pred_input.copy()
                pred_orig[:, 0] /= scale_x
                pred_orig[:, 1] /= scale_y

                err_px = np.linalg.norm(pred_orig - gt_orig, axis=1)  # (19,)
                err_mm = err_px * args.mm_per_pixel
                all_errors_mm.append(err_mm)

    all_errors_mm = np.stack(all_errors_mm, axis=0)  # (N, 19)

    mre_per_landmark = all_errors_mm.mean(axis=0)
    overall_mre = all_errors_mm.mean()

    print(f"\nOverall MRE: {overall_mre:.3f} mm\n")
    print("Per-landmark MRE (mm):")
    for i, v in enumerate(mre_per_landmark):
        print(f"  landmark {i:2d}: {v:.3f}")

    print("\nSDR (Successful Detection Rate):")
    for thresh in [2.0, 2.5, 3.0, 4.0]:
        sdr = (all_errors_mm <= thresh).mean() * 100
        print(f"  <= {thresh}mm: {sdr:.2f}%")


if __name__ == "__main__":
    main()