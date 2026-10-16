"""
Sanity-check tool: draws the proposed cervical-region crop box (and the
Sella/Nasion/Articulare anchor points) on top of full cephalograms, so you can
visually confirm it actually lands on the C2-C4 vertebrae BEFORE plugging it
into training. This heuristic was never validated against ground-truth
vertebra boxes -- always eyeball it first.

Usage:
  python visualize_crop.py --data_root ./Aariz --split test --num_samples 8
"""
import os
import argparse
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from crop_utils import load_landmarks, compute_cervical_crop_box, IDX_SELLA, IDX_NASION, IDX_ARTICULARE
from dataset import _resolve_split_dir, CEPHALOGRAMS_DIRNAME, IMAGE_EXTENSIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="./crop_viz_output")
    p.add_argument("--width_scale", type=float, default=1.4)
    p.add_argument("--height_scale", type=float, default=2.2)
    p.add_argument("--x_offset_scale", type=float, default=-0.35)
    p.add_argument("--y_offset_scale", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    split_name = _resolve_split_dir(args.data_root, args.split)
    images_root = os.path.join(args.data_root, split_name, CEPHALOGRAMS_DIRNAME)
    all_files = sorted(f for f in os.listdir(images_root) if f.lower().endswith(IMAGE_EXTENSIONS))

    random.seed(args.seed)
    sample_files = random.sample(all_files, min(args.num_samples, len(all_files)))

    for i, fname in enumerate(sample_files):
        img_path = os.path.join(images_root, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        landmarks = load_landmarks(args.data_root, split_name, fname)
        box = compute_cervical_crop_box(
            landmarks,
            width_scale=args.width_scale,
            height_scale=args.height_scale,
            x_offset_scale=args.x_offset_scale,
            y_offset_scale=args.y_offset_scale,
        )
        x0, y0, x1, y1 = box

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2,
                                  edgecolor="lime", facecolor="none", label="proposed crop")
        ax.add_patch(rect)

        for idx, name, color in [(IDX_SELLA, "Sella", "cyan"),
                                   (IDX_NASION, "Nasion", "yellow"),
                                   (IDX_ARTICULARE, "Articulare", "red")]:
            x, y = landmarks[idx]
            ax.scatter([x], [y], c=color, s=40)
            ax.annotate(name, (x, y), color=color, fontsize=9, xytext=(5, 5), textcoords="offset points")

        ax.set_title(fname)
        ax.axis("off")
        save_path = os.path.join(args.output_dir, f"crop_{i:02d}_{fname}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved {save_path}")

    print(f"\nDone. Check {args.output_dir}/ -- the green box should tightly "
          f"contain C2-C4 (the top few visible neck vertebrae). If it's "
          f"missing them, off to one side, or including too much jaw/skull, "
          f"adjust --width_scale / --height_scale / --x_offset_scale / "
          f"--y_offset_scale and re-run until it looks right across several "
          f"samples -- these constants were a starting-point estimate, not "
          f"a validated fit.")


if __name__ == "__main__":
    main()
