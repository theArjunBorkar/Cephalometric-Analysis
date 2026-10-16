import numpy as np
import torch
import matplotlib.pyplot as plt


def heatmaps_to_coords(heatmaps, radius=2):
    """
    heatmaps: (B, K, H, W) tensor or array
    Returns: (B, K, 2) array of (x, y) coords in heatmap-pixel space, using a
             LOCAL SOFT-ARGMAX around each channel's peak for sub-pixel precision.

    Why: a hard argmax can only ever land on an integer pixel, so any two
    landmarks whose true locations are closer together than one heatmap pixel
    get rounded onto the same (or adjacent) cell -- this shows up as apparent
    "duplicate" predictions. Taking a weighted centroid over a small window
    around the peak recovers sub-pixel position and removes that artifact.

    radius: half-width (in heatmap pixels) of the window used for the
            weighted centroid. 2 -> a 5x5 window around the peak.
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    B, K, H, W = heatmaps.shape
    coords = np.zeros((B, K, 2), dtype=np.float32)

    for b in range(B):
        for k in range(K):
            hm = heatmaps[b, k]
            idx = np.argmax(hm)
            py, px = np.unravel_index(idx, (H, W))

            y0, y1 = max(0, py - radius), min(H, py + radius + 1)
            x0, x1 = max(0, px - radius), min(W, px + radius + 1)
            patch = hm[y0:y1, x0:x1]

            # Only positive contributions count towards the centroid --
            # negative/near-zero background values shouldn't pull the estimate.
            weights = np.clip(patch, a_min=0, a_max=None)
            total = weights.sum()

            if total <= 1e-8:
                # degenerate (flat/negative patch) -- fall back to the hard peak
                coords[b, k] = [px, py]
                continue

            ys = np.arange(y0, y1, dtype=np.float32)
            xs = np.arange(x0, x1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)

            cx = (weights * xx).sum() / total
            cy = (weights * yy).sum() / total
            coords[b, k] = [cx, cy]

    return coords


def visualize_prediction(image_tensor, pred_coords_input_space, gt_coords_input_space=None, save_path=None, show_indices=True):
    """
    image_tensor: (1, H, W) tensor in [0,1]
    pred_coords_input_space: (K, 2) array, already scaled to the image's own resolution
    gt_coords_input_space: optional (K, 2) array, same scale
    show_indices: if True, labels each ground-truth point with its landmark index
                  (0-18), so you can identify which anatomical point a given
                  index corresponds to just by looking at the image.
    """
    img = image_tensor.squeeze(0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray")
    plt.scatter(pred_coords_input_space[:, 0], pred_coords_input_space[:, 1],
                c="red", s=15, label="predicted")
    if gt_coords_input_space is not None:
        plt.scatter(gt_coords_input_space[:, 0], gt_coords_input_space[:, 1],
                     c="lime", s=15, marker="x", label="ground truth")
        if show_indices:
            for i, (x, y) in enumerate(gt_coords_input_space):
                plt.annotate(str(i), (x, y), color="yellow", fontsize=8,
                             xytext=(4, 4), textcoords="offset points")
    plt.legend()
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()