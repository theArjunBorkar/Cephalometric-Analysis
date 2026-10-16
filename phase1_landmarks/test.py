import torch
from phase1_landmarks.dataset import CephDataset, build_file_list
from phase1_landmarks.model import UNetLandmark
from phase1_landmarks.utils import heatmaps_to_coords

entries = build_file_list("./data", image_subdirs=["Test1Data", "Test2Data"])
ds = CephDataset(entries[:1], 512, 128, augment=False)
image, gt_heatmaps, meta = ds[0]

model = UNetLandmark(num_landmarks=19)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    pred = model(image.unsqueeze(0))
coords = heatmaps_to_coords(pred)[0] * (512/128)
for i, (x, y) in enumerate(coords):
    print(f"landmark {i}: ({x:.1f}, {y:.1f})")