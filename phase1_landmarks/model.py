"""
Compact U-Net for heatmap-regression landmark detection.

Input:  (B, 1, H, W) grayscale ceph image
Output: (B, 19, H/4, W/4) heatmaps, one channel per landmark
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetLandmark(nn.Module):
    def __init__(self, num_landmarks=19, base_ch=32):
        super().__init__()
        c = base_ch

        self.enc1 = conv_block(1, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.enc4 = conv_block(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = conv_block(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)

        # One more level than a typical "stop early" heatmap head: this brings
        # output resolution to H/2 instead of H/4, halving the pixel grid that
        # coordinates get quantized to (on top of soft-argmax decoding in utils.py).
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)

        self.head = nn.Conv2d(c * 2, num_landmarks, 1)

    def forward(self, x):
        e1 = self.enc1(x)          # H
        e2 = self.enc2(self.pool(e1))    # H/2
        e3 = self.enc3(self.pool(e2))    # H/4
        e4 = self.enc4(self.pool(e3))    # H/8

        b = self.bottleneck(self.pool(e4))  # H/16

        d4 = self.up4(b)                 # H/8
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)                # H/4
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)                # H/2
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        out = self.head(d2)              # H/2, num_landmarks channels
        return out


if __name__ == "__main__":
    # quick shape sanity check
    m = UNetLandmark()
    x = torch.randn(2, 1, 512, 512)
    y = m(x)
    print("output shape:", y.shape)  # expect (2, 19, 256, 256)