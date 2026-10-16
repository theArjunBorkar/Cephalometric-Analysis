"""
ResNet18-based 6-way CVM stage classifier, using ImageNet-pretrained weights.

Transfer learning is used deliberately here: Aariz has ~1000 images total,
which is small for training a CNN from scratch. Starting from ImageNet
features (edges, textures, shapes) gives the model a real head start even
though X-rays look nothing like natural photos -- low-level visual features
still transfer reasonably well.
"""
import torch.nn as nn
import torchvision.models as models


def build_cvm_classifier(num_classes=6, pretrained=True, freeze_backbone=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final classification layer (always trainable, even if the
    # rest of the backbone is frozen).
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    import torch
    m = build_cvm_classifier()
    x = torch.randn(2, 3, 384, 384)
    y = m(x)
    print("output shape:", y.shape)  # expect (2, 6)
