"""VGG16 CNN model for jewelry classification."""

import torch
import torch.nn as nn
from torchvision import models


def load_vgg16_backbone():
    """Load ImageNet-pretrained VGG16, with offline fallback."""
    try:
        return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    except Exception as exc:
        print(f"Warning: could not load pretrained VGG16 weights ({exc}). Using random initialization.")
        return models.vgg16(weights=None)


class VGG16Classifier(nn.Module):
    """
    VGG16 with replaced classifier head for 4 jewelry categories.
    """

    def __init__(self, num_classes=4):
        super().__init__()
        vgg = load_vgg16_backbone()
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_dim = 512 * 7 * 7
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

    def get_embeddings(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
