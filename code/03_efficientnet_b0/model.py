"""
EfficientNet-B0 CNN model for jewelry classification.
Single-phase training: full model trained end-to-end.
"""

import torch
import torch.nn as nn
from torchvision import models


def load_efficientnet_b0_backbone():
    """Load ImageNet-pretrained EfficientNet-B0, with offline fallback."""
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception as exc:
        print(f"Warning: could not load pretrained EfficientNet-B0 weights ({exc}). Using random initialization.")
        return models.efficientnet_b0(weights=None)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 with replaced classifier head for 4 jewelry categories.
    """

    def __init__(self, num_classes=4):
        super().__init__()
        backbone = load_efficientnet_b0_backbone()
        self.backbone = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = backbone.classifier[1].in_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

    def get_embeddings(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        return features.view(features.size(0), -1)
