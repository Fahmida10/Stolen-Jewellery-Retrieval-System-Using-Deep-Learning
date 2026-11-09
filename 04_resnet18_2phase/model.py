"""
ResNet18 CNN model for jewelry classification.
Two-phase training: freeze then unfreeze backbone.
"""

import torch
import torch.nn as nn
from torchvision import models


def load_resnet18_backbone():
    """Load ImageNet-pretrained ResNet18, with offline fallback."""
    try:
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as exc:
        print(f"Warning: could not load pretrained ResNet18 weights ({exc}). Using random initialization.")
        return models.resnet18(weights=None)


class ResNet18Classifier(nn.Module):
    """ResNet18 with replaced classifier head for 4 jewelry categories."""

    def __init__(self, num_classes=4, freeze_backbone=True):
        super().__init__()
        resnet = load_resnet18_backbone()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

    def get_embeddings(self, x):
        features = self.backbone(x)
        return features.view(features.size(0), -1)
