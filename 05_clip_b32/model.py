"""OpenCLIP ViT-B/32 model for jewelry classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class JewelryRetrievalModel(nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="openai", embedding_dim=512, normalize_embeddings=True, freeze_backbone=False):
        super().__init__()
        print(f"Loading OpenCLIP {model_name} with {pretrained} weights...")
        try:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        except Exception as exc:
            print(f"Warning: could not load pretrained OpenCLIP weights ({exc}). Using random initialization.")
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
        self.visual_encoder = self.clip_model.visual
        feature_dim = self.visual_encoder.proj.shape[1] if hasattr(self.visual_encoder, 'proj') else 512
        if feature_dim != embedding_dim:
            self.projection = nn.Sequential(nn.Linear(feature_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        else:
            self.projection = nn.Identity()
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        if freeze_backbone:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.visual_encoder(x)
        embeddings = self.projection(features)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class JewelryRetrievalModelWithHead(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.embedding_dim, num_classes)

    def forward(self, x, return_embeddings=False):
        embeddings = self.base_model(x)
        if return_embeddings:
            return embeddings, self.classifier(embeddings)
        return self.classifier(embeddings)

    def get_embeddings(self, x):
        return self.base_model(x)


def build_model(config, num_classes=4, use_classifier=False, freeze_backbone=False):
    base = JewelryRetrievalModel(
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        embedding_dim=config.EMBEDDING_DIM,
        normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        freeze_backbone=freeze_backbone
    )
    return JewelryRetrievalModelWithHead(base, num_classes) if use_classifier else base
