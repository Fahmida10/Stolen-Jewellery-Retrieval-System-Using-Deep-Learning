"""
Model wrapper for OpenCLIP ViT-L/14 backbone
Optimized for jewelry image retrieval with L2-normalized embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import Optional


class JewelryRetrievalModel(nn.Module):
    """
    Jewelry retrieval model using OpenCLIP ViT-L/14 as backbone
    Outputs L2-normalized embeddings for cosine similarity
    """
    
    def __init__(self, 
                 model_name: str = "ViT-L-14",
                 pretrained: str = "openai",
                 embedding_dim: int = 768,
                 normalize_embeddings: bool = True,
                 freeze_backbone: bool = False):
        """
        Args:
            model_name: OpenCLIP model name (ViT-L-14 or ViT-H-14)
            pretrained: Pretrained weights to use
            embedding_dim: Output embedding dimension
            normalize_embeddings: Whether to L2-normalize embeddings
            freeze_backbone: Whether to freeze backbone (for fine-tuning only head)
        """
        super(JewelryRetrievalModel, self).__init__()
        
        # Load backbone: EVA-CLIP (if requested) or OpenCLIP by default
        if model_name.startswith("EVA-CLIP"):
            # EVA-CLIP branch (requires external eva_clip package)
            try:
                import eva_clip  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "EVA-CLIP model requested via MODEL_NAME starting with 'EVA-CLIP', "
                    "but the 'eva_clip' package is not installed. "
                    "Please install EVA-CLIP following its official instructions."
                ) from e
            
            print(f"Loading EVA-CLIP model: {model_name} with {pretrained} weights...")
            # NOTE: This assumes EVA-CLIP exposes a similar API to open_clip.
            # Adjust `create_model_and_transforms` call as needed for the actual repo.
            self.clip_model, self.preprocess = eva_clip.create_model_and_transforms(  # type: ignore[attr-defined]
                model_name, pretrained=pretrained
            )
        else:
            # OpenCLIP branch (default)
            print(f"Loading OpenCLIP model: {model_name} with {pretrained} weights...")
            try:
                # Fix QuickGELU mismatch: OpenAI weights use quick_gelu=True
                if pretrained == "openai" and model_name == "ViT-L-14":
                    self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained, quick_gelu=True
                    )
                else:
                    self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained
                    )
            except Exception as exc:
                print(f"Warning: could not load pretrained OpenCLIP weights ({exc}). Using random initialization.")
                self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=None
                )
        
        # Get the visual encoder only (we don't need text encoder)
        self.visual_encoder = self.clip_model.visual
        
        # Get the feature dimension from the model
        if hasattr(self.visual_encoder, 'output_dim'):
            feature_dim = self.visual_encoder.output_dim
        else:
            # For ViT models, the output dim is in the transformer
            feature_dim = self.visual_encoder.proj.shape[1] if hasattr(self.visual_encoder, 'proj') else 768
        
        print(f"Visual encoder output dimension: {feature_dim}")
        
        # Optional: Freeze backbone for fine-tuning only projection head
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        
        # Projection head (optional, to adjust embedding dimension)
        if feature_dim != embedding_dim:
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
            print(f"Added projection head: {feature_dim} -> {embedding_dim}")
        else:
            self.projection = nn.Identity()
            print(f"No projection needed, using {feature_dim}D embeddings directly")
        
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        
        print(f"Model initialized with {self.count_parameters():,} trainable parameters")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract embeddings
        
        Args:
            images: Input images [batch_size, 3, H, W]
        
        Returns:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
        """
        # Extract visual features from CLIP
        features = self.visual_encoder(images)
        
        # Apply projection head
        embeddings = self.projection(features)
        
        # L2 normalization for cosine similarity
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str, device: str = 'cuda'):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return self


class JewelryRetrievalModelWithHead(nn.Module):
    """
    Model with classification head for supervised contrastive pre-training
    Used during training, head is discarded during inference
    """
    
    def __init__(self, 
                 base_model: JewelryRetrievalModel,
                 num_classes: int = 4):
        """
        Args:
            base_model: Base retrieval model
            num_classes: Number of jewelry categories
        """
        super(JewelryRetrievalModelWithHead, self).__init__()
        
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.embedding_dim, num_classes)
        
        print(f"Added classification head: {base_model.embedding_dim} -> {num_classes} classes")
    
    def forward(self, images: torch.Tensor, return_embeddings: bool = False):
        """
        Forward pass with optional classification
        
        Args:
            images: Input images
            return_embeddings: If True, return both embeddings and logits
        
        Returns:
            embeddings (if return_embeddings=True) or logits or both
        """
        embeddings = self.base_model(images)
        
        if return_embeddings:
            logits = self.classifier(embeddings)
            return embeddings, logits
        
        return embeddings
    
    def get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get embeddings without classification"""
        return self.base_model(images)


def build_model(config, num_classes: int = 4, use_classifier: bool = False, freeze_backbone: bool = False):
    """
    Build jewelry retrieval model from config
    
    Args:
        config: Configuration object
        num_classes: Number of categories
        use_classifier: Whether to add classification head
        freeze_backbone: Whether to freeze the backbone (for head-only training)
    
    Returns:
        model: JewelryRetrievalModel or JewelryRetrievalModelWithHead
    """
    base_model = JewelryRetrievalModel(
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        embedding_dim=config.EMBEDDING_DIM,
        normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        freeze_backbone=freeze_backbone
    )
    
    if use_classifier:
        model = JewelryRetrievalModelWithHead(base_model, num_classes)
    else:
        model = base_model
    
    return model
