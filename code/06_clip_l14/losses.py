"""
Loss functions for jewelry retrieval
- Supervised Contrastive Loss (SupCon)
- Triplet Loss with hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss from:
    "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    
    Best for learning discriminative embeddings when you have category labels
    Pulls together samples from same category, pushes apart different categories
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for scaling
            base_temperature: Base temperature for normalization
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss
        
        Args:
            features: Normalized embeddings [batch_size, embedding_dim]
            labels: Category labels [batch_size]
        
        Returns:
            loss: Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Create logits mask (exclude self-comparisons)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask
        
        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        # Handle case where a sample has no positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class TripletLossWithHardMining(nn.Module):
    """
    Triplet Loss with hard negative mining
    Useful when you have same-item multiple photos
    """
    
    def __init__(self, margin: float = 0.2, hard_mining: bool = True):
        """
        Args:
            margin: Margin for triplet loss
            hard_mining: Whether to use hard negative mining
        """
        super(TripletLossWithHardMining, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with hard negative mining
        
        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Category labels [batch_size]
        
        Returns:
            loss: Scalar loss value
        """
        # Compute pairwise distances
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Cosine distance = 1 - cosine similarity (for normalized vectors)
        dist_matrix = 1 - torch.matmul(embeddings, embeddings.T)
        
        # Create masks for positives and negatives
        labels = labels.unsqueeze(1)
        mask_positive = (labels == labels.T).float()
        mask_negative = (labels != labels.T).float()
        
        # Remove self-comparisons from positives
        mask_positive = mask_positive - torch.eye(len(labels), device=embeddings.device)
        
        if self.hard_mining:
            # Hard positive: farthest positive sample
            dist_ap = (dist_matrix * mask_positive).max(dim=1)[0]
            
            # Hard negative: closest negative sample
            # Add large value to positives so they're not selected
            dist_an = (dist_matrix + 1e6 * mask_positive).min(dim=1)[0]
        else:
            # Average over all positives and negatives
            dist_ap = (dist_matrix * mask_positive).sum(dim=1) / (mask_positive.sum(dim=1) + 1e-12)
            dist_an = (dist_matrix * mask_negative).sum(dim=1) / (mask_negative.sum(dim=1) + 1e-12)
        
        # Triplet loss
        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combination of Supervised Contrastive Loss and Cross-Entropy
    For even better performance
    """
    
    def __init__(self, temperature: float = 0.07, weight_supcon: float = 1.0, weight_ce: float = 0.1):
        """
        Args:
            temperature: Temperature for SupCon loss
            weight_supcon: Weight for supervised contrastive loss
            weight_ce: Weight for cross-entropy loss
        """
        super(CombinedLoss, self).__init__()
        self.supcon_loss = SupervisedContrastiveLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_supcon = weight_supcon
        self.weight_ce = weight_ce
    
    def forward(self, embeddings: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Compute combined loss
        
        Args:
            embeddings: Normalized embeddings
            logits: Classification logits
            labels: Ground truth labels
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        loss_supcon = self.supcon_loss(embeddings, labels)
        loss_ce = self.ce_loss(logits, labels)
        
        total_loss = self.weight_supcon * loss_supcon + self.weight_ce * loss_ce
        
        loss_dict = {
            'total': total_loss.item(),
            'supcon': loss_supcon.item(),
            'ce': loss_ce.item()
        }
        
        return total_loss, loss_dict


def get_loss_function(config, use_combined: bool = False):
    """
    Get loss function based on config
    
    Args:
        config: Configuration object
        use_combined: Whether to use combined loss (SupCon + CE)
    
    Returns:
        loss_fn: Loss function
    """
    if use_combined:
        return CombinedLoss(temperature=config.TEMPERATURE)
    elif config.LOSS_TYPE == "supcon":
        return SupervisedContrastiveLoss(temperature=config.TEMPERATURE)
    elif config.LOSS_TYPE == "triplet":
        return TripletLossWithHardMining(margin=config.MARGIN, hard_mining=config.HARD_MINING)
    else:
        raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")

