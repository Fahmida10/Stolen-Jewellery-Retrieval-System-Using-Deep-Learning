"""
Utility functions for training and evaluation
"""

import os
import torch
import numpy as np
import random
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, loss, filepath, is_best=False):
    """Save model checkpoint"""
    
    # Handle DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)
    print(f"{'Best model' if is_best else 'Checkpoint'} saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Handle DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch, loss


def compute_cosine_similarity(embeddings1, embeddings2):
    """
    Compute cosine similarity between two sets of embeddings
    
    Args:
        embeddings1: [N, D] tensor
        embeddings2: [M, D] tensor
    
    Returns:
        similarity_matrix: [N, M] tensor of cosine similarities
    """
    # Normalize embeddings
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.matmul(embeddings1, embeddings2.T)
    
    return similarity


def compute_euclidean_distance(embeddings1, embeddings2):
    """
    Compute Euclidean distance between two sets of embeddings
    
    Args:
        embeddings1: [N, D] tensor
        embeddings2: [M, D] tensor
    
    Returns:
        distance_matrix: [N, M] tensor of Euclidean distances
    """
    # Expand dimensions for broadcasting
    embeddings1_expanded = embeddings1.unsqueeze(1)  # [N, 1, D]
    embeddings2_expanded = embeddings2.unsqueeze(0)  # [1, M, D]
    
    # Compute L2 distance
    distances = torch.norm(embeddings1_expanded - embeddings2_expanded, p=2, dim=2)
    
    return distances


def get_top_k_matches(query_embedding, database_embeddings, k=10, metric='cosine'):
    """
    Get top-k most similar items from database
    
    Args:
        query_embedding: [1, D] or [D] tensor
        database_embeddings: [N, D] tensor
        k: Number of top matches to return
        metric: 'cosine' or 'euclidean'
    
    Returns:
        top_k_indices: Indices of top-k matches
        top_k_scores: Similarity/distance scores
    """
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
    
    if metric == 'cosine':
        # Higher is better for cosine similarity
        scores = compute_cosine_similarity(query_embedding, database_embeddings)
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, len(database_embeddings)), dim=1)
    elif metric == 'euclidean':
        # Lower is better for Euclidean distance
        distances = compute_euclidean_distance(query_embedding, database_embeddings)
        top_k_scores, top_k_indices = torch.topk(distances, k=min(k, len(database_embeddings)), 
                                                   dim=1, largest=False)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return top_k_indices[0], top_k_scores[0]


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

