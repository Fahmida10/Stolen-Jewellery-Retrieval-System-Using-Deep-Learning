"""
Dataset loader for jewelry images with high-resolution transforms
Optimized for fine-grained visual details
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional


class JewelryDataset(Dataset):
    """Dataset class for jewelry images with category labels"""
    
    def __init__(self, root_dir: str, transform=None, return_path: bool = False):
        """
        Args:
            root_dir: Root directory containing category folders
            transform: Optional transform to be applied on images
            return_path: If True, return image path along with image and label
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        
        # Build dataset: list of (image_path, category_idx)
        self.samples = []
        self.categories = sorted(os.listdir(root_dir))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        for category in self.categories:
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(category_path, img_name)
                    self.samples.append((img_path, self.category_to_idx[category]))
        
        print(f"Loaded {len(self.samples)} images from {len(self.categories)} categories")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        return image, label


class StolenJewelryDataset(Dataset):
    """Dataset class for stolen jewelry database (no labels needed)"""
    
    def __init__(self, stolen_dir: str, transform=None):
        """
        Args:
            stolen_dir: Directory containing stolen jewelry images
            transform: Transform to be applied on images
        """
        self.stolen_dir = stolen_dir
        self.transform = transform
        
        # List all images
        self.image_paths = []
        for img_name in os.listdir(stolen_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(stolen_dir, img_name))
        
        print(f"Loaded {len(self.image_paths)} stolen jewelry images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path


def get_transforms(input_size: int = 336, is_train: bool = True, use_augmentation: bool = True):
    """
    Get image transforms optimized for jewelry images
    Higher resolution (336 or 384) to capture fine details
    
    Args:
        input_size: Target image size (336 or 384 recommended)
        is_train: Whether this is for training (with augmentation)
        use_augmentation: Whether to use data augmentation
    """
    
    # CLIP normalization (OpenCLIP uses same stats as original CLIP)
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    if is_train and use_augmentation:
        # Training transforms with augmentation for better generalization
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    return transform


def get_dataloader(root_dir: str, 
                   input_size: int,
                   batch_size: int,
                   is_train: bool = True,
                   use_augmentation: bool = True,
                   num_workers: int = 4,
                   pin_memory: bool = True,
                   return_path: bool = False):
    """
    Create DataLoader for jewelry dataset
    
    Args:
        root_dir: Root directory containing category folders
        input_size: Input image size
        batch_size: Batch size
        is_train: Whether this is training data
        use_augmentation: Whether to use augmentation
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        return_path: Whether to return image paths
    """
    transform = get_transforms(input_size, is_train, use_augmentation)
    dataset = JewelryDataset(root_dir, transform=transform, return_path=return_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train  # Drop last incomplete batch in training
    )
    
    return dataloader, dataset


def get_stolen_dataloader(stolen_dir: str,
                          input_size: int,
                          batch_size: int = 32,
                          num_workers: int = 4):
    """Create DataLoader for stolen jewelry database"""
    transform = get_transforms(input_size, is_train=False, use_augmentation=False)
    dataset = StolenJewelryDataset(stolen_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset

