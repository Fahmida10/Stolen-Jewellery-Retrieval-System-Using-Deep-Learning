"""
Extract embeddings from trained model for all datasets
Creates embedding database for retrieval
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import pickle

from config import Config
from dataset import get_dataloader, get_stolen_dataloader
from model import build_model
from utils import load_checkpoint


def extract_embeddings(model, dataloader, device, desc="Extracting embeddings"):
    """
    Extract embeddings from a dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device to run on
        desc: Description for progress bar
    
    Returns:
        embeddings: numpy array of embeddings [N, D]
        labels: numpy array of labels [N] (if available)
        paths: list of image paths [N]
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch[0].to(device)
            
            # Handle different batch formats
            if len(batch) == 3:
                # Dataset with return_path=True: (images, labels, paths)
                labels = batch[1]
                paths = batch[2]
            elif len(batch) == 2:
                # Check if second element is labels (tensor) or paths (list of strings)
                if isinstance(batch[1], (list, tuple)) and len(batch[1]) > 0 and isinstance(batch[1][0], str):
                    # Stolen dataset: (images, paths)
                    paths = batch[1]
                    labels = None
                else:
                    # Regular dataset: (images, labels)
                    labels = batch[1]
                    paths = None
            else:
                labels = None
                paths = None
            
            # Extract embeddings
            if isinstance(model, torch.nn.DataParallel):
                if hasattr(model.module, 'base_model'):
                    embeddings = model.module.base_model(images)
                elif hasattr(model.module, 'get_embeddings'):
                    embeddings = model.module.get_embeddings(images)
                else:
                    embeddings = model.module(images)
            else:
                if hasattr(model, 'base_model'):
                    embeddings = model.base_model(images)
                elif hasattr(model, 'get_embeddings'):
                    embeddings = model.get_embeddings(images)
                else:
                    embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            if labels is not None:
                all_labels.append(labels.cpu().numpy())
            
            if paths is not None:
                all_paths.extend(paths)
    
    # Concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    if all_labels:
        labels = np.concatenate(all_labels, axis=0)
    else:
        labels = None
    
    print(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    return embeddings, labels, all_paths


def save_embeddings(embeddings, labels, paths, save_path, metadata=None):
    """
    Save embeddings to disk
    
    Args:
        embeddings: numpy array of embeddings
        labels: numpy array of labels (or None)
        paths: list of image paths
        save_path: Path to save file
        metadata: Optional metadata dictionary
    """
    data = {
        'embeddings': embeddings,
        'labels': labels,
        'paths': paths,
        'metadata': metadata or {}
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Embeddings saved to {save_path}")


def load_embeddings(load_path):
    """
    Load embeddings from disk
    
    Args:
        load_path: Path to load from
    
    Returns:
        embeddings, labels, paths, metadata
    """
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['embeddings'], data['labels'], data['paths'], data['metadata']


def main():
    """Main function to extract embeddings for all datasets"""
    config = Config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)
    
    # Load trained model
    print("\nLoading trained model...")
    model_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_MODEL_NAME)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Build model
    model = build_model(config, num_classes=4, use_classifier=True)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")
    
    # Extract embeddings for stolen jewelry database
    print("\n" + "="*80)
    print("Extracting embeddings for STOLEN JEWELRY DATABASE")
    print("="*80)
    
    stolen_loader, stolen_dataset = get_stolen_dataloader(
        config.STOLEN_DIR,
        config.INPUT_SIZE,
        batch_size=32,
        num_workers=config.NUM_WORKERS
    )
    
    stolen_embeddings, _, stolen_paths = extract_embeddings(
        model, stolen_loader, device, desc="Stolen Database"
    )
    
    save_embeddings(
        stolen_embeddings,
        None,
        stolen_paths,
        os.path.join(config.EMBEDDING_DIR, 'stolen_embeddings.pkl'),
        metadata={'dataset': 'stolen', 'num_images': len(stolen_paths)}
    )
    
    # Extract embeddings for test set
    print("\n" + "="*80)
    print("Extracting embeddings for TEST SET")
    print("="*80)
    
    test_loader, test_dataset = get_dataloader(
        config.TEST_DIR,
        config.INPUT_SIZE,
        batch_size=32,
        is_train=False,
        use_augmentation=False,
        num_workers=config.NUM_WORKERS,
        return_path=True
    )
    
    test_embeddings, test_labels, test_paths = extract_embeddings(
        model, test_loader, device, desc="Test Set"
    )
    
    save_embeddings(
        test_embeddings,
        test_labels,
        test_paths,
        os.path.join(config.EMBEDDING_DIR, 'test_embeddings.pkl'),
        metadata={
            'dataset': 'test',
            'num_images': len(test_paths),
            'categories': test_dataset.categories
        }
    )
    
    # Extract embeddings for validation set
    print("\n" + "="*80)
    print("Extracting embeddings for VALIDATION SET")
    print("="*80)
    
    val_loader, val_dataset = get_dataloader(
        config.VAL_DIR,
        config.INPUT_SIZE,
        batch_size=32,
        is_train=False,
        use_augmentation=False,
        num_workers=config.NUM_WORKERS,
        return_path=True
    )
    
    val_embeddings, val_labels, val_paths = extract_embeddings(
        model, val_loader, device, desc="Validation Set"
    )
    
    save_embeddings(
        val_embeddings,
        val_labels,
        val_paths,
        os.path.join(config.EMBEDDING_DIR, 'val_embeddings.pkl'),
        metadata={
            'dataset': 'validation',
            'num_images': len(val_paths),
            'categories': val_dataset.categories
        }
    )
    
    # Extract embeddings for training set (optional, for analysis)
    print("\n" + "="*80)
    print("Extracting embeddings for TRAINING SET")
    print("="*80)
    
    train_loader, train_dataset = get_dataloader(
        config.TRAIN_DIR,
        config.INPUT_SIZE,
        batch_size=32,
        is_train=False,
        use_augmentation=False,
        num_workers=config.NUM_WORKERS,
        return_path=True
    )
    
    train_embeddings, train_labels, train_paths = extract_embeddings(
        model, train_loader, device, desc="Training Set"
    )
    
    save_embeddings(
        train_embeddings,
        train_labels,
        train_paths,
        os.path.join(config.EMBEDDING_DIR, 'train_embeddings.pkl'),
        metadata={
            'dataset': 'train',
            'num_images': len(train_paths),
            'categories': train_dataset.categories
        }
    )
    
    print("\n" + "="*80)
    print("Embedding extraction completed!")
    print(f"All embeddings saved to: {config.EMBEDDING_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

