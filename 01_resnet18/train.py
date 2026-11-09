"""Single-phase ResNet18 training with a 100-epoch budget."""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add parent for any shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from model import ResNet18Classifier
from dataset import JewelryDataset


def get_imagenet_transforms(input_size=224, is_train=True, use_augmentation=False):
    """ImageNet normalization for ResNet."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train and use_augmentation:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])
    return transform


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Data
    train_transform = get_imagenet_transforms(INPUT_SIZE, is_train=True, use_augmentation=USE_AUGMENTATION)
    val_transform = get_imagenet_transforms(INPUT_SIZE, is_train=False)
    
    train_dataset = JewelryDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = JewelryDataset(VAL_DIR, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    # Model
    model = ResNet18Classifier(num_classes=NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )
    # No scheduler - single phase
    
    print("\n" + "="*60)
    print("ResNet18 baseline")
    print("="*60)
    print(f"Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}")
    print(f"Backbone: {BACKBONE} | Frozen: {FREEZE_BACKBONE}")
    print("Single-phase, CE-only, no SupCon, no scheduler")
    print("="*60)
    
    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")
        
        train_acc = 100. * correct / total
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Train Acc={train_acc:.2f}% Val Acc={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "train_acc": train_acc
            }, path)
            print(f"  -> Saved best model (val_acc={val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoint: {os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)}")
    print("="*60)


if __name__ == "__main__":
    main()
