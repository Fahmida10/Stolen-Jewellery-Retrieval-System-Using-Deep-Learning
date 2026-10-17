"""
Two-phase CNN training
Phase 1: Freeze backbone, train classifier head.
Phase 2: Unfreeze backbone, fine-tune with lower LR.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
from tqdm import tqdm

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
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

    # Model - start with frozen backbone
    model = ResNet18Classifier(num_classes=NUM_CLASSES, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda') if MIXED_PRECISION and device.type == 'cuda' else None

    print("\n" + "="*60)
    print("Commit 2 - Improved CNN baseline (two-phase)")
    print("="*60)
    print("Phase 1: Frozen backbone, train head only")
    print("Phase 2: Unfreeze backbone, fine-tune with lower LR")
    print("="*60)

    best_acc = 0.0
    global_epoch = 0

    # Phase 1: Freeze backbone, train head only
    print("\n--- PHASE 1: Frozen backbone ---")
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS) if USE_SCHEDULER else None

    for epoch in range(1, PHASE1_EPOCHS + 1):
        global_epoch += 1
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Phase1 Epoch {epoch}/{PHASE1_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if MIXED_PRECISION and device.type == 'cuda':
                with autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        if scheduler:
            scheduler.step()

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
        print(f"Phase1 Epoch {epoch}: Train Loss={train_loss:.4f} Train Acc={train_acc:.2f}% Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "phase": 1
            }, os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME))

    # Phase 2: Unfreeze backbone, fine-tune with lower LR
    print("\n--- PHASE 2: Unfreeze backbone ---")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": PHASE2_BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": model.classifier.parameters(), "lr": PHASE2_HEAD_LR, "weight_decay": WEIGHT_DECAY}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS) if USE_SCHEDULER else None

    for epoch in range(1, PHASE2_EPOCHS + 1):
        global_epoch += 1
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Phase2 Epoch {epoch}/{PHASE2_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if MIXED_PRECISION and device.type == 'cuda':
                with autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        if scheduler:
            scheduler.step()

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
        print(f"Phase2 Epoch {epoch}: Train Loss={train_loss:.4f} Train Acc={train_acc:.2f}% Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "phase": 2
            }, os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME))
            print(f"  -> Saved best model (val_acc={val_acc:.2f}%)")

    print("\n" + "="*60)
    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoint: {os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)}")
    print("="*60)


if __name__ == "__main__":
    main()
