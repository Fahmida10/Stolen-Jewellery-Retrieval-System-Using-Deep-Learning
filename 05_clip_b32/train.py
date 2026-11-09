"""
Stage 5 - Basic OpenCLIP: single-phase training, CE-only
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import *
from model import build_model
from dataset import get_dataloader


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, _ = get_dataloader(TRAIN_DIR, INPUT_SIZE, BATCH_SIZE, is_train=True, use_augmentation=USE_AUGMENTATION, num_workers=NUM_WORKERS)
    val_loader, _ = get_dataloader(VAL_DIR, INPUT_SIZE, BATCH_SIZE, is_train=False, use_augmentation=False, num_workers=NUM_WORKERS)

    import config
    model = build_model(config, num_classes=NUM_CLASSES, use_classifier=True, freeze_backbone=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda') if MIXED_PRECISION and device.type == 'cuda' else None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print("\n" + "="*60)
    print("Stage 5 - Basic OpenCLIP (single-phase, CE-only)")
    print("="*60)
    print(f"Model: {MODEL_NAME} | Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE}")
    print("="*60)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if MIXED_PRECISION and device.type == 'cuda':
                with autocast('cuda'):
                    _, logits = model(images, return_embeddings=True)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, logits = model(images, return_embeddings=True)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, logits = model(images, return_embeddings=True)
                _, pred = logits.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch}: Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc}, os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME))
            print(f"  -> Saved best (val_acc={val_acc:.2f}%)")

    print("\n" + "="*60)
    print(f"Done. Best val accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
