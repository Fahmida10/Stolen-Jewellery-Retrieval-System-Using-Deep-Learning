"""Extract embeddings for Stage 5 using the standard jewelry_retrieval artifact format."""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
import config
from model import build_model
from dataset import get_dataloader, get_stolen_dataloader
from shared_retrieval_pipeline import extract_embeddings, save_embeddings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    model_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    model = build_model(config, num_classes=NUM_CLASSES, use_classifier=True, freeze_backbone=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    stolen_loader, _ = get_stolen_dataloader(STOLEN_DIR, INPUT_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    stolen_embeddings, _, stolen_paths = extract_embeddings(model, stolen_loader, device, desc="Stolen")
    save_embeddings(
        stolen_embeddings,
        None,
        stolen_paths,
        os.path.join(EMBEDDING_DIR, "stolen_embeddings.pkl"),
        metadata={"dataset": "stolen", "num_images": len(stolen_paths), "stage": STAGE_NAME},
    )

    test_loader, test_dataset = get_dataloader(
        TEST_DIR,
        INPUT_SIZE,
        BATCH_SIZE,
        is_train=False,
        use_augmentation=False,
        num_workers=NUM_WORKERS,
        return_path=True,
    )
    test_embeddings, test_labels, test_paths = extract_embeddings(model, test_loader, device, desc="Test")
    save_embeddings(
        test_embeddings,
        test_labels,
        test_paths,
        os.path.join(EMBEDDING_DIR, "test_embeddings.pkl"),
        metadata={"dataset": "test", "num_images": len(test_paths), "categories": test_dataset.categories, "stage": STAGE_NAME},
    )

    val_loader, val_dataset = get_dataloader(
        VAL_DIR,
        INPUT_SIZE,
        BATCH_SIZE,
        is_train=False,
        use_augmentation=False,
        num_workers=NUM_WORKERS,
        return_path=True,
    )
    val_embeddings, val_labels, val_paths = extract_embeddings(model, val_loader, device, desc="Validation")
    save_embeddings(
        val_embeddings,
        val_labels,
        val_paths,
        os.path.join(EMBEDDING_DIR, "val_embeddings.pkl"),
        metadata={"dataset": "validation", "num_images": len(val_paths), "categories": val_dataset.categories, "stage": STAGE_NAME},
    )

    train_loader, train_dataset = get_dataloader(
        TRAIN_DIR,
        INPUT_SIZE,
        BATCH_SIZE,
        is_train=False,
        use_augmentation=False,
        num_workers=NUM_WORKERS,
        return_path=True,
    )
    train_embeddings, train_labels, train_paths = extract_embeddings(model, train_loader, device, desc="Train")
    save_embeddings(
        train_embeddings,
        train_labels,
        train_paths,
        os.path.join(EMBEDDING_DIR, "train_embeddings.pkl"),
        metadata={"dataset": "train", "num_images": len(train_paths), "categories": train_dataset.categories, "stage": STAGE_NAME},
    )

    print(f"Embeddings saved to: {EMBEDDING_DIR}")


if __name__ == "__main__":
    main()
