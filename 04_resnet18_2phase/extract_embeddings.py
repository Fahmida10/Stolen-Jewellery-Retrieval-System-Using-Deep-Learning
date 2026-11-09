"""Extract embeddings for Stage 4 using the standard jewelry_retrieval artifact format."""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from model import ResNet18Classifier
from dataset import JewelryDataset, StolenJewelryDataset
from shared_retrieval_pipeline import extract_embeddings, save_embeddings


def get_eval_transform(input_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])


def build_loader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    model_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    model = ResNet18Classifier(num_classes=NUM_CLASSES, freeze_backbone=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    transform = get_eval_transform(INPUT_SIZE)

    stolen_dataset = StolenJewelryDataset(STOLEN_DIR, transform=transform)
    stolen_loader = build_loader(stolen_dataset, BATCH_SIZE, NUM_WORKERS)
    stolen_embeddings, _, stolen_paths = extract_embeddings(model, stolen_loader, device, desc="Stolen")
    save_embeddings(
        stolen_embeddings,
        None,
        stolen_paths,
        os.path.join(EMBEDDING_DIR, "stolen_embeddings.pkl"),
        metadata={"dataset": "stolen", "num_images": len(stolen_paths), "stage": STAGE_NAME},
    )

    test_dataset = JewelryDataset(TEST_DIR, transform=transform, return_path=True)
    test_loader = build_loader(test_dataset, BATCH_SIZE, NUM_WORKERS)
    test_embeddings, test_labels, test_paths = extract_embeddings(model, test_loader, device, desc="Test")
    save_embeddings(
        test_embeddings,
        test_labels,
        test_paths,
        os.path.join(EMBEDDING_DIR, "test_embeddings.pkl"),
        metadata={"dataset": "test", "num_images": len(test_paths), "categories": test_dataset.categories, "stage": STAGE_NAME},
    )

    val_dataset = JewelryDataset(VAL_DIR, transform=transform, return_path=True)
    val_loader = build_loader(val_dataset, BATCH_SIZE, NUM_WORKERS)
    val_embeddings, val_labels, val_paths = extract_embeddings(model, val_loader, device, desc="Validation")
    save_embeddings(
        val_embeddings,
        val_labels,
        val_paths,
        os.path.join(EMBEDDING_DIR, "val_embeddings.pkl"),
        metadata={"dataset": "validation", "num_images": len(val_paths), "categories": val_dataset.categories, "stage": STAGE_NAME},
    )

    train_dataset = JewelryDataset(TRAIN_DIR, transform=transform, return_path=True)
    train_loader = build_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
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
