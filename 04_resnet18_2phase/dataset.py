"""
Dataset utilities with optional path return and stolen-image support.
"""

import os
from torch.utils.data import Dataset
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class JewelryDataset(Dataset):
    """Jewelry dataset - only includes actual category directories."""

    def __init__(self, root_dir: str, transform=None, return_path: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        self.samples = []
        self.categories = sorted([
            c for c in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, c))
        ])
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        for category in self.categories:
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith(IMAGE_EXTENSIONS):
                    img_path = os.path.join(category_path, img_name)
                    self.samples.append((img_path, self.category_to_idx[category]))

        print(f"Loaded {len(self.samples)} images from {len(self.categories)} categories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, label, img_path
        return image, label


class StolenJewelryDataset(Dataset):
    """Dataset for stolen jewelry images (no labels)."""

    def __init__(self, stolen_dir: str, transform=None):
        self.stolen_dir = stolen_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(stolen_dir, img_name)
            for img_name in os.listdir(stolen_dir)
            if img_name.lower().endswith(IMAGE_EXTENSIONS)
        ])
        print(f"Loaded {len(self.image_paths)} stolen jewelry images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path
