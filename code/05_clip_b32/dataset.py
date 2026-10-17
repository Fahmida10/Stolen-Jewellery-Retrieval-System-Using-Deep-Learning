"""
Dataset loader for jewelry images (CLIP normalization)
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class JewelryDataset(Dataset):
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
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(category_path, img_name), self.category_to_idx[category]))
        print(f"Loaded {len(self.samples)} images from {len(self.categories)} categories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.return_path:
            return image, label, img_path
        return image, label


class StolenJewelryDataset(Dataset):
    def __init__(self, stolen_dir: str, transform=None):
        self.stolen_dir = stolen_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(stolen_dir, name)
            for name in os.listdir(stolen_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
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


def get_transforms(input_size=224, is_train=True, use_augmentation=True):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    if is_train and use_augmentation:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])


def get_dataloader(root_dir, input_size, batch_size, is_train=True, use_augmentation=True, num_workers=4, return_path=False):
    transform = get_transforms(input_size, is_train, use_augmentation)
    dataset = JewelryDataset(root_dir, transform=transform, return_path=return_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True, drop_last=is_train), dataset


def get_stolen_dataloader(stolen_dir, input_size, batch_size=32, num_workers=4):
    transform = get_transforms(input_size, is_train=False, use_augmentation=False)
    dataset = StolenJewelryDataset(stolen_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, dataset
