import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class MultiLabelImageDataset(Dataset):
    def __init__(self, image_dir, super_labels, sub_labels, indices, transform=None):
        self.image_dir = image_dir
        self.super_labels = super_labels
        self.sub_labels = sub_labels
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_name = self.image_files[real_idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        super_label = torch.tensor(self.super_labels[real_idx], dtype=torch.float32)
        sub_label = torch.tensor(self.sub_labels[real_idx], dtype=torch.float32)
        return image, super_label, sub_label

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def create_data_loaders(image_dir, super_labels, sub_labels, batch_size=32, val_split=0.2):
    train_transform, val_transform = get_data_transforms()
    dataset_size = len(os.listdir(image_dir))
    indices = np.arange(dataset_size)
    split = int(np.floor(val_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_dataset = MultiLabelImageDataset(
        image_dir,
        super_labels,
        sub_labels,
        train_indices,
        transform=train_transform
    )
    val_dataset = MultiLabelImageDataset(
        image_dir,
        super_labels,
        sub_labels,
        val_indices,
        transform=val_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader 