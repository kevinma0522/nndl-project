import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import pickle

def download_cifar100():
    """Download CIFAR-100 dataset and extract fine and coarse labels."""
    print("Downloading CIFAR-100 dataset...")
    transform = transforms.ToTensor()  # Only convert to tensor, no normalization
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    # Load the raw data to get coarse labels
    with open('./data/cifar-100-python/train', 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        coarse_labels = entry['coarse_labels']
    return trainset, coarse_labels

def create_multi_label_dataset(trainset, coarse_labels, output_dir='./dataset', num_samples=1000):
    """Create a multi-label dataset from CIFAR-100"""
    print("Creating multi-label dataset...")
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    super_labels = []
    sub_labels = []

    for i in tqdm(range(num_samples)):
        img, fine_label = trainset[i]
        coarse_label = coarse_labels[i]

        img = transforms.ToPILImage()(img)
        img_path = os.path.join(images_dir, f'image_{i:04d}.png')
        img.save(img_path)

        super_label = np.zeros(20)  # 20 super-classes
        sub_label = np.zeros(100)   # 100 sub-classes

        super_label[coarse_label] = 1
        sub_label[fine_label] = 1

        super_labels.append(super_label)
        sub_labels.append(sub_label)

    super_labels = np.array(super_labels)
    sub_labels = np.array(sub_labels)
    np.save(os.path.join(output_dir, 'super_labels.npy'), super_labels)
    np.save(os.path.join(output_dir, 'sub_labels.npy'), sub_labels)

    print(f"Dataset created successfully in {output_dir}")
    print(f"Number of images: {num_samples}")
    print(f"Super-class labels shape: {super_labels.shape}")
    print(f"Sub-class labels shape: {sub_labels.shape}")

def main():
    trainset, coarse_labels = download_cifar100()
    create_multi_label_dataset(trainset, coarse_labels, num_samples=1000)

if __name__ == '__main__':
    main() 