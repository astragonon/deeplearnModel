import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_cifar10_dataloaders(batch_size=32):
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 检查数据是否已下载
    data_dir = './datasets'
    os.makedirs(data_dir, exist_ok=True)
    
    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=not os.path.exists(f'{data_dir}/cifar-10-batches-py'),
        transform=train_transform
    )
    
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,  # 假设训练集下载时测试集也已下载
        transform=test_transform
    )
    
    num_workers = min(4, os.cpu_count()-1) if os.cpu_count() > 1 else 0
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes
