"""
Contains functionality to create dataloader for image classification task
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloader(
    train_dir,
    test_dir,
    transform,
    batch_size,
    num_workers=NUM_WORKERS
):
    train_data = datasets.ImageFolder(train_dir, transform)
    test_data = datasets.ImageFolder(test_dir, transform)
    
    class_names = train_data.classes
    
    train_dataloader = DataLoader(train_data, batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(train_data, batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    
    return train_dataloader, test_dataloader, class_names
