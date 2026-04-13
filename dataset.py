import os
from torchvision import datasets, transforms

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return train_transform, val_test_transform

def load_datasets(dataset_root):
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "validation")
    test_dir = os.path.join(dataset_root, "test")

    train_tf, val_test_tf = get_transforms()
    datasets_dict = {
        "train": datasets.ImageFolder(train_dir, transform=train_tf),
        "val": datasets.ImageFolder(val_dir, transform=val_test_tf),
        "test": datasets.ImageFolder(test_dir, transform=val_test_tf)
    }
    return datasets_dict