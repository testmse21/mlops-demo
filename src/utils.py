# import os
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# def load_data(data_dir='data/images'):
#     # Assuming the data is in a directory with subfolders 'cat' and 'not_cat'
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#     ])
    
#     dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
#     return dataloader  # This will return a DataLoader object, which can be iterated over in the training loop

# def preprocess(image):
#     # Preprocessing steps: resize and normalize (optional)
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
#     ])
#     return transform(image)
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def load_data():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder("data/images", transform=transform)

    # Split into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader
