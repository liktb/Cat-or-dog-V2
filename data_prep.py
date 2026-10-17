import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.amp import autocast, GradScaler

# Define data paths
data_dir = Path('dogs_vs_cats_dataset')
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((244, 224)),  # Resize to 244x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load pre-trained model and modify the final layer
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # For binary classification

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Initialize gradient scaler for mixed precision training
scaler = GradScaler(device="cuda")

# Print model and dataset information
print(f"Model: {model}")
print(f"Train Dataset size: {len(train_dataset)}")
print(f"Test Dataset size: {len(test_dataset)}")
