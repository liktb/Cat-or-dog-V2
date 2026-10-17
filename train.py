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
scaler = GradScaler()

# Training loop
def train_model(model, train_loader, optimizer, criterion, scaler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_dataset)}")

# Test loop
def test_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Loss: {running_loss / len(test_dataset)}, Accuracy: {100 * correct / total}%")

# Train the model
train_model(model, train_loader, optimizer, criterion, scaler, num_epochs=10)

# Test the model
test_model(model, test_loader, criterion)

# Save the trained model
model_save_path = Path('dogs_vs_cats_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
