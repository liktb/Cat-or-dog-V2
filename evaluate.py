import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# Define data paths
data_dir = Path('dogs_vs_cats_dataset')
test_dir = data_dir / 'test'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((244, 224)),  # Resize to 244x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create dataset
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create data loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load pre-trained model and modify the final layer
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # For binary classification

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the saved model
model_save_path = Path('dogs_vs_cats_model.pth')
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Initialize variables to store predictions and true labels
predictions = []
true_labels = []

# Evaluate the model
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast(device_type="cuda"):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
