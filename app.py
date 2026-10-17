import os
from pathlib import Path
from PIL import Image
import streamlit as st
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from torchvision import datasets

# Define paths and transformations
data_dir = Path('dogs_vs_cats_dataset')
test_dir = data_dir / 'test'

# Convert image to RGB to ensure 3 channels
def to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pre-trained model and modify the final layer
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the saved model
model_save_path = Path('dogs_vs_cats_model.pth')
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Function to evaluate the model on a single image
def evaluate_image(image):
    # Convert to RGB first to ensure 3 channels
    image = to_rgb(image)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Function to get label based on prediction
def get_label(prediction, class_names):
    return class_names[prediction]

# Streamlit app
st.title('Dog vs Cat Classifier')

# Load train dataset to get class names
train_dataset = datasets.ImageFolder(root=data_dir / 'train', transform=transform)

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=400)
    
    # Get prediction
    prediction = evaluate_image(image)
    class_names = train_dataset.classes
    label = get_label(prediction, class_names)

    # Display prediction without confidence
    st.markdown(f"""
        <div style="text-align: center;">
            <h2 style="font-size: 24px; margin: 0;">Prediction: {label.upper()}</h2>
        </div>
    """, unsafe_allow_html=True)

# Evaluate model on test dataset only if no image is uploaded
if uploaded_file is None:
    if st.button('Evaluate Model on Test Dataset'):
        # Create dataset and data loader
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        # Initialize variables to store predictions and true labels
        predictions = []
        true_labels = []

        # Evaluate the model
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.amp.autocast(device_type="cuda"):
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

        # Display results
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Confusion Matrix:\n{conf_matrix}")

