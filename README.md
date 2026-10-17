# Cats vs. Dogs Image Classification Pipeline

A production-grade Deep Learning computer vision pipeline built using PyTorch, transfer learning with EfficientNet-B0,
Accelerated by NVIDIA CUDA and optimized for high accuracy and rapid inference on custom image classification tasks.
while using a 50K reference image dataset.

---

## Model Performance & Evaluation Metrics

Evaluated on 4,653 unseen test images:

| Metric | Score |
|--------|-------|
| Overall Accuracy | 97.32% |
| F1 Score | 97.36% |
| Recall | 98.88% |
| Precision | 95.89% |

### Class-Specific Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cat | 95.89% | 98.88% | 97.36% |
| Dog | 98.88% | 95.89% | 97.36% |

### Confusion Matrix

| Class | Cat | Dog |
|-------|-----------|--------|
| correct | 2394 | 2472 |
| wrong | 106 | 28 |
**Interpretation:**
- 2,394 cats correctly identified as cats
- 2,472 dogs correctly identified as dogs
- 106 cats misclassified as dogs (false positives)
- 28 dogs misclassified as cats (false positives)

---

## Project Structure

```
trial/
│
├── cats_dogs_244/              # Dataset folder (train / test splits)
├── train_cv.py                 # Data augmentation, transfer learning setup, training loop & artifact saving
├── predict_cv.py               # Single-image inference script loading the saved .pt weights
├── evaluate_cv.py              # Comprehensive evaluation script (Accuracy, Classification Report, Confusion Matrix)
├── vision_model_best.pt        # Serialized PyTorch model weights & class mapping dictionary
├── requirements.txt            # Locked Python package dependencies
└── README.md                   # Project documentation & metrics summary
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU (optional, for CUDA acceleration)
- 4GB+ RAM (for training)

### Installation

1. Clone or download this repository
   ```bash
   git clone https://github.com/liktb/Cat-or-dog-V2.git
   cd Cat-or-dog-V2
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Train the Model (Generates vision_model_best.pt)
```bash
python train_cv.py
```

### Evaluate Model Performance
```bash
python evaluate_cv.py
```

### Run Single-Image Prediction
```bash
python predict_cv.py
```

---

## Deployment

### Web Application
- Live Demo: [Cat or Dog Classifier](https://cat-or-dog-v2.streamlit.app/)
- Local Streamlit App: streamlit run app.py

---

## Technical Specifications

| Component | Details |
|-----------|---------|
| Framework | PyTorch 2.1+ |
| Architecture | EfficientNet-B0 (Transfer Learning) |
| Input Size | 244×244 pixels |
| Training Epochs | 5 |
| Batch Size | 32 |
| Optimizer | SGD with momentum |
| Learning Rate | 0.001 |
| CUDA Support | Yes (NVIDIA GPU) |

---

## Notes

- vision_model_best.pt is a pre-trained model (5 epochs) ready for inference
- Dataset located in cats_dogs_244/ (train/test splits included)
- Model achieves 97.32% accuracy on unseen test data

---

## Contributors

- Model Pipeline & Training: [Ryan Dinto](https://github.com/liktb)
