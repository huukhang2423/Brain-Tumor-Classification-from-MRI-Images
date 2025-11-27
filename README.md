# Brain Tumor Classification System

A machine learning-based system for classifying brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary Tumor using Support Vector Machine (SVM).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a brain tumor classification system using traditional machine learning techniques. The system achieves **83.83% accuracy** on the test set, demonstrating that classical ML approaches can compete with deep learning methods while being more lightweight and interpretable.

### Key Highlights
- **High Accuracy**: 83.83% overall accuracy
- **Lightweight**: Model size only 546 KB (vs. 100-300 MB for deep learning)
- **Fast Inference**: No GPU required
- **Easy Deployment**: Simple scikit-learn dependencies
- **Interpretable**: Clear feature engineering and model structure

## Features

- **4-Class Classification**: Glioma, Meningioma, No Tumor, Pituitary
- **Handcrafted Features**:
  - Histogram-based features (intensity distribution)
  - Statistical features (mean, standard deviation)
  - Texture features (Laplacian variance)
  - Edge density (Canny edge detection)
- **Comprehensive Evaluation**:
  - Confusion matrix visualization
  - ROC curves with AUC scores
  - Sample predictions display
  - Class distribution analysis
- **Production-Ready**: Saved models and scalers for deployment

## Dataset

The system uses the **Brain Tumor MRI Dataset** with the following structure:

```
archive/
├── Training/
│   ├── glioma/          (1321 images)
│   ├── meningioma/      (1339 images)
│   ├── notumor/         (1595 images)
│   └── pituitary/       (1457 images)
└── Testing/
    ├── glioma/          (300 images)
    ├── meningioma/      (306 images)
    ├── notumor/         (405 images)
    └── pituitary/       (300 images)
```

**Total**: 5,712 training images | 1,311 test images

### Tumor Types

1. **Glioma**: Tumors that arise from glial cells in the brain and spinal cord
2. **Meningioma**: Tumors that form in the meninges (membranes surrounding the brain)
3. **No Tumor**: Normal brain scans without tumors
4. **Pituitary**: Tumors that develop in the pituitary gland

## Performance

### Overall Metrics
- **Accuracy**: 83.83%
- **Macro Average F1-Score**: 0.83
- **Weighted Average F1-Score**: 0.84

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support | AUC |
|-------|-----------|--------|----------|---------|-----|
| Glioma | 0.73 | 0.66 | 0.69 | 300 | 0.937 |
| Meningioma | 0.72 | 0.80 | 0.76 | 306 | 0.940 |
| No Tumor | **0.99** | **0.96** | **0.97** | 405 | **0.993** |
| Pituitary | 0.88 | 0.90 | 0.89 | 300 | 0.983 |

### Key Findings
✅ **Excellent "No Tumor" detection** (97% F1-score) - Critical for screening
✅ **High Pituitary accuracy** (89% F1-score) - Distinctive location helps
✅ **All AUC scores > 0.93** - Strong discriminative power
⚠️ **Glioma-Meningioma confusion** - Main area for improvement (28% misclassification)

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
Place your Brain Tumor MRI dataset in the `archive/` folder with the structure shown above.

## Usage

### Training the Model
Run the complete pipeline (data loading, feature extraction, training, evaluation):

```bash
python brain_tumor_system.py
```

The script will:
1. Load and preprocess images (resize to 224×224)
2. Extract handcrafted features
3. Train SVM model with RBF kernel
4. Evaluate on test set
5. Generate visualizations
6. Save models and results

### Making Predictions
```python
import joblib
import cv2
import numpy as np

# Load saved model and scaler
model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load and preprocess image
img = cv2.imread('path/to/brain_scan.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

# Extract features (use extract_features function from script)
features = extract_features([img])  # Returns (1, 20) array
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(f"Predicted class: {classes[prediction[0]]}")
print(f"Confidence: {probabilities[0][prediction[0]]:.2%}")
```

## Results

### Generated Files

After running the script, the following files are created:

**Models** (`models/`):
- `svm_model.pkl` - Trained SVM model (546 KB)
- `scaler.pkl` - StandardScaler for feature normalization (1 KB)

**Results** (`results/`):
- `confusion_matrix.png` - Normalized confusion matrix
- `roc_curves.png` - ROC curves for all classes
- `sample_predictions.png` - 16 sample predictions
- `class_distribution.png` - Train/test distribution
- `svm_results.json` - Detailed results in JSON format

### Sample Visualizations

#### Confusion Matrix
Shows model predictions vs. actual labels with normalization.

#### ROC Curves
All classes achieve AUC > 0.93, indicating excellent separation.

#### Sample Predictions
Visual verification of model performance with confidence scores.

## Model Details

### Feature Engineering
The system extracts **20 features** per image:

1. **Histogram Features (16 bins)**: Intensity distribution
2. **Mean Intensity**: Average grayscale value
3. **Standard Deviation**: Intensity variance
4. **Texture Variance**: Laplacian filter response
5. **Edge Density**: Percentage of edge pixels (Canny)

### SVM Configuration
```python
SVC(
    kernel='rbf',           # Radial Basis Function
    C=10,                   # Regularization parameter
    gamma='scale',          # Kernel coefficient
    probability=True,       # Enable probability estimates
    class_weight='balanced', # Handle class imbalance
    random_state=42
)
```

### Why SVM?
- **Effective in high-dimensional spaces** (20 features)
- **Memory efficient** (only support vectors stored)
- **Handles non-linear boundaries** (RBF kernel)
- **Robust to overfitting** (C regularization)
- **Probability calibration** (for confidence scores)

## Project Structure

```
brain-tumor-classification/
│
├── brain_tumor_system.py      # Main script
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── LICENSE                     # Project license
│
├── archive/                    # Dataset (not included)
│   ├── Training/
│   └── Testing/
│
├── models/                     # Saved models
│   ├── svm_model.pkl
│   └── scaler.pkl
│
└── results/                    # Output visualizations
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── sample_predictions.png
    ├── class_distribution.png
    └── svm_results.json
```

## Future Improvements

### Short-term
- [ ] Add more texture features (GLCM, LBP, HOG)
- [ ] Implement GridSearchCV for hyperparameter tuning
- [ ] Create ensemble with Random Forest
- [ ] Add cross-validation for robust evaluation
- [ ] Implement threshold tuning for better Glioma recall

### Long-term
- [ ] Web interface for easy prediction
- [ ] REST API for integration
- [ ] Mobile app deployment
- [ ] Comparison with deep learning models
- [ ] LIME/SHAP for explainability
- [ ] Clinical trial integration

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Improved feature extraction methods
- Additional ML algorithms (XGBoost, Neural Networks)
- Data augmentation techniques
- Model interpretation tools
- Documentation improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: Brain Tumor MRI Dataset (Kaggle/public domain)
- **Libraries**: scikit-learn, OpenCV, NumPy, Matplotlib, Seaborn
- **Inspiration**: Medical image analysis research community


