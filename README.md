# Automated Material Stream Identification (MSI) System

A machine learning-based waste classification system using SVM and k-NN classifiers with real-time camera integration.

## Project Overview

This system classifies waste materials into 6 categories (glass, paper, cardboard, plastic, metal, trash) plus an "unknown" class for out-of-distribution items.

## Features

- **Data Augmentation**: Balances dataset to 500 samples per class using rotation, flipping, brightness adjustment, scaling, and noise
- **Feature Extraction**: Combines HOG (Histogram of Oriented Gradients) and color histogram features
- **Dual Classifiers**: Implements both SVM and k-NN classifiers
- **Rejection Mechanism**: Handles unknown/out-of-distribution samples
- **Real-time Classification**: Live webcam feed processing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Augmentation

Balance the dataset to 500 samples per class:

```bash
python data_augmentation.py
```

### 2. Train Models

Train both SVM and k-NN classifiers:

```bash
python train_models.py
```

### 3. Real-time Classification

Run the live camera application:

```bash
# Use SVM model (default)
python realtime_classifier.py svm

# Use k-NN model
python realtime_classifier.py knn
```

Controls:
- Press 'q' to quit
- Press 's' to switch between SVM and k-NN models

### 4. Model Evaluation

Evaluate model performance:

```bash
python evaluate_model.py
```

## Architecture

### Feature Extraction Pipeline

**Image → Preprocessing → Feature Extraction → Feature Vector**

1. **Preprocessing**:
   - Resize to 128x128 pixels
   - Gaussian blur for noise reduction

2. **HOG Features** (Shape/Edge Information):
   - 9 orientations
   - 8x8 pixels per cell
   - 2x2 cells per block
   - Captures object shape and structure

3. **Color Histogram Features** (Color Distribution):
   - 32 bins per channel (BGR)
   - Normalized histograms
   - Captures material color characteristics

4. **Combined Feature Vector**: ~1,800 dimensions

### Classifiers

#### SVM (Support Vector Machine)
- **Kernel**: RBF (Radial Basis Function)
- **C**: 10.0 (regularization parameter)
- **Gamma**: 'scale' (kernel coefficient)
- **Advantages**: Better generalization, handles high-dimensional data well
- **Rejection**: Uses probability estimates with threshold

#### k-NN (k-Nearest Neighbors)
- **k**: 5 neighbors
- **Weights**: Distance-based (closer neighbors have more influence)
- **Metric**: Euclidean distance
- **Advantages**: Simple, no training phase, intuitive
- **Rejection**: Uses distance-based confidence

### Unknown Class Handling

Both models implement rejection mechanisms:
- **Confidence Threshold**: 0.6
- Samples below threshold → classified as "unknown"
- Prevents misclassification of out-of-distribution items

## Project Structure

```
.
├── config.py                 # Configuration settings
├── data_augmentation.py      # Dataset augmentation
├── feature_extraction.py     # Feature extraction module
├── train_models.py          # Model training script
├── realtime_classifier.py   # Real-time application
├── evaluate_model.py        # Model evaluation
├── requirements.txt         # Dependencies
├── dataset/                 # Original dataset
├── dataset_augmented/       # Augmented dataset
└── models/                  # Saved models
    ├── svm_classifier.pkl
    ├── knn_classifier.pkl
    └── feature_scaler.pkl
```

## Technical Details

### Data Augmentation Techniques

1. **Rotation**: ±20 degrees random rotation
2. **Horizontal Flip**: Mirror image
3. **Brightness**: 0.7-1.3x adjustment in HSV space
4. **Scaling**: 0.8-1.2x with center crop/padding
5. **Gaussian Noise**: Random noise addition

### Feature Justification

**HOG Features**: Excellent for capturing shape and edge information, invariant to illumination changes, widely used in object detection.

**Color Histograms**: Captures color distribution which is crucial for material identification (e.g., glass is often transparent/green, metal is gray/silver).

**Combination**: Provides both structural and appearance information for robust classification.

### Model Comparison

| Aspect | SVM | k-NN |
|--------|-----|------|
| Training Time | Moderate | Instant |
| Prediction Time | Fast | Slower (distance computation) |
| Memory | Compact (support vectors) | Large (stores all data) |
| Generalization | Better | Can overfit |
| Hyperparameters | C, gamma, kernel | k, weights, metric |

## Performance Targets

- **Minimum Validation Accuracy**: 0.85 (85%)
- **Augmentation Factor**: >30% increase in dataset size
- **Real-time Processing**: >10 FPS on standard webcam

## Competition Submission

The trained model files in `models/` directory can be submitted for evaluation on the hidden test set.

## Authors

[Your Name/Team Name]

## License

Academic Project
