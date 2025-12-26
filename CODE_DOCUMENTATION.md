# Complete Code Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Configuration File](#configuration-file)
4. [Core Components](#core-components)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation and Comparison](#evaluation-and-comparison)
7. [Deployment](#deployment)
8. [Helper Scripts](#helper-scripts)

---

## Project Overview

**Goal:** Automated Material Stream Identification (MSI) system that classifies waste into 6 categories + 1 unknown class using deep learning transfer learning with classical ML classifiers.

**Architecture:**
```
Raw Images → Data Augmentation → ResNet-18 Features → SVM/k-NN → Classification
```

**Key Technologies:**
- **Deep Learning:** ResNet-18 (pre-trained on ImageNet) via PyTorch
- **Classical ML:** SVM with RBF kernel, k-NN with distance weighting
- **Feature Extraction:** img2vec_pytorch (512-dimensional features)
- **Real-time Processing:** OpenCV for camera integration

---

## Project Structure

```
machine-learning-project/
├── config.py                      # Central configuration (hyperparameters, paths)
├── feature_extraction.py          # ResNet-18 feature extraction + preprocessing
├── data_augmentation.py           # Augment dataset to 800 samples/class
├── train_models.py                # Train SVM and k-NN classifiers
├── ensemble_classifier.py         # Combine SVM + k-NN (weighted voting)
├── evaluate_model.py              # Evaluate on test set
├── compare_models.py              # Speed and accuracy comparison
├── realtime_classifier.py         # Live camera application
├── predict_hidden_dataset.py      # Generate predictions for competition
├── run_pipeline.py                # Main orchestration script
├── test_setup.py                  # Verify dependencies installed
├── clean_dataset.py               # Remove corrupted images
├── test_feature_pipeline.py       # Test feature extraction
├── requirements.txt               # Python dependencies
├── dataset/                       # Original images (1,865 total)
│   ├── glass/                    # 385 images
│   ├── paper/                    # 449 images
│   ├── cardboard/                # 247 images
│   ├── plastic/                  # 363 images
│   ├── metal/                    # 315 images
│   └── trash/                    # 106 images
├── dataset_augmented/            # Augmented images (4,800 total)
│   ├── glass/                    # 800 images
│   ├── paper/                    # 800 images
│   ├── cardboard/                # 800 images
│   ├── plastic/                  # 800 images
│   ├── metal/                    # 800 images
│   └── trash/                    # 800 images
├── models/                       # Saved trained models
│   ├── svm_classifier.pkl        # Trained SVM (97.08% accuracy)
│   ├── knn_classifier.pkl        # Trained k-NN (94.79% accuracy)
│   ├── feature_scaler.pkl        # StandardScaler for normalization
│   └── accuracies.txt            # Training accuracy log
└── Documentation/
    ├── README.md                 # Quick start guide
    ├── TECHNICAL_REPORT_TEMPLATE.md  # Comprehensive technical report
    ├── ACCURACY_GUIDE.md         # Tips for achieving high accuracy
    ├── ARCHITECTURE.md           # System architecture diagrams
    ├── QUICKSTART.md            # Fast setup instructions
    └── CODE_DOCUMENTATION.md    # This file
```

---

## Configuration File

### `config.py`

**Purpose:** Centralized configuration for all hyperparameters, paths, and settings.

**Key Parameters:**

```python
# Dataset Paths
DATASET_PATH = "dataset"                    # Original dataset
AUGMENTED_PATH = "dataset_augmented"        # Augmented dataset (800/class)

# Class Definitions
CLASSES = {
    0: "glass",       # Glass bottles, jars
    1: "paper",       # Newspapers, office paper
    2: "cardboard",   # Boxes, corrugated material
    3: "plastic",     # Bottles, containers
    4: "metal",       # Aluminum cans, metal objects
    5: "trash",       # Non-recyclable waste
    6: "unknown"      # Out-of-distribution items
}

# Data Augmentation Settings
TARGET_SAMPLES_PER_CLASS = 800              # Balance all classes to 800 samples
AUGMENTATION_FACTOR = 1.6                   # 60% increase

# Feature Extraction (ResNet-18)
IMAGE_SIZE = (224, 224)                     # ResNet-18 input size
# ResNet-18 outputs 512-dimensional features automatically

# SVM Hyperparameters
SVM_KERNEL = 'rbf'                          # Radial Basis Function kernel
SVM_C = 100.0                               # Regularization parameter
SVM_GAMMA = 'scale'                         # Kernel coefficient (auto-scaled)

# k-NN Hyperparameters
KNN_NEIGHBORS = 5                           # Number of neighbors
KNN_WEIGHTS = 'distance'                    # Distance-weighted voting

# Unknown Class Handling
CONFIDENCE_THRESHOLD = 0.25                 # Reject predictions below this confidence

# Model Save Paths
SVM_MODEL_PATH = "models/svm_classifier.pkl"
KNN_MODEL_PATH = "models/knn_classifier.pkl"
SCALER_PATH = "models/feature_scaler.pkl"

# Train/Test Split
TEST_SIZE = 0.2                             # 80% training, 20% validation
RANDOM_STATE = 42                           # Reproducibility seed
```

**Why These Values:**
- **C=100:** Optimal for ResNet features (tested 100, 200, 500)
- **k=5:** Small k works best with well-separated deep learning features
- **Threshold=0.25:** ResNet features are high-quality, lower threshold acceptable
- **800 samples/class:** Balanced dataset prevents class imbalance

---

## Core Components

### 1. `feature_extraction.py`

**Purpose:** Convert images to 512-dimensional feature vectors using ResNet-18.

**Key Functions:**

#### `extract_features(image)`
```python
def extract_features(image):
    """
    Extract deep learning features using pre-trained ResNet-18.
    
    Args:
        image: PIL Image or numpy array (RGB format)
    
    Returns:
        feature_vector: 512-dimensional numpy array
    """
```

**Process:**
1. Convert numpy array → PIL Image
2. ResNet-18 forward pass (pre-trained on ImageNet)
3. Extract features from final layer (before classification)
4. Return 512-D feature vector

**Key Components:**
```python
from img2vec_pytorch import Img2Vec

# Initialize ResNet-18 (global instance)
img2vec = Img2Vec(cuda=False, model='resnet18')
```

**Parameters:**
- `cuda=False`: Use CPU (set to True if GPU available)
- `model='resnet18'`: 18-layer residual network

#### `preprocess_image(image)`
```python
def preprocess_image(image):
    """
    Preprocess image before feature extraction.
    
    Args:
        image: Input image (numpy array or PIL)
    
    Returns:
        Preprocessed image (224x224, minimal processing)
    """
```

**Steps:**
1. Resize to 224×224 (ResNet input size)
2. Minimal preprocessing (preserve features for transfer learning)

**Why Minimal Preprocessing?**
- ResNet-18 was trained on raw ImageNet images
- Over-processing (e.g., aggressive normalization) hurts transfer learning
- Let ResNet handle feature extraction

---

### 2. `data_augmentation.py`

**Purpose:** Balance dataset to 800 samples per class using augmentation techniques.

**Key Function:**

#### `augment_dataset()`
```python
def augment_dataset():
    """
    Augment dataset to TARGET_SAMPLES_PER_CLASS for each class.
    
    Techniques:
    - Rotation (±20 degrees)
    - Horizontal flip
    - Brightness adjustment (0.7-1.3x)
    - Scaling (0.8-1.2x)
    - Gaussian noise
    """
```

**Augmentation Techniques:**

1. **Rotation:** ±20 degrees
   - Simulates different object orientations
   - Waste items can appear at any angle

2. **Horizontal Flip:** Mirror image
   - Doubles dataset diversity
   - No semantic change (bottle flipped = still bottle)

3. **Brightness Adjustment:** 0.7-1.3x
   - Simulates indoor/outdoor lighting
   - Uses HSV color space (better than RGB)

4. **Scaling:** 0.8-1.2x with padding
   - Simulates objects at different distances
   - Center-crops or pads to maintain size

5. **Gaussian Noise:** Random pixel noise
   - Simulates camera sensor noise
   - Improves robustness

**Output:**
- `dataset_augmented/` with 800 images per class (4,800 total)
- Balanced dataset prevents class imbalance during training

**Statistics:**
```
Original → Augmented (Increase %)
Glass:     385 → 800 (107.8%)
Paper:     449 → 800 (78.2%)
Cardboard: 247 → 800 (223.9%)
Plastic:   363 → 800 (120.4%)
Metal:     315 → 800 (154.0%)
Trash:     106 → 800 (654.7%)
Total:   1,865 → 4,800 (257.4%)
```

---

## Training Pipeline

### 3. `train_models.py`

**Purpose:** Train SVM and k-NN classifiers on ResNet-18 features.

**Pipeline:**
```
Load Images → Extract ResNet Features → Scale Features → Train/Test Split → Train Models
```

**Key Functions:**

#### `load_dataset(dataset_path)`
```python
def load_dataset(dataset_path):
    """
    Load images and extract ResNet-18 features.
    
    Returns:
        X: Feature matrix (4800 × 512)
        y: Labels (4800,)
    """
```

**Process:**
1. Iterate through each class folder
2. Load images (JPG/PNG)
3. Extract ResNet-18 features (512-D)
4. Store in feature matrix X and label array y

#### `train_svm(X_train, y_train, X_test, y_test)`
```python
def train_svm(X_train, y_train, X_test, y_test):
    """
    Train SVM with grid search for optimal hyperparameters.
    
    Grid Search Space:
    - C: [100, 200, 500]
    - kernel: ['rbf', 'linear']
    - gamma: ['scale']
    
    Returns:
        best_svm: Trained SVM model
        best_accuracy: Validation accuracy
    """
```

**SVM Configuration:**
```python
SVC(
    kernel='rbf',           # Radial Basis Function
    C=100.0,               # Regularization
    gamma='scale',         # Auto-scale for 512 features
    probability=True,      # Enable predict_proba()
    random_state=42
)
```

**Why RBF Kernel?**
- Non-linear decision boundaries
- Maps features to infinite-dimensional space
- Works well with ResNet features

**Grid Search Results:**
```
C=100, kernel=rbf, gamma=scale → 97.08% ✅ BEST
C=200, kernel=rbf, gamma=scale → 97.08%
C=500, kernel=rbf, gamma=scale → 97.08%
C=100, kernel=linear          → 93.23%
```

#### `train_knn(X_train, y_train, X_test, y_test)`
```python
def train_knn(X_train, y_train, X_test, y_test):
    """
    Train k-NN with grid search.
    
    Grid Search Space:
    - n_neighbors: [5, 7, 9, 11, 13, 15]
    - weights: ['uniform', 'distance']
    
    Returns:
        best_knn: Trained k-NN model
        best_accuracy: Validation accuracy
    """
```

**k-NN Configuration:**
```python
KNeighborsClassifier(
    n_neighbors=5,         # Optimal for ResNet features
    weights='distance',    # Weight by inverse distance
    metric='euclidean',    # L2 distance
    n_jobs=-1             # Use all CPU cores
)
```

**Grid Search Results:**
```
k=5, weights=distance  → 94.79% ✅ BEST
k=5, weights=uniform   → 92.60%
k=7, weights=distance  → 94.37%
k=9, weights=distance  → 94.27%
```

**Feature Scaling:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why Scale?**
- SVM and k-NN are distance-based
- Features must have equal contribution
- Standardization: z = (x - μ) / σ

**Training Output:**
```
SVM Accuracy: 97.08%
k-NN Accuracy: 94.79%

Models saved to:
- models/svm_classifier.pkl
- models/knn_classifier.pkl
- models/feature_scaler.pkl
```

---

### 4. `ensemble_classifier.py`

**Purpose:** Combine SVM + k-NN using weighted voting for maximum accuracy.

**Key Function:**

#### `predict_ensemble(X_test, y_test)`
```python
def predict_ensemble(X_test, y_test):
    """
    Ensemble prediction using weighted voting.
    
    Process:
    1. Get SVM probabilities
    2. Get k-NN probabilities (from distances)
    3. Average probabilities (soft voting)
    4. Select class with highest combined probability
    
    Returns:
        predictions: Class predictions
        confidences: Confidence scores
    """
```

**Voting Strategy:**
```python
# Soft voting (weighted by confidence)
svm_proba = svm.predict_proba(X_test)      # (n_samples, 6)
knn_proba = knn.predict_proba(X_test)      # (n_samples, 6)

# Average probabilities
ensemble_proba = (svm_proba + knn_proba) / 2

# Predict class with max probability
predictions = np.argmax(ensemble_proba, axis=1)
confidences = np.max(ensemble_proba, axis=1)
```

**Results:**
```
Ensemble Accuracy: 97.71%
Average Confidence: 93.90%

Improvement:
- +0.63% over SVM alone
- +2.92% over k-NN alone
```

**Why It Works:**
- SVM and k-NN make different errors
- Ensemble corrects individual mistakes
- Combines complementary decision boundaries

---

## Evaluation and Comparison

### 5. `evaluate_model.py`

**Purpose:** Evaluate trained models on test set with detailed metrics.

**Metrics Computed:**

1. **Accuracy:** Overall correct predictions / total predictions
2. **Precision:** True positives / (True positives + False positives)
3. **Recall:** True positives / (True positives + False negatives)
4. **F1-Score:** Harmonic mean of precision and recall
5. **Confusion Matrix:** Shows which classes are confused

**Example Output:**
```
SVM Classification Report:
              precision    recall  f1-score   support
       glass       0.94      0.96      0.95       160
       paper       0.98      0.96      0.97       160
   cardboard       1.00      0.99      0.99       160
     plastic       0.98      0.98      0.98       160
       metal       0.94      0.95      0.94       160
       trash       0.99      0.99      0.99       160

    accuracy                           0.97       960
```

### 6. `compare_models.py`

**Purpose:** Compare SVM vs k-NN on speed and accuracy.

**Comparisons:**

1. **Prediction Speed:**
   - SVM: 0.85ms per sample
   - k-NN: 6.52ms per sample
   - Winner: SVM (7.7× faster)

2. **Accuracy:**
   - SVM: 97.08%
   - k-NN: 94.79%
   - Winner: SVM

3. **Per-Class F1 Scores:**
   - Shows which model performs better on each class
   - Example: SVM better on glass, k-NN better on trash

4. **Model Characteristics:**
   - Memory usage
   - Training time
   - Interpretability

---

## Deployment

### 7. `realtime_classifier.py`

**Purpose:** Live camera application for real-time waste classification.

**Architecture:**
```
Camera → BGR→RGB → ResNet-18 → Classifier → Display
```

**Key Class:**

#### `RealtimeClassifier`
```python
class RealtimeClassifier:
    def __init__(self, model_type='svm'):
        """
        Initialize real-time classifier.
        
        Args:
            model_type: 'svm' or 'knn'
        """
        # Load trained model
        self.model = joblib.load(config.SVM_MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
```

**Key Features:**

#### 1. **BGR to RGB Conversion**
```python
# OpenCV captures in BGR, ResNet trained on RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

**Critical:** Without this, colors are inverted!

#### 2. **Prediction with Rejection**
```python
def predict(self, image):
    """
    Predict with confidence threshold.
    
    Returns:
        class_id, class_name, confidence
    """
    # Extract features
    features = extract_features(image)
    features_scaled = self.scaler.transform([features])
    
    # Predict with probability
    proba = self.model.predict_proba(features_scaled)[0]
    class_id = np.argmax(proba)
    confidence = proba[class_id]
    
    # Reject if low confidence
    if confidence < config.CONFIDENCE_THRESHOLD:
        return 6, "unknown", confidence
    
    return class_id, config.CLASSES[class_id], confidence
```

#### 3. **Camera Settings**
```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
```

#### 4. **Temporal Smoothing**
```python
# Vote across last 5 frames
prediction_buffer.append(class_id)
if len(prediction_buffer) >= 5:
    from collections import Counter
    vote_counts = Counter(prediction_buffer)
    smoothed_id = vote_counts.most_common(1)[0][0]
```

**Reduces flickering between frames**

#### 5. **ROI Selection**
```python
# User can click-and-drag to define region of interest
# Focuses classification on specific object
roi = frame[y1:y2, x1:x2]
```

**Controls:**
- `q`: Quit
- `s`: Switch between SVM/k-NN
- `f`: Freeze frame
- `r`: Reset ROI
- `SPACE`: Toggle temporal smoothing

**Performance:**
- **FPS:** 10-20 (limited by ResNet inference)
- **Latency:** 50-80ms total
  - ResNet: 30-50ms
  - Classifier: 1-8ms
  - Display: ~10ms

---

### 8. `predict_hidden_dataset.py`

**Purpose:** Generate predictions for competition hidden test set.

**Usage:**
```python
python predict_hidden_dataset.py --model svm --hidden_path path/to/hidden/
```

**Output:**
```
predictions.csv:
image1.jpg,plastic
image2.jpg,glass
image3.jpg,cardboard
...
```

---

## Helper Scripts

### 9. `run_pipeline.py`

**Purpose:** Orchestrate entire pipeline (one-click execution).

**Pipeline:**
```python
# 1. Clean dataset
python clean_dataset.py

# 2. Augment data
python data_augmentation.py

# 3. Train models
python train_models.py

# 4. Evaluate models
python evaluate_model.py

# 5. Compare models
python compare_models.py

# 6. Launch real-time demo
python realtime_classifier.py
```

### 10. `test_setup.py`

**Purpose:** Verify all dependencies installed correctly.

**Checks:**
- NumPy
- scikit-learn
- OpenCV
- PyTorch
- img2vec_pytorch
- PIL/Pillow

### 11. `clean_dataset.py`

**Purpose:** Remove corrupted or unreadable images.

**Process:**
1. Scan all image files
2. Try to open with PIL
3. Remove corrupted files
4. Report statistics

### 12. `test_feature_pipeline.py`

**Purpose:** Test feature extraction on sample images.

**Verifies:**
- ResNet-18 loads correctly
- Features are 512-dimensional
- Preprocessing works
- No errors in pipeline

---

## Key Parameters Summary

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **SVM_C** | 100.0 | Regularization strength |
| **SVM_KERNEL** | 'rbf' | Non-linear decision boundaries |
| **SVM_GAMMA** | 'scale' | Auto-adjust for 512 features |
| **KNN_NEIGHBORS** | 5 | Number of nearest neighbors |
| **KNN_WEIGHTS** | 'distance' | Weight by inverse distance |
| **CONFIDENCE_THRESHOLD** | 0.25 | Reject low-confidence predictions |
| **TARGET_SAMPLES** | 800 | Balance all classes |
| **IMAGE_SIZE** | (224, 224) | ResNet-18 input size |
| **TEST_SIZE** | 0.2 | 80/20 train/test split |
| **RANDOM_STATE** | 42 | Reproducibility seed |

### Feature Extraction

| Feature | Details |
|---------|---------|
| **Method** | ResNet-18 (pre-trained on ImageNet) |
| **Input Size** | 224×224×3 (RGB) |
| **Output Size** | 512-dimensional vector |
| **Library** | img2vec_pytorch |
| **GPU Support** | Yes (automatic detection) |
| **Preprocessing** | Minimal (resize only) |

### Data Augmentation

| Technique | Parameters |
|-----------|-----------|
| **Rotation** | ±20 degrees |
| **Flip** | Horizontal only |
| **Brightness** | 0.7-1.3× (HSV space) |
| **Scaling** | 0.8-1.2× (center crop/pad) |
| **Noise** | Gaussian (σ=5) |
| **Target** | 800 samples/class |

---

## Performance Summary

### Model Accuracies

| Model | Validation Accuracy | Inference Speed | Memory |
|-------|-------------------|----------------|--------|
| **SVM** | 97.08% | 0.85ms/sample | Low |
| **k-NN** | 94.79% | 6.52ms/sample | High |
| **Ensemble** | 97.71% | ~8ms/sample | High |

### Per-Class Performance (SVM)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glass | 94% | 96% | 95% | 160 |
| Paper | 98% | 96% | 97% | 160 |
| Cardboard | 100% | 99% | 99% | 160 |
| Plastic | 98% | 98% | 98% | 160 |
| Metal | 94% | 95% | 94% | 160 |
| Trash | 99% | 99% | 99% | 160 |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Original Images | 1,865 |
| Augmented Images | 4,800 |
| Augmentation Increase | 257.4% |
| Training Samples | 3,840 (80%) |
| Validation Samples | 960 (20%) |
| Feature Dimensions | 512 |
| Classes | 6 + 1 unknown |

---

## Common Issues and Solutions

### 1. ImportError: No module named 'img2vec_pytorch'
**Solution:**
```bash
pip install img2vec-pytorch torch torchvision
```

### 2. Camera not opening
**Solution:**
```python
# Try different camera indices
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

### 3. Low accuracy on real camera
**Causes:**
- BGR not converted to RGB
- Wrong confidence threshold
- Poor lighting

**Solution:**
- Ensure BGR→RGB conversion
- Lower confidence threshold to 0.15-0.25
- Use bright, even lighting

### 4. Slow feature extraction
**Solution:**
```python
# Enable GPU if available
img2vec = Img2Vec(cuda=True, model='resnet18')
```

### 5. Out of memory during training
**Solution:**
- Process images in batches
- Use smaller image size
- Close other applications

---

## How to Use the Code

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline
python run_pipeline.py

# 3. Test camera application
python realtime_classifier.py
```

### Step-by-Step
```bash
# 1. Clean dataset
python clean_dataset.py

# 2. Augment data
python data_augmentation.py

# 3. Train models
python train_models.py

# 4. Compare models
python compare_models.py

# 5. Test real-time
python realtime_classifier.py

# 6. Generate competition predictions
python predict_hidden_dataset.py --model svm --hidden_path ./hidden_test/
```

### Testing
```bash
# Verify setup
python test_setup.py

# Test feature extraction
python test_feature_pipeline.py

# Evaluate models
python evaluate_model.py
```

---

## Future Improvements

1. **Fine-tune ResNet-18:** Train last layers on waste dataset (may reach 98-99%)
2. **Larger models:** ResNet-50 or EfficientNet-B0
3. **Object detection:** YOLO for automatic ROI detection
4. **Mobile deployment:** Convert to ONNX or TensorFlow Lite
5. **Active learning:** Collect and label misclassified examples
6. **Multi-material handling:** Segment composite objects

---

## References

1. **ResNet Paper:** He et al., "Deep Residual Learning for Image Recognition" (2016)
2. **ImageNet:** Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database" (2009)
3. **SVM:** Cortes & Vapnik, "Support-Vector Networks" (1995)
4. **k-NN:** Cover & Hart, "Nearest Neighbor Pattern Classification" (1967)
5. **scikit-learn:** Pedregosa et al., "Scikit-learn: Machine Learning in Python" (2011)
6. **PyTorch:** Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019)

---

**Last Updated:** December 26, 2025
**Project Version:** 1.0
**Author:** Machine Learning Project Team
