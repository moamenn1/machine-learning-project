"""Configuration file for the MSI system."""

# Dataset paths
DATASET_PATH = "dataset"
AUGMENTED_PATH = "dataset_augmented"

# Class definitions
CLASSES = {
    0: "glass",
    1: "paper",
    2: "cardboard",
    3: "plastic",
    4: "metal",
    5: "trash",
    6: "unknown"
}

CLASS_TO_ID = {v: k for k, v in CLASSES.items()}

# Data augmentation settings
TARGET_SAMPLES_PER_CLASS = 800  # More data for better training (60%+ increase from ~500)
AUGMENTATION_FACTOR = 1.6  # 60% increase for better generalization

# Feature extraction settings (optimized for speed and accuracy)
HOG_ORIENTATIONS = 9  # Standard 9 orientations (was 12)
HOG_PIXELS_PER_CELL = (16, 16)  # Larger cells = fewer features (was 8x8)
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_HIST_BINS = 32  # Reduced bins for speed (was 64)
IMAGE_SIZE = (128, 128)  # Standard size for feature extraction

# LBP settings for texture features
LBP_RADIUS = 3
LBP_POINTS = 24

# Model settings - Optimized for 85%+ accuracy
SVM_KERNEL = 'rbf'
SVM_C = 100.0  # Higher C for better fit
SVM_GAMMA = 'scale'  # Auto-scale based on features
SVM_DEGREE = 3  # For polynomial kernel
KNN_NEIGHBORS = 7  # Better k value for this dataset
KNN_WEIGHTS = 'distance'

# Rejection threshold for unknown class
CONFIDENCE_THRESHOLD = 0.25  # Very low to force predictions

# Model save paths
SVM_MODEL_PATH = "models/svm_classifier.pkl"
KNN_MODEL_PATH = "models/knn_classifier.pkl"
SCALER_PATH = "models/feature_scaler.pkl"

# Train/validation split
TEST_SIZE = 0.2
RANDOM_STATE = 42
