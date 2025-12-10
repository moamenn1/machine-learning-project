"""Model evaluation and comparison script."""

import numpy as np
import joblib
import cv2
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import config
from feature_extraction import extract_features, preprocess_image

def load_test_data(dataset_path):
    """Load test dataset."""
    X = []
    y = []
    
    for class_id, class_name in config.CLASSES.items():
        if class_name == "unknown":
            continue
        
        class_dir = Path(dataset_path) / class_name
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = preprocess_image(img)
                features = extract_features(img)
                X.append(features)
                y.append(class_id)
    
    return np.array(X), np.array(y)

def evaluate_models():
    """Evaluate both SVM and k-NN models."""
    print("Loading models...")
    svm_model = joblib.load(config.SVM_MODEL_PATH)
    knn_model = joblib.load(config.KNN_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    
    print("Loading test data...")
    dataset_path = config.AUGMENTED_PATH if Path(config.AUGMENTED_PATH).exists() else config.DATASET_PATH
    X, y = load_test_data(dataset_path)
    X_scaled = scaler.transform(X)
    
    print(f"Test samples: {len(X)}")
    
    # Evaluate SVM
    print("\n" + "="*60)
    print("SVM EVALUATION")
    print("="*60)
    y_pred_svm = svm_model.predict(X_scaled)
    print(classification_report(y, y_pred_svm, 
                                target_names=[config.CLASSES[i] for i in range(6)]))
    
    # Evaluate k-NN
    print("\n" + "="*60)
    print("k-NN EVALUATION")
    print("="*60)
    y_pred_knn = knn_model.predict(X_scaled)
    print(classification_report(y, y_pred_knn, 
                                target_names=[config.CLASSES[i] for i in range(6)]))

if __name__ == "__main__":
    evaluate_models()
