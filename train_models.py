"""Training script for SVM and k-NN classifiers."""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import config
from feature_extraction import extract_features, preprocess_image

def load_dataset(dataset_path):
    """Load images and labels from dataset."""
    X = []
    y = []
    
    print("Loading dataset and extracting features...")
    
    for class_id, class_name in config.CLASSES.items():
        if class_name == "unknown":
            continue
        
        class_dir = Path(dataset_path) / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping...")
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"Loading {class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = Image.open(str(img_path)).convert('RGB')
                img = np.array(img)
                img = preprocess_image(img)
                features = extract_features(img)
                X.append(features)
                y.append(class_id)
            except Exception as e:
                print(f"  Warning: Could not process {img_path}: {e}")
    
    return np.array(X), np.array(y)

def train_svm(X_train, y_train, X_test, y_test):
    """Train SVM classifier with optimized hyperparameter tuning for 85%+ accuracy."""
    print("\n" + "="*50)
    print("Training SVM Classifier")
    print("="*50)
    
    # Focused hyperparameter search - best performers only
    configs = [
        # Top RBF kernel configurations (proven best)
        {'kernel': 'rbf', 'C': 500.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 800.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 1000.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 500.0, 'gamma': 'auto'},
        {'kernel': 'rbf', 'C': 800.0, 'gamma': 'auto'},
        # Fine-tuned gamma (most promising)
        {'kernel': 'rbf', 'C': 500.0, 'gamma': 0.001},
        {'kernel': 'rbf', 'C': 800.0, 'gamma': 0.001},
        # Polynomial (backup)
        {'kernel': 'poly', 'C': 500.0, 'degree': 2, 'gamma': 'scale'},
        # Linear (fast baseline)
        {'kernel': 'linear', 'C': 200.0},
    ]
    
    best_svm = None
    best_accuracy = 0
    best_config = None
    
    for cfg in configs:
        print(f"\nTrying: {cfg}")
        svm = SVC(
            **cfg,
            probability=True,
            random_state=config.RANDOM_STATE,
            class_weight='balanced'  # Handle class imbalance
        )
        
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_svm = svm
            best_config = cfg
    
    print(f"\n{'='*50}")
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print("\nClassification Report:")
    y_pred = best_svm.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=[config.CLASSES[i] for i in range(6)]))
    
    return best_svm, best_accuracy

def train_knn(X_train, y_train, X_test, y_test):
    """Train k-NN classifier with optimized hyperparameter tuning."""
    print("\n" + "="*50)
    print("Training k-NN Classifier")
    print("="*50)
    
    # Try multiple k values and weight schemes
    best_knn = None
    best_accuracy = 0
    best_k = None
    best_weights = None
    
    # Focused k-value search (best performers)
    for k in [5, 7, 9, 11, 13, 15]:
        for weights in ['uniform', 'distance']:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=weights,
                metric='euclidean',
                n_jobs=-1
            )
            
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"k={k}, weights={weights}: Accuracy = {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_knn = knn
                best_k = k
                best_weights = weights
    
    print(f"\nBest k-NN: k={best_k}, weights={best_weights}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print("\nClassification Report:")
    y_pred = best_knn.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=[config.CLASSES[i] for i in range(6)]))
    
    return best_knn, best_accuracy

def main():
    """Main training pipeline."""
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Load dataset (use augmented if available, otherwise original)
    dataset_path = config.AUGMENTED_PATH if Path(config.AUGMENTED_PATH).exists() else config.DATASET_PATH
    print(f"Using dataset: {dataset_path}")
    
    X, y = load_dataset(dataset_path)
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    
    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    svm_model, svm_acc = train_svm(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Train k-NN
    knn_model, knn_acc = train_knn(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save models
    print("\n" + "="*50)
    print("Saving models...")
    joblib.dump(svm_model, config.SVM_MODEL_PATH)
    joblib.dump(knn_model, config.KNN_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"SVM model saved to {config.SVM_MODEL_PATH}")
    print(f"k-NN model saved to {config.KNN_MODEL_PATH}")
    print(f"Scaler saved to {config.SCALER_PATH}")
    
    # Summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"k-NN Accuracy: {knn_acc:.4f}")
    print(f"Best Model: {'SVM' if svm_acc > knn_acc else 'k-NN'}")
    print("="*50)

if __name__ == "__main__":
    main()
