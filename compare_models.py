"""Detailed comparison between SVM and k-NN classifiers."""

import numpy as np
import joblib
import time
from pathlib import Path
import config
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2
from feature_extraction import extract_features, preprocess_image

def load_test_samples(n_samples=100):
    """Load a subset of test samples."""
    X = []
    y = []
    
    for class_id, class_name in config.CLASSES.items():
        if class_name == "unknown":
            continue
        
        dataset_path = config.AUGMENTED_PATH if Path(config.AUGMENTED_PATH).exists() else config.DATASET_PATH
        class_dir = Path(dataset_path) / class_name
        
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob("*.jpg"))[:n_samples//6]
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = preprocess_image(img)
                features = extract_features(img)
                X.append(features)
                y.append(class_id)
    
    return np.array(X), np.array(y)

def compare_models():
    """Compare SVM and k-NN models."""
    print("="*70)
    print("MODEL COMPARISON: SVM vs k-NN")
    print("="*70)
    
    # Load models
    print("\nLoading models...")
    svm_model = joblib.load(config.SVM_MODEL_PATH)
    knn_model = joblib.load(config.KNN_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    
    # Load test data
    print("Loading test samples...")
    X, y = load_test_samples(n_samples=300)
    X_scaled = scaler.transform(X)
    print(f"Test samples: {len(X)}")
    
    # Compare prediction time
    print("\n" + "-"*70)
    print("PREDICTION SPEED")
    print("-"*70)
    
    # SVM timing
    start = time.time()
    y_pred_svm = svm_model.predict(X_scaled)
    svm_time = time.time() - start
    svm_time_per_sample = (svm_time / len(X)) * 1000  # ms
    
    # k-NN timing
    start = time.time()
    y_pred_knn = knn_model.predict(X_scaled)
    knn_time = time.time() - start
    knn_time_per_sample = (knn_time / len(X)) * 1000  # ms
    
    print(f"SVM:  {svm_time:.4f}s total, {svm_time_per_sample:.2f}ms per sample")
    print(f"k-NN: {knn_time:.4f}s total, {knn_time_per_sample:.2f}ms per sample")
    print(f"Winner: {'SVM' if svm_time < knn_time else 'k-NN'} ({abs(svm_time - knn_time):.4f}s faster)")
    
    # Compare accuracy
    print("\n" + "-"*70)
    print("ACCURACY")
    print("-"*70)
    
    svm_acc = accuracy_score(y, y_pred_svm)
    knn_acc = accuracy_score(y, y_pred_knn)
    
    print(f"SVM:  {svm_acc:.4f} ({svm_acc*100:.2f}%)")
    print(f"k-NN: {knn_acc:.4f} ({knn_acc*100:.2f}%)")
    print(f"Winner: {'SVM' if svm_acc > knn_acc else 'k-NN'} ({abs(svm_acc - knn_acc)*100:.2f}% better)")
    
    # Per-class performance
    print("\n" + "-"*70)
    print("PER-CLASS PERFORMANCE")
    print("-"*70)
    
    svm_prec, svm_rec, svm_f1, _ = precision_recall_fscore_support(y, y_pred_svm, average=None, zero_division=0)
    knn_prec, knn_rec, knn_f1, _ = precision_recall_fscore_support(y, y_pred_knn, average=None, zero_division=0)
    
    print(f"\n{'Class':<12} {'SVM F1':<10} {'k-NN F1':<10} {'Winner'}")
    print("-"*70)
    
    for i in range(6):
        class_name = config.CLASSES[i]
        winner = "SVM" if svm_f1[i] > knn_f1[i] else "k-NN" if knn_f1[i] > svm_f1[i] else "Tie"
        print(f"{class_name:<12} {svm_f1[i]:.4f}     {knn_f1[i]:.4f}     {winner}")
    
    # Model characteristics
    print("\n" + "-"*70)
    print("MODEL CHARACTERISTICS")
    print("-"*70)
    
    print(f"\n{'Characteristic':<25} {'SVM':<20} {'k-NN'}")
    print("-"*70)
    print(f"{'Training Required':<25} {'Yes':<20} {'No (lazy learning)'}")
    print(f"{'Memory Usage':<25} {'Low (support vectors)':<20} {'High (all data)'}")
    print(f"{'Interpretability':<25} {'Low':<20} {'High'}")
    print(f"{'Hyperparameters':<25} {'C, gamma, kernel':<20} {'k, weights, metric'}")
    print(f"{'Handles Non-linear':<25} {'Yes (kernel trick)':<20} {'Yes (local decision)'}")
    print(f"{'Scalability':<25} {'Good':<20} {'Poor (large datasets)'}")
    
    # Overall recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if svm_acc > knn_acc and svm_time < knn_time * 2:
        print("\n✓ SVM is recommended:")
        print("  - Higher accuracy")
        print("  - Comparable or better speed")
        print("  - Lower memory footprint")
    elif knn_acc > svm_acc:
        print("\n✓ k-NN is recommended:")
        print("  - Higher accuracy")
        print("  - More interpretable")
        print("  - No training required")
    else:
        print("\n✓ Both models perform similarly:")
        print("  - Use SVM for production (lower memory)")
        print("  - Use k-NN for quick prototyping")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        compare_models()
    except FileNotFoundError:
        print("Error: Models not found. Please run train_models.py first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
