"""Ensemble classifier combining SVM and k-NN for maximum accuracy."""

import numpy as np
import joblib
from pathlib import Path
import config
from feature_extraction import extract_features, preprocess_image

class EnsembleClassifier:
    """Ensemble classifier that combines SVM and k-NN predictions."""
    
    def __init__(self):
        """Initialize ensemble classifier."""
        self.svm_model = None
        self.knn_model = None
        self.scaler = None
        
    def load_models(self):
        """Load trained SVM and k-NN models."""
        print("Loading models...")
        self.svm_model = joblib.load(config.SVM_MODEL_PATH)
        self.knn_model = joblib.load(config.KNN_MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        print("Models loaded successfully!")
        
    def predict(self, image, voting='weighted'):
        """
        Predict class using ensemble of SVM and k-NN.
        
        Args:
            image: Input image (numpy array or PIL image)
            voting: 'soft' (probability-weighted), 'hard' (majority vote), or 'weighted'
        
        Returns:
            predicted_class: Predicted class ID
            confidence: Prediction confidence
            individual_predictions: Dict with SVM and k-NN predictions
        """
        # Preprocess and extract features
        img_preprocessed = preprocess_image(image)
        features = extract_features(img_preprocessed)
        features_scaled = self.scaler.transform([features])
        
        # Get SVM prediction and probabilities
        svm_pred = self.svm_model.predict(features_scaled)[0]
        svm_proba = self.svm_model.predict_proba(features_scaled)[0]
        svm_confidence = np.max(svm_proba)
        
        # Get k-NN prediction and probabilities
        knn_pred = self.knn_model.predict(features_scaled)[0]
        knn_proba = self.knn_model.predict_proba(features_scaled)[0]
        knn_confidence = np.max(knn_proba)
        
        # Ensemble decision
        if voting == 'soft':
            # Average probabilities and take argmax
            avg_proba = (svm_proba + knn_proba) / 2
            ensemble_pred = np.argmax(avg_proba)
            ensemble_confidence = np.max(avg_proba)
            
        elif voting == 'weighted':
            # Weight by individual confidences (trust more confident model)
            # SVM typically more accurate, give it 60% weight
            weighted_proba = (0.6 * svm_proba + 0.4 * knn_proba)
            ensemble_pred = np.argmax(weighted_proba)
            ensemble_confidence = np.max(weighted_proba)
            
        else:  # hard voting
            # Majority vote
            if svm_pred == knn_pred:
                ensemble_pred = svm_pred
                ensemble_confidence = (svm_confidence + knn_confidence) / 2
            else:
                # Use the prediction with higher confidence
                if svm_confidence > knn_confidence:
                    ensemble_pred = svm_pred
                    ensemble_confidence = svm_confidence
                else:
                    ensemble_pred = knn_pred
                    ensemble_confidence = knn_confidence
        
        # Unknown class detection
        if ensemble_confidence < config.CONFIDENCE_THRESHOLD:
            ensemble_pred = 6  # unknown class
        
        individual_predictions = {
            'svm': {'class': svm_pred, 'confidence': svm_confidence, 'probabilities': svm_proba},
            'knn': {'class': knn_pred, 'confidence': knn_confidence, 'probabilities': knn_proba}
        }
        
        return ensemble_pred, ensemble_confidence, individual_predictions
    
    def predict_batch(self, images, voting='weighted'):
        """Predict classes for multiple images."""
        predictions = []
        for img in images:
            pred, conf, _ = self.predict(img, voting)
            predictions.append((pred, conf))
        return predictions


def evaluate_ensemble(dataset_path='dataset_augmented', voting='weighted'):
    """Evaluate ensemble classifier on test set."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from PIL import Image
    
    # Load dataset
    X = []
    y = []
    
    print(f"Loading dataset from {dataset_path}...")
    for class_id, class_name in config.CLASSES.items():
        if class_name == "unknown":
            continue
        
        class_dir = Path(dataset_path) / class_name
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"Loading {class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = Image.open(str(img_path)).convert('RGB')
                img = np.array(img)
                X.append(img)
                y.append(class_id)
            except Exception as e:
                print(f"Warning: {e}")
    
    # Split dataset (use same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    print(f"\nEvaluating on {len(X_test)} test images...")
    
    # Load ensemble classifier
    ensemble = EnsembleClassifier()
    ensemble.load_models()
    
    # Make predictions
    predictions = []
    confidences = []
    
    for img in X_test:
        pred, conf, _ = ensemble.predict(img, voting=voting)
        predictions.append(pred)
        confidences.append(conf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE CLASSIFIER EVALUATION (voting={voting})")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Confidence: {np.mean(confidences):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=[config.CLASSES[i] for i in range(7)]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    return accuracy


if __name__ == "__main__":
    import sys
    
    voting_method = sys.argv[1] if len(sys.argv) > 1 else 'weighted'
    
    if not Path(config.SVM_MODEL_PATH).exists():
        print("Error: Models not found. Run train_models.py first!")
        sys.exit(1)
    
    # Evaluate with different voting strategies
    print("Testing ensemble with weighted voting...")
    acc_weighted = evaluate_ensemble(voting='weighted')
    
    print("\n" + "="*60)
    print(f"BEST ENSEMBLE ACCURACY: {acc_weighted*100:.2f}%")
    print("="*60)
