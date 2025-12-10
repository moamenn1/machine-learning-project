"""Real-time classification application using webcam."""

import cv2
import numpy as np
import joblib
from pathlib import Path
import config
from feature_extraction import extract_features, preprocess_image

class RealtimeClassifier:
    """Real-time waste classification system."""
    
    def __init__(self, model_type='svm'):
        """
        Initialize classifier.
        
        Args:
            model_type: 'svm' or 'knn'
        """
        self.model_type = model_type
        
        # Load model and scaler
        if model_type == 'svm':
            self.model = joblib.load(config.SVM_MODEL_PATH)
        else:
            self.model = joblib.load(config.KNN_MODEL_PATH)
        
        self.scaler = joblib.load(config.SCALER_PATH)
        print(f"Loaded {model_type.upper()} model successfully")
    
    def predict(self, image):
        """
        Predict class for an image with rejection mechanism.
        
        Returns:
            class_id, class_name, confidence
        """
        # Preprocess and extract features
        img = preprocess_image(image)
        features = extract_features(img)
        features_scaled = self.scaler.transform([features])
        
        # Get prediction with probability
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            class_id = np.argmax(proba)
            confidence = proba[class_id]
        else:
            # For k-NN without probability
            class_id = self.model.predict(features_scaled)[0]
            # Use distance-based confidence
            distances, indices = self.model.kneighbors(features_scaled)
            confidence = 1.0 / (1.0 + np.mean(distances))
        
        # Rejection mechanism for unknown class
        if confidence < config.CONFIDENCE_THRESHOLD:
            class_id = 6  # Unknown
            class_name = "unknown"
        else:
            class_name = config.CLASSES[class_id]
        
        return class_id, class_name, confidence
    
    def run(self):
        """Run real-time classification on webcam feed."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting real-time classification...")
        print("Press 'q' to quit, 's' to switch model")
        
        # Colors for each class (BGR)
        colors = {
            0: (255, 200, 0),    # glass - cyan
            1: (255, 255, 255),  # paper - white
            2: (0, 165, 255),    # cardboard - orange
            3: (0, 255, 0),      # plastic - green
            4: (128, 128, 128),  # metal - gray
            5: (0, 0, 255),      # trash - red
            6: (255, 0, 255)     # unknown - magenta
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make prediction
            class_id, class_name, confidence = self.predict(frame)
            
            # Draw results
            color = colors.get(class_id, (255, 255, 255))
            
            # Background rectangle for text
            cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 120), color, 2)
            
            # Display text
            cv2.putText(frame, f"Class: {class_name.upper()}", 
                       (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Model: {self.model_type.upper()}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Waste Classification System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Switch model
                self.model_type = 'knn' if self.model_type == 'svm' else 'svm'
                if self.model_type == 'svm':
                    self.model = joblib.load(config.SVM_MODEL_PATH)
                else:
                    self.model = joblib.load(config.KNN_MODEL_PATH)
                print(f"Switched to {self.model_type.upper()} model")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main entry point."""
    import sys
    
    # Check if models exist
    if not Path(config.SVM_MODEL_PATH).exists():
        print("Error: Models not found. Please run train_models.py first.")
        return
    
    # Get model type from command line
    model_type = 'svm'
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type not in ['svm', 'knn']:
            print("Invalid model type. Use 'svm' or 'knn'")
            return
    
    # Run classifier
    classifier = RealtimeClassifier(model_type=model_type)
    classifier.run()

if __name__ == "__main__":
    main()
