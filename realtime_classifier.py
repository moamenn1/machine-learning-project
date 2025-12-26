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
        
        # Set camera properties for better image quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting real-time classification...")
        print("Controls:")
        print("  'q' - quit")
        print("  's' - switch model")
        print("  'f' - freeze frame and classify")
        print("  'r' - draw ROI box (click and drag)")
        print("  SPACE - toggle temporal smoothing")
        
        # Colors for each class (BGR)
        colors = {
            0: (255, 200, 0),    # cardboard - cyan
            1: (0, 255, 255),    # glass - yellow
            2: (192, 192, 192),  # metal - silver
            3: (255, 255, 255),  # paper - white
            4: (0, 255, 0),      # plastic - green
            5: (0, 0, 255),      # trash - red
            6: (255, 0, 255)     # unknown - magenta
        }
        
        # Temporal smoothing buffer
        prediction_buffer = []
        buffer_size = 5
        use_smoothing = True
        
        # ROI (Region of Interest) settings
        roi_box = None
        drawing_roi = False
        roi_start = None
        
        # Mouse callback for ROI selection
        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_box, drawing_roi, roi_start
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing_roi = True
                roi_start = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing_roi = False
                if roi_start is not None:
                    roi_box = (roi_start[0], roi_start[1], x, y)
                    print(f"ROI set: {roi_box}")
        
        cv2.namedWindow('Waste Classification System')
        cv2.setMouseCallback('Waste Classification System', mouse_callback)
        
        frozen = False
        frozen_frame = None
        
        while True:
            if not frozen:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = frozen_frame.copy()
            
            # Extract ROI if set, otherwise use center region
            if roi_box is not None:
                x1, y1, x2, y2 = roi_box
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    frame_to_classify = roi
                else:
                    frame_to_classify = frame
            else:
                # Use center 50% of frame by default
                h, w = frame.shape[:2]
                margin_h, margin_w = int(h * 0.25), int(w * 0.25)
                frame_to_classify = frame[margin_h:h-margin_h, margin_w:w-margin_w]
            
            # Convert BGR to RGB (OpenCV uses BGR, PIL/training used RGB)
            frame_to_classify_rgb = cv2.cvtColor(frame_to_classify, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            class_id, class_name, confidence = self.predict(frame_to_classify_rgb)
            
            # Temporal smoothing - vote across last N frames
            if use_smoothing and not frozen:
                prediction_buffer.append(class_id)
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)
                
                # Majority voting
                if len(prediction_buffer) >= 3:
                    from collections import Counter
                    vote_counts = Counter(prediction_buffer)
                    smoothed_id = vote_counts.most_common(1)[0][0]
                    if smoothed_id != class_id:
                        class_id = smoothed_id
                        class_name = config.CLASSES[class_id] if class_id < 6 else "unknown"
            
            # Draw results
            color = colors.get(class_id, (255, 255, 255))
            
            # Draw ROI box if set
            if roi_box is not None:
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Show default center region
                h, w = frame.shape[:2]
                margin_h, margin_w = int(h * 0.25), int(w * 0.25)
                cv2.rectangle(frame, (margin_w, margin_h), (w-margin_w, h-margin_h), (128, 128, 128), 1)
            
            # Background rectangle for text
            cv2.rectangle(frame, (10, 10), (450, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (450, 150), color, 2)
            
            # Display text
            cv2.putText(frame, f"Class: {class_name.upper()}", 
                       (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Model: {self.model_type.upper()}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show status
            status = []
            if frozen:
                status.append("FROZEN")
            if use_smoothing:
                status.append(f"SMOOTH({len(prediction_buffer)})")
            if roi_box:
                status.append("ROI")
            
            if status:
                cv2.putText(frame, " | ".join(status), 
                           (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show preprocessed image being classified (small preview)
            try:
                processed = cv2.resize(frame_to_classify, (128, 128))
                preview_h, preview_w = 100, 100
                frame[10:10+preview_h, frame.shape[1]-preview_w-10:frame.shape[1]-10] = cv2.resize(processed, (preview_w, preview_h))
                cv2.rectangle(frame, (frame.shape[1]-preview_w-10, 10), (frame.shape[1]-10, 10+preview_h), (255, 255, 255), 1)
                cv2.putText(frame, "Processing", (frame.shape[1]-preview_w-10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            except:
                pass
            
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
                prediction_buffer.clear()
            elif key == ord('f'):
                # Freeze/unfreeze frame
                frozen = not frozen
                if frozen:
                    frozen_frame = frame.copy()
                    print("Frame frozen - press 'f' again to unfreeze")
                else:
                    print("Frame unfrozen")
                    prediction_buffer.clear()
            elif key == ord('r'):
                # Reset ROI
                roi_box = None
                print("ROI cleared - click and drag to set new ROI")
            elif key == ord(' '):
                # Toggle smoothing
                use_smoothing = not use_smoothing
                prediction_buffer.clear()
                print(f"Temporal smoothing: {'ON' if use_smoothing else 'OFF'}")
        
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
