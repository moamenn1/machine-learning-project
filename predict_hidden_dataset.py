"""Prediction script for hidden test dataset (competition submission)."""

import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import config
from feature_extraction import extract_features, preprocess_image

def predict_hidden_dataset(hidden_dir, model_path, output_file='predictions.csv', use_ensemble=False):
    """
    Generate predictions for unlabeled hidden test set.
    
    Args:
        hidden_dir: Directory containing unlabeled test images
        model_path: Path to trained model (.pkl file)
        output_file: Output CSV file for predictions
        use_ensemble: If True, use both SVM and k-NN for ensemble prediction
    
    Returns:
        DataFrame with predictions
    """
    print(f"Loading model from {model_path}...")
    
    # Load model and scaler
    if use_ensemble:
        print("Using ensemble (SVM + k-NN) for predictions...")
        svm_model = joblib.load(config.SVM_MODEL_PATH)
        knn_model = joblib.load(config.KNN_MODEL_PATH)
        model = None  # Will use both
    else:
        model = joblib.load(model_path)
    
    scaler = joblib.load(config.SCALER_PATH)
    print("Models loaded successfully!")
    
    # Get all image files
    hidden_path = Path(hidden_dir)
    if not hidden_path.exists():
        raise FileNotFoundError(f"Hidden dataset directory not found: {hidden_dir}")
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(hidden_path.glob(ext)))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {hidden_dir}")
    
    print(f"Found {len(image_files)} images to classify")
    
    # Process each image
    results = []
    
    for img_path in tqdm(image_files, desc="Classifying images"):
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_array = np.array(image)
            
            # Extract features
            preprocessed = preprocess_image(image_array)
            features = extract_features(preprocessed)
            features_scaled = scaler.transform([features])
            
            # Make prediction
            if use_ensemble:
                # Ensemble prediction (60% SVM + 40% k-NN)
                svm_proba = svm_model.predict_proba(features_scaled)[0]
                knn_proba = knn_model.predict_proba(features_scaled)[0]
                
                # Weighted voting
                ensemble_proba = 0.6 * svm_proba + 0.4 * knn_proba
                class_id = np.argmax(ensemble_proba)
                confidence = ensemble_proba[class_id]
            else:
                # Single model prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    class_id = np.argmax(proba)
                    confidence = proba[class_id]
                else:
                    # k-NN without probability
                    class_id = model.predict(features_scaled)[0]
                    distances, _ = model.kneighbors(features_scaled)
                    confidence = 1.0 / (1.0 + np.mean(distances))
            
            # Get class name
            if confidence < config.CONFIDENCE_THRESHOLD:
                class_name = "unknown"
            else:
                class_name = config.CLASSES[class_id]
            
            # Store result
            results.append({
                'filename': img_path.name,
                'predicted_class': class_name,
                'class_id': int(class_id),
                'confidence': float(confidence)
            })
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            results.append({
                'filename': img_path.name,
                'predicted_class': 'error',
                'class_id': -1,
                'confidence': 0.0
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Print summary statistics
    print("\n=== Prediction Summary ===")
    print(f"Total images: {len(df)}")
    print(f"\nClass distribution:")
    print(df['predicted_class'].value_counts())
    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    print(f"Min confidence: {df['confidence'].min():.3f}")
    print(f"Max confidence: {df['confidence'].max():.3f}")
    
    # Low confidence predictions
    low_conf = df[df['confidence'] < 0.7]
    if len(low_conf) > 0:
        print(f"\nWarning: {len(low_conf)} predictions with confidence < 0.7:")
        print(low_conf[['filename', 'predicted_class', 'confidence']].to_string())
    
    return df

def main():
    """Main entry point for competition submission."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict classes for hidden test dataset')
    parser.add_argument('--input', type=str, default='hidden_test_dataset',
                        help='Directory containing unlabeled test images')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file for predictions')
    parser.add_argument('--model', type=str, default='svm',
                        choices=['svm', 'knn', 'ensemble'],
                        help='Model to use for predictions (svm, knn, or ensemble)')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model == 'ensemble':
        use_ensemble = True
        model_path = config.SVM_MODEL_PATH  # Will use both in function
    elif args.model == 'svm':
        use_ensemble = False
        model_path = config.SVM_MODEL_PATH
    else:  # knn
        use_ensemble = False
        model_path = config.KNN_MODEL_PATH
    
    # Check if models exist
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run train_models.py first to train the models.")
        return
    
    if not Path(config.SCALER_PATH).exists():
        print(f"Error: Scaler not found at {config.SCALER_PATH}")
        print("Please run train_models.py first.")
        return
    
    # Generate predictions
    print(f"\n{'='*50}")
    print(f"Hidden Dataset Prediction System")
    print(f"Model: {args.model.upper()}")
    print(f"{'='*50}\n")
    
    try:
        df = predict_hidden_dataset(
            hidden_dir=args.input,
            model_path=model_path,
            output_file=args.output,
            use_ensemble=use_ensemble
        )
        
        print(f"\n{'='*50}")
        print(f"SUCCESS! Predictions ready for competition submission")
        print(f"File: {args.output}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nUsage examples:")
        print("  python predict_hidden_dataset.py --input hidden_test_dataset --output predictions.csv")
        print("  python predict_hidden_dataset.py --input test_images --model ensemble")

if __name__ == "__main__":
    main()
