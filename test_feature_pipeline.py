"""Test and demonstrate the image-to-vector conversion pipeline."""

import numpy as np
from PIL import Image
from pathlib import Path
import config
from feature_extraction import extract_features, preprocess_image

def test_fixed_size_conversion():
    """
    Demonstrate that ANY image (regardless of original size) 
    is converted to a FIXED-SIZE feature vector.
    """
    print("="*70)
    print("IMAGE TO FIXED-SIZE VECTOR CONVERSION TEST")
    print("="*70)
    
    # Test with different image sizes
    test_sizes = [
        (50, 50),
        (100, 100),
        (200, 150),
        (300, 400),
        (640, 480),
        (1024, 768),
    ]
    
    feature_sizes = []
    
    print(f"\nTarget image size: {config.IMAGE_SIZE}")
    print(f"\nTesting conversion from various input sizes:\n")
    
    for original_size in test_sizes:
        # Create random test image
        test_img = np.random.randint(0, 255, (*original_size, 3), dtype=np.uint8)
        pil_img = Image.fromarray(test_img)
        
        # Convert to numpy array and preprocess
        img_array = np.array(pil_img)
        preprocessed = preprocess_image(img_array)
        
        # Extract features
        features = extract_features(preprocessed)
        feature_sizes.append(len(features))
        
        print(f"  Input: {original_size[0]:4d}×{original_size[1]:<4d} → "
              f"Preprocessed: {preprocessed.shape[0]}×{preprocessed.shape[1]} → "
              f"Feature Vector: {len(features)} dimensions")
    
    # Verify all feature vectors have the same size
    print(f"\n{'='*70}")
    if len(set(feature_sizes)) == 1:
        print(f"✓ SUCCESS: All images converted to FIXED-SIZE vector of {feature_sizes[0]} dimensions")
        print(f"{'='*70}")
        return True, feature_sizes[0]
    else:
        print(f"✗ FAILURE: Feature vectors have different sizes: {set(feature_sizes)}")
        print(f"{'='*70}")
        return False, None

def demonstrate_feature_breakdown():
    """Show the breakdown of the feature vector."""
    print(f"\n{'='*70}")
    print("FEATURE VECTOR BREAKDOWN")
    print(f"{'='*70}\n")
    
    # Create a test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(test_img)
    img_array = np.array(pil_img)
    preprocessed = preprocess_image(img_array)
    
    # Calculate expected sizes
    from feature_extraction import (extract_hog_features, extract_color_histogram,
                                   extract_statistical_features, extract_texture_features)
    
    hog_feat = extract_hog_features(preprocessed)
    color_feat = extract_color_histogram(preprocessed)
    stat_feat = extract_statistical_features(preprocessed)
    texture_feat = extract_texture_features(preprocessed)
    
    print(f"1. HOG Features (shape/edges):        {len(hog_feat):4d} dimensions")
    print(f"2. Color Histogram (color dist):      {len(color_feat):4d} dimensions")
    print(f"3. Statistical Features (stats):      {len(stat_feat):4d} dimensions")
    print(f"4. Texture Features (texture):        {len(texture_feat):4d} dimensions")
    print(f"   {'-'*50}")
    print(f"   TOTAL FEATURE VECTOR SIZE:         {len(hog_feat) + len(color_feat) + len(stat_feat) + len(texture_feat):4d} dimensions")
    
    print(f"\n{'='*70}")
    print("FEATURE EXTRACTION DETAILS")
    print(f"{'='*70}\n")
    
    print(f"Image preprocessing:")
    print(f"  - Original size: variable")
    print(f"  - Resized to: {config.IMAGE_SIZE}")
    print(f"  - Gaussian blur applied: 3×3 kernel")
    
    print(f"\nHOG (Histogram of Oriented Gradients):")
    print(f"  - Orientations: {config.HOG_ORIENTATIONS}")
    print(f"  - Pixels per cell: {config.HOG_PIXELS_PER_CELL}")
    print(f"  - Cells per block: {config.HOG_CELLS_PER_BLOCK}")
    print(f"  - Block normalization: L2")
    
    print(f"\nColor Histogram:")
    print(f"  - Bins per channel: {config.COLOR_HIST_BINS}")
    print(f"  - Channels: 3 (RGB)")
    print(f"  - Total bins: {config.COLOR_HIST_BINS * 3}")
    print(f"  - Normalization: L1 (sum to 1)")
    
    print(f"\nStatistical Features:")
    print(f"  - Per-channel: mean, std, median (3 channels × 3 = 9)")
    print(f"  - Grayscale: mean, std, median (3)")
    print(f"  - Color ratios: R/G, R/B, G/B (3)")
    print(f"  - Total: 15 features")
    
    print(f"\nTexture Features:")
    print(f"  - Sobel edge detection (horizontal & vertical)")
    print(f"  - Edge statistics: mean, std for each direction")
    print(f"  - Total edge strength")
    print(f"  - Total: 5 features")

def test_with_real_images():
    """Test with actual images from the dataset."""
    print(f"\n{'='*70}")
    print("TESTING WITH REAL DATASET IMAGES")
    print(f"{'='*70}\n")
    
    dataset_path = Path(config.DATASET_PATH)
    feature_sizes = []
    
    # Test one image from each class
    for class_id, class_name in config.CLASSES.items():
        if class_name == "unknown":
            continue
        
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
        
        # Get first valid image
        image_files = list(class_dir.glob("*.jpg"))[:1]
        if not image_files:
            continue
        
        try:
            img = Image.open(str(image_files[0])).convert('RGB')
            original_size = img.size
            
            img_array = np.array(img)
            preprocessed = preprocess_image(img_array)
            features = extract_features(preprocessed)
            feature_sizes.append(len(features))
            
            print(f"  {class_name:10s}: {original_size[0]:4d}×{original_size[1]:<4d} → "
                  f"Feature Vector: {len(features)} dimensions")
        except Exception as e:
            print(f"  {class_name:10s}: Error - {e}")
    
    print(f"\n{'='*70}")
    if len(set(feature_sizes)) == 1:
        print(f"✓ All real images converted to FIXED-SIZE vector of {feature_sizes[0]} dimensions")
    else:
        print(f"✗ Inconsistent feature sizes: {set(feature_sizes)}")
    print(f"{'='*70}")

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*15 + "FEATURE EXTRACTION PIPELINE TEST")
    print("="*70)
    
    # Test 1: Fixed-size conversion
    success, feature_size = test_fixed_size_conversion()
    
    if success:
        # Test 2: Feature breakdown
        demonstrate_feature_breakdown()
        
        # Test 3: Real images
        test_with_real_images()
        
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"\n✓ Pipeline successfully converts ANY image to a FIXED-SIZE vector")
        print(f"✓ Feature vector size: {feature_size} dimensions")
        print(f"✓ This fixed-size vector is used for SVM and k-NN classification")
        print(f"\nThe conversion process:")
        print(f"  1. Input: Variable-size image (any dimensions)")
        print(f"  2. Preprocessing: Resize to {config.IMAGE_SIZE}, apply blur")
        print(f"  3. Feature Extraction: HOG + Color + Stats + Texture")
        print(f"  4. Output: Fixed {feature_size}-dimensional vector")
        print(f"\n{'='*70}\n")
    else:
        print("\n✗ Pipeline test failed!")

if __name__ == "__main__":
    main()
