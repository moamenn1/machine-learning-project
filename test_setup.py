"""Test script to verify setup and dependencies."""

import sys

def test_imports():
    """Test if all required packages are installed."""
    print("Testing package imports...")
    
    packages = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'joblib': 'joblib'
    }
    
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✓ All packages installed!")
    return True

def test_dataset():
    """Test if dataset exists and is accessible."""
    print("\nTesting dataset...")
    from pathlib import Path
    import config
    
    dataset_path = Path(config.DATASET_PATH)
    if not dataset_path.exists():
        print(f"  ✗ Dataset not found at {dataset_path}")
        return False
    
    total_images = 0
    for class_id, class_name in config.CLASSES.items():
        if class_name == "unknown":
            continue
        
        class_dir = dataset_path / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
            print(f"  ✓ {class_name}: {count} images")
            total_images += count
        else:
            print(f"  ✗ {class_name}: directory not found")
    
    print(f"\n✓ Total images: {total_images}")
    return total_images > 0

def test_feature_extraction():
    """Test feature extraction on a dummy image."""
    print("\nTesting feature extraction...")
    import numpy as np
    from feature_extraction import extract_features
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        features = extract_features(dummy_image)
        print(f"  ✓ Feature extraction works")
        print(f"  ✓ Feature vector shape: {features.shape}")
        print(f"  ✓ Feature vector length: {len(features)}")
        return True
    except Exception as e:
        print(f"  ✗ Feature extraction failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("SETUP VERIFICATION TEST")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test dataset
    results.append(("Dataset", test_dataset()))
    
    # Test feature extraction
    results.append(("Feature Extraction", test_feature_extraction()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to start.")
        print("\nNext steps:")
        print("  1. Run: python run_pipeline.py")
        print("  2. Or follow steps in QUICKSTART.md")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
