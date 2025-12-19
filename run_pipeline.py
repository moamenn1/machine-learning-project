"""Complete pipeline runner - executes all steps in sequence."""

import sys
from pathlib import Path

def run_pipeline():
    """Run the complete ML pipeline."""
    print("="*60)
    print("AUTOMATED MATERIAL STREAM IDENTIFICATION SYSTEM")
    print("="*60)
    
    # Step 1: Data Augmentation
    print("\n[STEP 1/3] Data Augmentation")
    print("-" * 60)
    if Path("dataset_augmented").exists():
        response = input("Augmented dataset exists. Re-run augmentation? (y/n): ")
        if response.lower() != 'y':
            print("Skipping augmentation...")
        else:
            from data_augmentation import augment_dataset
            augment_dataset()
    else:
        from data_augmentation import augment_dataset
        augment_dataset()
    
    # Step 2: Train Models
    print("\n[STEP 2/3] Model Training")
    print("-" * 60)
    if Path("models/svm_classifier.pkl").exists():
        response = input("Models exist. Re-train? (y/n): ")
        if response.lower() != 'y':
            print("Skipping training...")
            # Display saved accuracies
            accuracy_file = Path("models/accuracies.txt")
            if accuracy_file.exists():
                print("\n" + "="*50)
                print("TRAINED MODEL ACCURACIES")
                print("="*50)
                with open(accuracy_file, 'r') as f:
                    print(f.read().strip())
                print("="*50)
        else:
            from train_models import main as train_main
            train_main()
    else:
        from train_models import main as train_main
        train_main()
    
    # Step 3: Real-time Demo
    print("\n[STEP 3/3] Real-time Classification")
    print("-" * 60)
    response = input("Run real-time demo? (y/n): ")
    if response.lower() == 'y':
        from realtime_classifier import main as demo_main
        demo_main()
    else:
        print("Pipeline complete! Run 'python realtime_classifier.py' to test.")

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
