"""Clean dataset by removing corrupted images."""

from pathlib import Path
from PIL import Image
import config

def clean_dataset():
    """Remove corrupted images from dataset."""
    print("Cleaning dataset...")
    
    corrupted = []
    
    for class_id, class_name in config.CLASSES.items():
        if class_name == "unknown":
            continue
        
        class_dir = Path(config.DATASET_PATH) / class_name
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in image_files:
            try:
                img = Image.open(str(img_path))
                img.verify()  # Verify it's a valid image
                img = Image.open(str(img_path))  # Reopen after verify
                img.load()  # Actually load the image data
            except Exception as e:
                print(f"Corrupted: {img_path.name}")
                corrupted.append(img_path)
                try:
                    img_path.unlink()  # Delete corrupted file
                except:
                    pass
    
    print(f"\nRemoved {len(corrupted)} corrupted images")
    print("Dataset cleaned!")

if __name__ == "__main__":
    clean_dataset()
