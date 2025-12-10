"""Data augmentation module to balance and expand the dataset."""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import config

def augment_image(pil_image):
    """Apply random augmentation to a PIL image with more diverse techniques."""
    augmentations = []
    
    # 1. Rotation (various angles)
    for angle in [-30, -15, 15, 30]:
        augmentations.append(pil_image.rotate(angle, fillcolor=(128, 128, 128)))
    
    # 2. Horizontal flip
    augmentations.append(pil_image.transpose(Image.FLIP_LEFT_RIGHT))
    
    # 3. Vertical flip
    augmentations.append(pil_image.transpose(Image.FLIP_TOP_BOTTOM))
    
    # 4. Brightness adjustment (multiple levels)
    for factor in [0.6, 0.8, 1.2, 1.4]:
        enhancer = ImageEnhance.Brightness(pil_image)
        augmentations.append(enhancer.enhance(factor))
    
    # 5. Contrast adjustment
    for factor in [0.7, 0.85, 1.15, 1.3]:
        enhancer = ImageEnhance.Contrast(pil_image)
        augmentations.append(enhancer.enhance(factor))
    
    # 6. Saturation adjustment
    for factor in [0.7, 1.3]:
        enhancer = ImageEnhance.Color(pil_image)
        augmentations.append(enhancer.enhance(factor))
    
    # 7. Sharpness adjustment
    for factor in [0.5, 1.5]:
        enhancer = ImageEnhance.Sharpness(pil_image)
        augmentations.append(enhancer.enhance(factor))
    
    # 8. Scaling
    w, h = pil_image.size
    for scale in [0.75, 0.85, 1.15, 1.25]:
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        if scale > 1:
            # Crop center
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            augmentations.append(scaled.crop((left, top, left + w, top + h)))
        else:
            # Pad
            canvas = Image.new('RGB', (w, h), (128, 128, 128))
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            canvas.paste(scaled, (left, top))
            augmentations.append(canvas)
    
    # 9. Gaussian blur (slight)
    augmentations.append(pil_image.filter(ImageFilter.GaussianBlur(radius=1)))
    
    # 10. Slight edge enhancement
    augmentations.append(pil_image.filter(ImageFilter.EDGE_ENHANCE))
    
    # 11. Add Gaussian noise
    img_array = np.array(pil_image)
    for noise_level in [8, 15]:
        noise = np.random.normal(0, noise_level, img_array.shape).astype(np.int16)
        noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        augmentations.append(Image.fromarray(noisy))
    
    # 12. Combined transformations (rotation + brightness)
    rotated = pil_image.rotate(np.random.randint(-25, 25), fillcolor=(128, 128, 128))
    enhancer = ImageEnhance.Brightness(rotated)
    augmentations.append(enhancer.enhance(np.random.uniform(0.75, 1.25)))
    
    # 13. Horizontal flip + contrast
    flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Contrast(flipped)
    augmentations.append(enhancer.enhance(np.random.uniform(0.8, 1.2)))
    
    # 14. Color jitter (slight random color shifts)
    img_array = np.array(pil_image).astype(np.float32)
    for _ in range(2):
        jitter = np.random.uniform(0.9, 1.1, 3)
        jittered = np.clip(img_array * jitter, 0, 255).astype(np.uint8)
        augmentations.append(Image.fromarray(jittered))
    
    # Return random augmentation
    idx = np.random.randint(0, len(augmentations))
    return augmentations[idx]

def augment_dataset():
    """Augment dataset to balance classes."""
    print("Starting data augmentation...")
    
    # Create output directory
    Path(config.AUGMENTED_PATH).mkdir(exist_ok=True)
    
    for class_id, class_name in config.CLASSES.items():
        if class_name == "unknown":
            continue
            
        input_dir = Path(config.DATASET_PATH) / class_name
        output_dir = Path(config.AUGMENTED_PATH) / class_name
        output_dir.mkdir(exist_ok=True)
        
        if not input_dir.exists():
            print(f"Warning: {input_dir} does not exist, skipping...")
            continue
        
        # Get all images
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        current_count = len(image_files)
        
        print(f"{class_name}: {current_count} images")
        
        # Copy original images
        for idx, img_path in enumerate(image_files):
            try:
                img = Image.open(str(img_path)).convert('RGB')
                img.save(str(output_dir / f"orig_{idx:04d}.jpg"))
            except Exception as e:
                print(f"  Warning: Could not process {img_path}: {e}")
        
        # Calculate how many augmented images needed
        target = config.TARGET_SAMPLES_PER_CLASS
        needed = max(0, target - current_count)
        
        if needed > 0:
            print(f"  Generating {needed} augmented images...")
            for i in range(needed):
                # Randomly select source image
                src_img_path = np.random.choice(image_files)
                try:
                    img = Image.open(str(src_img_path)).convert('RGB')
                    aug_img = augment_image(img)
                    aug_img.save(str(output_dir / f"aug_{i:04d}.jpg"))
                except Exception as e:
                    print(f"  Warning: Could not augment image: {e}")
        
        final_count = len(list(output_dir.glob("*.jpg")))
        print(f"  Final count: {final_count} images")
    
    print("Data augmentation complete!")

if __name__ == "__main__":
    augment_dataset()
