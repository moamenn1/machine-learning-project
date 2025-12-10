# Setup Instructions

## Step 1: Install Required Packages

Run this command to install all dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy opencv-python scikit-learn scikit-image joblib matplotlib
```

## Step 2: Verify Setup

Run the test script to verify everything is installed correctly:

```bash
python test_setup.py
```

You should see:
```
✓ All tests passed! You're ready to start.
```

## Step 3: Run the Complete Pipeline

Option A - Automated (Recommended):
```bash
python run_pipeline.py
```

Option B - Manual Steps:
```bash
# 1. Augment dataset
python data_augmentation.py

# 2. Train models
python train_models.py

# 3. Run real-time demo
python realtime_classifier.py svm
```

## Expected Timeline

- **Setup**: 5 minutes
- **Data Augmentation**: 2-5 minutes
- **Model Training**: 3-10 minutes
- **Total**: ~15-20 minutes

## Your Dataset Status

✓ Dataset found with 1,960 images:
- Glass: 401 images
- Paper: 476 images
- Cardboard: 259 images
- Plastic: 386 images
- Metal: 328 images
- Trash: 110 images

After augmentation, each class will have ~500 images (3,000 total).

## Troubleshooting

### Issue: "No module named 'cv2'"
**Solution**: 
```bash
pip install opencv-python
```

### Issue: "No module named 'skimage'"
**Solution**: 
```bash
pip install scikit-image
```

### Issue: Camera not working
**Solution**: 
- Check if webcam is connected
- Try external USB camera
- Grant camera permissions to Python

### Issue: Low accuracy
**Solution**: 
- Ensure augmentation completed successfully
- Check dataset quality
- Adjust hyperparameters in config.py

## Project Structure

```
.
├── config.py                      # All settings and hyperparameters
├── data_augmentation.py           # Dataset balancing (→ 500 per class)
├── feature_extraction.py          # HOG + Color histogram features
├── train_models.py                # Train SVM and k-NN
├── realtime_classifier.py         # Live camera demo
├── evaluate_model.py              # Model evaluation
├── run_pipeline.py                # Run everything automatically
├── test_setup.py                  # Verify installation
├── requirements.txt               # Dependencies
├── README.md                      # Full documentation
├── QUICKSTART.md                  # Quick reference
├── TECHNICAL_REPORT_TEMPLATE.md   # Report template
└── dataset/                       # Your images
    ├── glass/
    ├── paper/
    ├── cardboard/
    ├── plastic/
    ├── metal/
    └── trash/
```

## What Gets Created

After running the pipeline:

```
dataset_augmented/          # Balanced dataset (500 per class)
models/
  ├── svm_classifier.pkl    # Trained SVM model
  ├── knn_classifier.pkl    # Trained k-NN model
  └── feature_scaler.pkl    # Feature normalization
```

## Key Configuration (config.py)

You can adjust these settings:

```python
# Data augmentation
TARGET_SAMPLES_PER_CLASS = 500

# Feature extraction
IMAGE_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
COLOR_HIST_BINS = 32

# SVM settings
SVM_KERNEL = 'rbf'
SVM_C = 10.0

# k-NN settings
KNN_NEIGHBORS = 5
KNN_WEIGHTS = 'distance'

# Unknown class threshold
CONFIDENCE_THRESHOLD = 0.6
```

## Next Steps After Setup

1. ✓ Install packages
2. ✓ Run test_setup.py
3. ✓ Run pipeline
4. Review results in terminal
5. Test real-time demo
6. Fill in TECHNICAL_REPORT_TEMPLATE.md
7. Submit models for competition

## Getting Help

- Check README.md for detailed documentation
- Check QUICKSTART.md for common commands
- Review code comments for implementation details

Good luck with your project! 🚀
