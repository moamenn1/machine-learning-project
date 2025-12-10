# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run complete pipeline**:
   ```bash
   python run_pipeline.py
   ```
   
   This will:
   - Augment your dataset to 500 samples per class
   - Train both SVM and k-NN models
   - Launch the real-time demo

## Individual Steps

### Data Augmentation Only
```bash
python data_augmentation.py
```

### Training Only
```bash
python train_models.py
```

### Real-time Demo Only
```bash
# SVM model
python realtime_classifier.py svm

# k-NN model
python realtime_classifier.py knn
```

### Evaluation Only
```bash
python evaluate_model.py
```

## Expected Results

- **Data Augmentation**: ~500 images per class (6 classes)
- **Training Time**: 2-5 minutes depending on hardware
- **Validation Accuracy**: Target >85%
- **Real-time FPS**: 10-30 FPS depending on hardware

## Troubleshooting

**Issue**: "No module named 'cv2'"
- **Solution**: `pip install opencv-python`

**Issue**: "Camera not found"
- **Solution**: Check webcam connection or use external USB camera

**Issue**: Low accuracy (<70%)
- **Solution**: 
  - Ensure data augmentation completed successfully
  - Check if dataset has sufficient samples
  - Try adjusting hyperparameters in config.py

**Issue**: Slow real-time processing
- **Solution**: 
  - Reduce IMAGE_SIZE in config.py (e.g., to 64x64)
  - Use k-NN model (faster prediction)
  - Close other applications

## Next Steps

1. Review the technical report template
2. Experiment with different feature extraction methods
3. Tune hyperparameters in config.py
4. Test with hidden dataset for competition
5. Document your findings

## Key Files

- `config.py` - Adjust all hyperparameters here
- `feature_extraction.py` - Modify feature extraction methods
- `train_models.py` - Main training logic
- `realtime_classifier.py` - Demo application

## Tips for Better Performance

1. **Feature Engineering**:
   - Try different HOG parameters
   - Add texture features (LBP, Gabor filters)
   - Experiment with different color spaces (HSV, LAB)

2. **Model Tuning**:
   - Grid search for optimal C and gamma (SVM)
   - Try different k values (k-NN)
   - Experiment with different kernels

3. **Data Quality**:
   - Ensure augmented images look realistic
   - Balance classes properly
   - Remove corrupted images

Good luck with your project!
