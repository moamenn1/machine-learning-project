# Technical Report: Automated Material Stream Identification System

**Team Name**: [Your Team Name]  
**Date**: [Date]  
**Course**: [Course Code]

---

## 1. Executive Summary

[Brief overview of the project, approach, and key results - 1 paragraph]

---

## 2. Introduction

### 2.1 Problem Statement
The goal is to develop an automated waste classification system that can identify 6 material types (glass, paper, cardboard, plastic, metal, trash) plus an "unknown" class for out-of-distribution samples.

### 2.2 Approach Overview
[Describe your overall approach in 2-3 sentences]

---

## 3. Data Augmentation

### 3.1 Original Dataset Statistics

| Class | Original Count |
|-------|----------------|
| Glass | 401 |
| Paper | 476 |
| Cardboard | 259 |
| Plastic | 386 |
| Metal | 328 |
| Trash | 110 |
| **Total** | **1,960** |

### 3.2 Augmentation Techniques

**Techniques Applied**:
1. **Rotation**: Random rotation ±20 degrees
   - Justification: Waste items can appear at any orientation
   
2. **Horizontal Flip**: Mirror image
   - Justification: Increases dataset diversity, no semantic change
   
3. **Brightness Adjustment**: 0.7-1.3x in HSV space
   - Justification: Simulates different lighting conditions
   
4. **Scaling**: 0.8-1.2x with center crop/padding
   - Justification: Objects appear at different distances from camera
   
5. **Gaussian Noise**: Random noise addition
   - Justification: Simulates camera sensor noise

### 3.3 Augmented Dataset Statistics

| Class | Augmented Count | Increase |
|-------|-----------------|----------|
| Glass | 500 | 24.7% |
| Paper | 500 | 5.0% |
| Cardboard | 500 | 93.1% |
| Plastic | 500 | 29.5% |
| Metal | 500 | 52.4% |
| Trash | 500 | 354.5% |
| **Total** | **3,000** | **53.1%** |

**Result**: Achieved >30% augmentation requirement and balanced all classes.

---

## 4. Feature Extraction

### 4.1 Feature Extraction Pipeline

**Image → Preprocessing → Feature Extraction → Feature Vector (1D array)**

### 4.2 Preprocessing Steps

1. **Resize**: 128×128 pixels (standardization)
2. **Gaussian Blur**: 3×3 kernel (noise reduction)

### 4.3 Feature Descriptors

#### 4.3.1 HOG (Histogram of Oriented Gradients)

**Parameters**:
- Orientations: 9
- Pixels per cell: 8×8
- Cells per block: 2×2
- Block normalization: L2-Hys

**Justification**:
- Captures shape and edge information
- Invariant to illumination changes
- Effective for object recognition
- Widely used in computer vision (e.g., pedestrian detection)

**Feature Dimension**: ~1,700 features

#### 4.3.2 Color Histogram

**Parameters**:
- Color space: BGR
- Bins per channel: 32
- Total bins: 96 (32×3)
- Normalization: L1 (sum to 1)

**Justification**:
- Material types have distinct color characteristics
  - Glass: transparent, green, brown
  - Paper: white, beige
  - Metal: gray, silver
  - Plastic: various bright colors
- Complements HOG's shape information
- Computationally efficient

**Feature Dimension**: 96 features

#### 4.3.3 Combined Feature Vector

**Total Dimension**: ~1,800 features  
**Combination Strategy**: Concatenation of HOG + Color Histogram

**Rationale**: Combines structural (HOG) and appearance (color) information for comprehensive material representation.

---

## 5. Classification Models

### 5.1 Support Vector Machine (SVM)

#### 5.1.1 Architecture

**Hyperparameters**:
- Kernel: RBF (Radial Basis Function)
- C (Regularization): 10.0
- Gamma: 'scale' (1 / (n_features × X.var()))
- Probability: Enabled

#### 5.1.2 Kernel Choice Justification

**RBF Kernel Selected** because:
1. Non-linear decision boundaries (materials not linearly separable)
2. Handles high-dimensional feature space well
3. Only 2 hyperparameters to tune (C, gamma)
4. Proven effective for image classification tasks

**Alternatives Considered**:
- Linear: Too simple for complex material patterns
- Polynomial: More hyperparameters, risk of overfitting
- Sigmoid: Can behave like neural network but less stable

#### 5.1.3 Hyperparameter Selection

**C = 10.0**:
- Controls regularization strength
- Higher C → less regularization → fits training data closely
- Chosen through validation performance

**Gamma = 'scale'**:
- Controls RBF kernel width
- 'scale' automatically adjusts based on feature variance
- Prevents manual tuning

### 5.2 k-Nearest Neighbors (k-NN)

#### 5.2.1 Architecture

**Hyperparameters**:
- k (neighbors): 5
- Weights: Distance-based
- Metric: Euclidean distance

#### 5.2.2 Design Choices

**k = 5**:
- Odd number prevents ties
- Not too small (noise sensitive) or too large (oversmoothing)
- Balances bias-variance tradeoff

**Distance-based Weighting**:
- Closer neighbors have more influence
- Formula: weight = 1 / distance
- More robust than uniform weighting

**Euclidean Distance**:
- Natural choice for continuous feature vectors
- Computationally efficient
- Works well with scaled features

### 5.3 Feature Scaling

**Method**: StandardScaler (z-score normalization)

**Formula**: z = (x - μ) / σ

**Justification**:
- Essential for distance-based methods (k-NN, RBF SVM)
- Prevents features with large ranges from dominating
- Improves convergence and performance

---

## 6. Unknown Class Handling

### 6.1 Rejection Mechanism

**Confidence Threshold**: 0.6

**SVM Approach**:
- Uses `predict_proba()` for probability estimates
- If max(probability) < 0.6 → classify as "unknown"

**k-NN Approach**:
- Computes confidence from neighbor distances
- Confidence = 1 / (1 + mean_distance)
- If confidence < 0.6 → classify as "unknown"

### 6.2 Justification

- Prevents misclassification of out-of-distribution items
- Threshold of 0.6 balances precision and recall
- Critical for real-world deployment (handles unexpected objects)

---

## 7. Experimental Results

### 7.1 Training Configuration

- **Training Set**: 80% (2,400 samples)
- **Validation Set**: 20% (600 samples)
- **Stratified Split**: Maintains class distribution
- **Random State**: 42 (reproducibility)

### 7.2 Performance Metrics

#### SVM Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | [Fill in] |
| Training Time | [Fill in] |
| Prediction Time (per image) | [Fill in] |

**Classification Report**:
```
[Paste classification report here]
```

**Confusion Matrix**:
```
[Paste confusion matrix here]
```

#### k-NN Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | [Fill in] |
| Training Time | [Fill in] |
| Prediction Time (per image) | [Fill in] |

**Classification Report**:
```
[Paste classification report here]
```

**Confusion Matrix**:
```
[Paste confusion matrix here]
```

### 7.3 Model Comparison

| Aspect | SVM | k-NN | Winner |
|--------|-----|------|--------|
| Accuracy | [X.XX] | [X.XX] | [Model] |
| Training Time | [X.X]s | [X.X]s | [Model] |
| Prediction Time | [X.X]ms | [X.X]ms | [Model] |
| Memory Usage | Low | High | SVM |
| Interpretability | Low | High | k-NN |

### 7.4 Analysis

**SVM Strengths**:
- [Analyze based on your results]

**SVM Weaknesses**:
- [Analyze based on your results]

**k-NN Strengths**:
- [Analyze based on your results]

**k-NN Weaknesses**:
- [Analyze based on your results]

**Best Model**: [SVM/k-NN] because [justification]

---

## 8. Real-Time System Deployment

### 8.1 Implementation

- **Framework**: OpenCV for camera capture
- **Processing Pipeline**: Frame → Preprocess → Extract Features → Classify → Display
- **Performance**: [X] FPS on [hardware specs]

### 8.2 User Interface

- Real-time classification display
- Confidence score visualization
- Color-coded class indicators
- Model switching capability (press 's')

### 8.3 Challenges and Solutions

**Challenge 1**: [Describe]
- **Solution**: [Describe]

**Challenge 2**: [Describe]
- **Solution**: [Describe]

---

## 9. Competition Results

### 9.1 Hidden Test Set Performance

| Model | Accuracy | Rank |
|-------|----------|------|
| SVM | [X.XX] | [X] |
| k-NN | [X.XX] | [X] |

### 9.2 Error Analysis

[Analyze misclassifications, common errors, potential improvements]

---

## 10. Conclusion

### 10.1 Summary

[Summarize key achievements and findings]

### 10.2 Future Improvements

1. **Feature Engineering**:
   - Add texture features (LBP, Gabor filters)
   - Experiment with deep features (pre-trained CNN)
   
2. **Model Enhancements**:
   - Ensemble methods (voting classifier)
   - Deep learning approaches (CNN)
   
3. **System Improvements**:
   - Multi-object detection
   - Temporal smoothing for video
   - Mobile deployment

### 10.3 Lessons Learned

[Reflect on the project experience]

---

## 11. References

1. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection.
2. Cortes, C., & Vapnik, V. (1995). Support-vector networks.
3. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
4. [Add more relevant references]

---

## Appendix A: Code Repository

GitHub/GitLab Link: [Your repository URL]

---

## Appendix B: Hyperparameter Tuning

[Document any hyperparameter search performed]

---

## Appendix C: Additional Experiments

[Document any additional experiments or ablation studies]
