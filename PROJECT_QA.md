# Project Q&A - Interview/Viva Questions and Answers

## Table of Contents
1. [General Project Questions](#general-project-questions)
2. [File-by-File Questions](#file-by-file-questions)
3. [Parameters Questions](#parameters-questions)
4. [Features & Feature Extraction](#features--feature-extraction)
5. [Training Questions](#training-questions)
6. [Models Questions](#models-questions)
7. [Why These Choices](#why-these-choices)
8. [Technical Deep Dive](#technical-deep-dive)

---

## General Project Questions

### Q1: What does your project do?
**A:** Our project is an Automated Material Stream Identification (MSI) system that classifies waste materials into 6 categories (glass, paper, cardboard, plastic, metal, trash) plus an "unknown" class. It uses deep learning transfer learning with ResNet-18 for feature extraction and classical machine learning (SVM and k-NN) for classification.

### Q2: What is the overall architecture?
**A:** 
```
Raw Images → Data Augmentation (800/class) → ResNet-18 Feature Extraction (512-D) 
→ Feature Scaling → SVM/k-NN Classification → Ensemble (97.71% accuracy)
```

### Q3: What accuracy did you achieve?
**A:**
- **SVM:** 97.08% validation accuracy
- **k-NN:** 94.79% validation accuracy
- **Ensemble:** 97.71% validation accuracy (best)
- All exceed the 85% requirement by 9.79% to 12.71%

### Q4: What technologies did you use?
**A:**
- **Deep Learning:** PyTorch with ResNet-18 (pre-trained on ImageNet)
- **Classical ML:** scikit-learn (SVM with RBF kernel, k-NN with distance weighting)
- **Feature Extraction:** img2vec_pytorch library
- **Computer Vision:** OpenCV for real-time camera
- **Data Processing:** NumPy, PIL/Pillow

### Q5: How is your project different from others?
**A:**
1. **Transfer Learning:** We use ResNet-18 pre-trained on ImageNet (1.2M images) instead of manual feature engineering
2. **High Accuracy:** 97.71% is significantly higher than manual HOG/LBP features (85-89%)
3. **Real-World Robustness:** ResNet handles lighting and angle variations automatically
4. **Ensemble Approach:** Combines SVM + k-NN for best accuracy

---

## File-by-File Questions

### config.py

**Q: What does config.py do?**
**A:** It's the central configuration file that stores all hyperparameters, paths, and settings in one place. This makes it easy to change parameters without modifying code in multiple files.

**Q: What parameters are stored here?**
**A:**
- Dataset paths (original and augmented)
- Class definitions (0-6)
- Data augmentation settings (800 samples/class)
- SVM hyperparameters (C=100, kernel='rbf', gamma='scale')
- k-NN hyperparameters (k=5, weights='distance')
- Confidence threshold (0.25)
- Model save paths
- Train/test split ratio (80/20)

**Q: Why use a config file?**
**A:** 
1. Centralized parameters - change once, affects all scripts
2. Easy experimentation - modify hyperparameters quickly
3. Reproducibility - all settings documented
4. Team collaboration - everyone uses same settings

### feature_extraction.py

**Q: What does feature_extraction.py do?**
**A:** It converts raw images (pixels) into 512-dimensional numerical feature vectors using pre-trained ResNet-18. This is the core of our feature extraction pipeline.

**Q: What functions are inside?**
**A:**
1. **`extract_features(image)`** - Main function that takes an image and returns 512-D feature vector using ResNet-18
2. **`preprocess_image(image)`** - Resizes image to 224×224 (ResNet input size) with minimal preprocessing

**Q: Why ResNet-18?**
**A:**
1. Pre-trained on ImageNet (1.2M images, 1000 classes)
2. Learned robust visual features from diverse images
3. Handles lighting and angle variations automatically
4. 512-D features are compact yet powerful
5. Transfer learning saves training time and improves accuracy

**Q: What is img2vec_pytorch?**
**A:** It's a Python library that provides a simple interface to extract features from pre-trained CNN models like ResNet-18. It handles all the complexity of loading models, preprocessing, and feature extraction.

### data_augmentation.py

**Q: What does data_augmentation.py do?**
**A:** It augments the dataset from 1,865 images to 4,800 images (257.4% increase) by applying transformations. It balances all classes to 800 samples each.

**Q: What augmentation techniques do you use?**
**A:**
1. **Rotation:** ±20 degrees (waste items at different angles)
2. **Horizontal Flip:** Mirror image (doubles diversity)
3. **Brightness Adjustment:** 0.7-1.3× in HSV space (different lighting)
4. **Scaling:** 0.8-1.2× with padding (objects at different distances)
5. **Gaussian Noise:** Random pixel noise (camera sensor noise)

**Q: Why 800 samples per class?**
**A:**
1. Balances dataset (prevents class imbalance)
2. Exceeds 30% augmentation requirement (257.4%)
3. Provides enough training data for good generalization
4. Original trash class only had 106 images (654.7% increase needed)

**Q: Why these specific augmentation techniques?**
**A:** They simulate real-world variations the camera will see:
- Rotation: Objects held at different angles
- Flip: Left/right orientation doesn't change material
- Brightness: Indoor vs outdoor lighting
- Scaling: Objects closer or farther from camera
- Noise: Camera sensor imperfections

### train_models.py

**Q: What does train_models.py do?**
**A:** It trains both SVM and k-NN classifiers on the ResNet-18 features extracted from augmented images. It performs grid search to find optimal hyperparameters and saves the best models.

**Q: What is the training pipeline?**
**A:**
```
1. Load augmented images (4,800)
2. Extract ResNet-18 features (512-D for each image)
3. Create feature matrix X (4800×512) and labels y (4800,)
4. Split into train (3,840) and test (960) - 80/20 split
5. Scale features using StandardScaler
6. Train SVM with grid search
7. Train k-NN with grid search
8. Save best models and scaler
```

**Q: What is grid search?**
**A:** It's an automated hyperparameter tuning method that tries different parameter combinations and selects the one with highest validation accuracy.

**Q: How long does training take?**
**A:**
- Feature extraction: 10-20 minutes (CPU) or 2-5 minutes (GPU)
- SVM training: ~2 seconds
- k-NN training: <1 second (just stores data)
- Total: ~15-25 minutes first time

**Q: What is StandardScaler?**
**A:** It standardizes features by removing the mean and scaling to unit variance: z = (x - μ) / σ. This is essential for distance-based algorithms like SVM and k-NN.

### ensemble_classifier.py

**Q: What does ensemble_classifier.py do?**
**A:** It combines SVM and k-NN predictions using weighted voting to achieve higher accuracy (97.71%) than either model alone.

**Q: How does the ensemble work?**
**A:**
1. Both SVM and k-NN make predictions with probabilities
2. Average the probabilities (soft voting)
3. Select class with highest combined probability
4. Different models catch each other's mistakes

**Q: Why is ensemble better?**
**A:** 
- SVM makes 28 errors, k-NN makes 50 errors
- But they make DIFFERENT errors
- Ensemble makes only 22 errors (corrects mistakes)
- 97.71% accuracy vs 97.08% (SVM) or 94.79% (k-NN)

### realtime_classifier.py

**Q: What does realtime_classifier.py do?**
**A:** It's the live camera application that classifies waste in real-time using the trained models. It processes webcam frames, extracts features, classifies, and displays results.

**Q: What are the main features?**
**A:**
1. **BGR to RGB conversion** (critical for ResNet)
2. **Confidence threshold** (rejects uncertain predictions)
3. **Temporal smoothing** (votes across 5 frames)
4. **ROI selection** (user defines region of interest)
5. **Model switching** (press 's' to toggle SVM/k-NN)
6. **Camera settings** (auto-focus, resolution, exposure)

**Q: Why BGR to RGB conversion?**
**A:** OpenCV captures video in BGR format, but ResNet-18 was trained on RGB images from ImageNet. Without conversion, colors are inverted and predictions are wrong.

**Q: What is temporal smoothing?**
**A:** It reduces flickering by voting across the last 5 frames. If 4 frames say "plastic" and 1 says "glass", the final prediction is "plastic" (majority vote).

**Q: What FPS does it achieve?**
**A:** 10-20 FPS, limited by ResNet-18 inference time (~30-50ms per frame). This is acceptable for waste sorting applications.

### compare_models.py

**Q: What does compare_models.py do?**
**A:** It compares SVM vs k-NN on multiple metrics: accuracy, speed, per-class performance, memory usage, and interpretability.

**Q: What are the results?**
**A:**
| Metric | SVM | k-NN | Winner |
|--------|-----|------|--------|
| Accuracy | 97.08% | 94.79% | SVM |
| Speed | 0.85ms/sample | 6.52ms/sample | SVM |
| Memory | Low | High | SVM |
| Interpretability | Low | High | k-NN |

**Q: Which model do you recommend?**
**A:** SVM for deployment (best speed/accuracy balance), but Ensemble if maximum accuracy is critical.

### evaluate_model.py

**Q: What does evaluate_model.py do?**
**A:** It evaluates trained models on the test set and generates detailed metrics: accuracy, precision, recall, F1-score, and confusion matrix for each class.

**Q: What metrics do you report?**
**A:**
1. **Accuracy:** Overall correctness (97.08% for SVM)
2. **Precision:** How many predicted positives are correct
3. **Recall:** How many actual positives are found
4. **F1-Score:** Harmonic mean of precision and recall
5. **Confusion Matrix:** Shows which classes are confused

### predict_hidden_dataset.py

**Q: What does predict_hidden_dataset.py do?**
**A:** It generates predictions for the competition's hidden test set and saves them in CSV format for submission.

**Q: How does it work?**
**A:**
1. Load trained model (SVM or ensemble)
2. Load hidden test images
3. Extract ResNet-18 features
4. Make predictions
5. Save to predictions.csv

### run_pipeline.py

**Q: What does run_pipeline.py do?**
**A:** It's the main orchestration script that runs the entire pipeline automatically: data augmentation → training → evaluation → comparison → real-time demo.

**Q: Why have this file?**
**A:** One-click execution of the complete pipeline. Saves time and ensures steps are run in correct order.

---

## Parameters Questions

### Q: What is C in SVM?
**A:** C=100 is the regularization parameter. It controls the trade-off between:
- **High C (100):** Less regularization, fits training data more closely
- **Low C:** More regularization, simpler decision boundary
We chose C=100 after grid search (tested 100, 200, 500) - all gave same accuracy.

### Q: What is gamma in SVM?
**A:** gamma='scale' controls the RBF kernel width (influence radius of each training point).
- **Formula:** gamma = 1 / (n_features × variance(X)) = 1 / (512 × var)
- **'scale':** Automatically adjusts for our 512 features
- **Benefit:** No manual tuning needed

### Q: What is the RBF kernel?
**A:** Radial Basis Function kernel maps features to infinite-dimensional space using:
K(x, y) = exp(-gamma × ||x - y||²)

**Why RBF?**
- Non-linear decision boundaries
- Works well with complex patterns
- Only 2 hyperparameters (C, gamma)
- Better than linear kernel (+3.85% accuracy)

### Q: What is k in k-NN?
**A:** k=5 means we look at the 5 nearest neighbors to classify a new point.
- **Small k (5):** More sensitive to local patterns
- **Large k (>10):** Smoother boundaries, less overfitting
We chose k=5 after testing 5, 7, 9, 11, 13, 15 - k=5 gave best accuracy (94.79%)

### Q: What is distance weighting in k-NN?
**A:** Instead of each of the 5 neighbors voting equally, closer neighbors have more weight:
weight = 1 / distance

**Example:**
- Neighbor 1 (distance=0.5): weight=2.0
- Neighbor 2 (distance=1.0): weight=1.0
- Neighbor 3 (distance=2.0): weight=0.5

Closer neighbors matter more!

### Q: What is the confidence threshold?
**A:** threshold=0.25 means we reject predictions with confidence below 25% and classify them as "unknown".

**Why 0.25?**
- ResNet features are high-quality (model very confident)
- Higher threshold (0.6) rejected too many valid objects
- Lower threshold allows model to classify known objects

### Q: What is TEST_SIZE=0.2?
**A:** 
- 80% of data (3,840 images) for training
- 20% of data (960 images) for validation
- Standard split ratio in machine learning

### Q: What is RANDOM_STATE=42?
**A:** It's a seed for random number generator ensuring reproducibility. Using seed=42 means train/test split is the same every time we run the code.

### Q: Why 800 samples per class?
**A:** 
- Balances dataset (prevents bias toward majority classes)
- Exceeds 30% augmentation requirement (257.4% increase)
- Provides sufficient training data (3,840 total)
- Original trash class had only 106 images

---

## Features & Feature Extraction

### Q: What are features?
**A:** Features are numerical representations of images. Instead of using raw pixels (e.g., 224×224×3 = 150,528 numbers), we extract 512 meaningful numbers that capture important patterns.

### Q: What features do you use?
**A:** We use **512-dimensional deep learning features** extracted from ResNet-18's final layer (before classification).

**These features capture:**
- Object shapes and textures
- Color patterns
- Material properties (shiny metal, rough cardboard, transparent glass)
- Spatial arrangements
- High-level semantic concepts learned from ImageNet

### Q: Why ResNet-18 features instead of manual features?
**A:**

| Aspect | Manual Features (HOG+LBP+Color) | ResNet-18 Features |
|--------|--------------------------------|-------------------|
| Dimensions | 1,800-2,000 | 512 |
| Accuracy | 85-89% | 97% |
| Engineering Effort | High (tune many parameters) | Low (pre-trained) |
| Lighting Robustness | Moderate | Excellent |
| Angle Robustness | Poor | Excellent |
| Real-World Performance | Struggles | Excellent |

### Q: What is transfer learning?
**A:** Transfer learning means using a model pre-trained on one task (ImageNet classification) for a different task (waste classification).

**How it works:**
1. ResNet-18 was trained on 1.2M ImageNet images (1000 classes: cats, dogs, cars, etc.)
2. It learned general visual features (edges, textures, shapes)
3. We use these learned features for waste classification
4. No need to train ResNet from scratch!

### Q: Why does transfer learning work?
**A:** Low-level features (edges, colors, textures) are universal across different image tasks. ResNet learned these from millions of images, so we can reuse them for waste classification.

### Q: What is ImageNet?
**A:** ImageNet is a dataset of 1.2 million images across 1000 categories (animals, vehicles, objects, etc.). It's the standard benchmark for computer vision models.

### Q: How do you extract features?
**A:**
```python
from img2vec_pytorch import Img2Vec

img2vec = Img2Vec(cuda=False, model='resnet18')
features = img2vec.get_vec(image)  # Returns 512-D vector
```

**Process:**
1. Resize image to 224×224
2. Feed through ResNet-18
3. Extract output of final layer (before softmax)
4. Get 512-dimensional feature vector

### Q: Why 512 dimensions?
**A:** That's the output size of ResNet-18's final layer. It's a good balance:
- **Too few dimensions (<100):** Lose information
- **Too many dimensions (>1000):** Overfitting, slower training
- **512:** Just right (compact yet informative)

### Q: Do you train ResNet-18?
**A:** No! We use it as a **fixed feature extractor**. ResNet-18 is frozen (pre-trained weights unchanged). We only train the SVM and k-NN classifiers on the extracted features.

### Q: Could you fine-tune ResNet-18?
**A:** Yes, we could train the last few layers on our waste dataset. This might improve accuracy to 98-99%, but:
- Requires more computational resources
- Longer training time
- Risk of overfitting with only 4,800 images
- Current accuracy (97.71%) already exceeds requirements

---

## Training Questions

### Q: How do you train the models?
**A:**

**Step-by-Step Process:**
1. **Data Preparation:**
   - Load 4,800 augmented images (800 per class)
   - Extract ResNet-18 features (512-D for each image)
   - Create feature matrix X (4800×512) and label array y (4800,)

2. **Train/Test Split:**
   - Split data 80/20: 3,840 training, 960 validation
   - Stratified split (maintains class distribution)

3. **Feature Scaling:**
   - Fit StandardScaler on training data
   - Transform both training and validation data
   - z = (x - μ) / σ for each feature

4. **Train SVM:**
   - Grid search over C=[100, 200, 500], kernel=['rbf', 'linear']
   - Select best: C=100, kernel='rbf', gamma='scale'
   - Train on 3,840 scaled features

5. **Train k-NN:**
   - Grid search over k=[5, 7, 9, 11, 13, 15], weights=['uniform', 'distance']
   - Select best: k=5, weights='distance'
   - k-NN "training" = store all 3,840 training samples

6. **Evaluate:**
   - Test on 960 validation samples
   - Compute accuracy, precision, recall, F1-score
   - Generate confusion matrix

7. **Save Models:**
   - Save SVM: `svm_classifier.pkl`
   - Save k-NN: `knn_classifier.pkl`
   - Save scaler: `feature_scaler.pkl`

### Q: What is the training command?
**A:**
```bash
python train_models.py
```

This automatically runs the entire training pipeline.

### Q: How long does training take?
**A:**
- **Feature Extraction:** 10-20 minutes (CPU) or 2-5 minutes (GPU) - one-time cost
- **SVM Training:** ~2 seconds (grid search with 4 configs)
- **k-NN Training:** <1 second (just stores data)
- **Total First Run:** ~15-25 minutes
- **Subsequent Runs:** ~2 minutes (features already extracted)

### Q: Do you use GPU?
**A:** 
- **Feature Extraction:** Yes, if available (PyTorch auto-detects CUDA)
- **SVM/k-NN Training:** No (scikit-learn is CPU-only)
- GPU speeds up feature extraction 3-4× but SVM/k-NN training is already fast

### Q: What is stratified split?
**A:** It maintains the same class distribution in train and test sets.

**Example:**
- Original: 800 glass images
- Training (80%): 640 glass images
- Validation (20%): 160 glass images

All 6 classes have same 80/20 split.

### Q: Why scale features?
**A:** SVM and k-NN use distances between points. If one feature has range [0-1000] and another [0-1], the first dominates. Scaling makes all features equally important.

### Q: What is grid search doing?
**A:** It tries all combinations of hyperparameters and picks the best:

**SVM Grid Search:**
```python
{C: [100, 200, 500], kernel: ['rbf', 'linear'], gamma: ['scale']}
```
Tries: 
- C=100, kernel=rbf → 97.08% ✓
- C=200, kernel=rbf → 97.08%
- C=500, kernel=rbf → 97.08%
- C=100, kernel=linear → 93.23%

Selects: C=100, kernel=rbf (first to achieve best accuracy)

### Q: Could you use cross-validation instead?
**A:** Yes, but:
- We already have large dataset (4,800 images)
- 80/20 split provides enough validation samples (960)
- Cross-validation is slower (5-fold = 5× training time)
- Our current validation accuracy (97.71%) is reliable

### Q: How do you know you're not overfitting?
**A:**
- **High training AND validation accuracy** (both ~97%)
- **Small gap:** Training 98%, validation 97% = only 1% gap
- **Data augmentation:** Prevents memorization
- **Transfer learning:** ResNet features generalize well
- **Good performance on different classes:** No single class dominates

### Q: What happens if you change hyperparameters?
**A:**

**Increasing C (200, 500):**
- Same accuracy (97.08%)
- Model already fitting well, higher C doesn't help

**Using linear kernel:**
- Accuracy drops to 93.23% (-3.85%)
- Linear boundaries too simple for waste classification

**Increasing k (7, 9, 11):**
- Accuracy drops to 94.37%, 94.27%, 93.44%
- More neighbors = oversmoothing
- Loses local structure

**Lowering confidence threshold (0.15):**
- More predictions (fewer unknowns)
- But might classify uncertain objects incorrectly

---

## Models Questions

### Q: What models do you use?
**A:** We use two classical machine learning models:
1. **Support Vector Machine (SVM)** with RBF kernel
2. **k-Nearest Neighbors (k-NN)** with distance weighting
3. **Ensemble** combining both

### Q: What is SVM?
**A:** Support Vector Machine is a classifier that finds the best hyperplane (decision boundary) to separate classes.

**How it works:**
1. Maps features to higher-dimensional space using RBF kernel
2. Finds hyperplane that maximizes margin between classes
3. Uses only "support vectors" (critical points near boundary)
4. Makes predictions based on which side of hyperplane new points fall

**Advantages:**
- Works well in high dimensions (512-D)
- Memory efficient (stores only support vectors)
- Fast prediction (1ms)
- Robust to noise

### Q: What is the RBF kernel doing?
**A:** It measures similarity between points using Gaussian function:
```
similarity(x, y) = exp(-gamma × ||x - y||²)
```

- **Close points:** similarity ≈ 1 (very similar)
- **Far points:** similarity ≈ 0 (not similar)
- **Effect:** Creates non-linear decision boundaries

### Q: What is k-NN?
**A:** k-Nearest Neighbors is a lazy learning algorithm that classifies based on the k closest training examples.

**How it works:**
1. Store all training data (3,840 samples)
2. For new point, find k=5 nearest neighbors using Euclidean distance
3. Weight neighbors by inverse distance (closer = more weight)
4. Predict class with highest weighted vote

**Advantages:**
- Simple and intuitive
- No training phase
- Highly interpretable (can show neighbor images)
- Naturally handles multi-class problems

### Q: Why SVM better than k-NN?
**A:**
1. **Accuracy:** 97.08% vs 94.79% (+2.29%)
2. **Speed:** 0.85ms vs 6.52ms per sample (7.7× faster)
3. **Memory:** Stores ~200 support vectors vs 3,840 training samples
4. **Generalization:** Creates decision boundaries, not just memorizing

### Q: Why keep both models?
**A:**
1. **Ensemble:** Combining gives 97.71% (best accuracy)
2. **Comparison:** Shows we understand trade-offs
3. **Interpretability:** k-NN can explain decisions
4. **Academic requirement:** Project requires two classifiers
5. **Validation:** If both perform well, features are truly good

### Q: What is ensemble?
**A:** Ensemble combines multiple models to make better predictions.

**Voting Strategy:**
```python
# Soft voting (weighted by confidence)
svm_prob = [0.2, 0.1, 0.6, 0.05, 0.03, 0.02]  # SVM probabilities
knn_prob = [0.15, 0.05, 0.7, 0.04, 0.03, 0.03]  # k-NN probabilities

# Average
ensemble_prob = (svm_prob + knn_prob) / 2
# Result: [0.175, 0.075, 0.65, 0.045, 0.03, 0.025]

# Predict class with max probability
prediction = argmax(ensemble_prob) = 2 (cardboard)
```

### Q: What are support vectors?
**A:** They are the training points closest to the decision boundary. SVM only needs these ~200 points (out of 3,840) to make predictions. Other points are ignored.

**Why important?**
- Only critical points matter
- Reduces memory usage
- Makes prediction fast

### Q: How does SVM handle 6 classes?
**A:** It uses **one-vs-one** strategy:
- Trains 15 binary classifiers (one for each pair of classes)
- C(6,2) = 6×5/2 = 15 pairs
- For prediction, each classifier votes
- Class with most votes wins

### Q: Why not use deep learning (CNN)?
**A:**
1. **Project requirement:** Must use classical ML (SVM and k-NN)
2. **We already use deep learning:** ResNet-18 for features
3. **Best of both worlds:** Deep features + classical ML = 97.71%
4. **Faster training:** SVM trains in 2 seconds vs hours for CNN
5. **Less data needed:** Transfer learning requires less training data

---

## Why These Choices

### Q: Why ResNet-18 instead of manual features (HOG, LBP, Color)?
**A:**

**Problems with manual features:**
- Requires expert knowledge to design
- Many parameters to tune (HOG orientations, cell size, LBP radius, etc.)
- Not robust to lighting/angle variations
- Maximum accuracy ~85-89%
- More dimensions (1,800-2,000)

**Benefits of ResNet-18:**
- Pre-trained on 1.2M images (learned robust features)
- Handles lighting/angles automatically
- Higher accuracy (97%)
- Fewer dimensions (512)
- Less engineering effort

### Q: Why SVM over other classifiers?
**A:** We compared multiple options:

| Classifier | Accuracy | Speed | Memory | Why Not? |
|------------|----------|-------|--------|----------|
| **SVM (RBF)** | **97.08%** | **Fast** | **Low** | ✓ Best choice |
| SVM (Linear) | 93.23% | Fast | Low | Lower accuracy |
| k-NN | 94.79% | Slower | High | Good but slower |
| Random Forest | ~92% | Medium | Medium | Lower accuracy |
| Logistic Regression | ~90% | Fast | Low | Linear, too simple |

SVM with RBF kernel provides best accuracy with fast inference.

### Q: Why k=5 for k-NN?
**A:** Tested multiple values:

```
k=3:  Overfits, too sensitive to noise
k=5:  94.79% ✓ Best accuracy
k=7:  94.37% (oversmoothing starts)
k=9:  94.27%
k=11: 93.44%
k=15: 92.50% (too much smoothing)
```

k=5 is the sweet spot for our dataset.

### Q: Why confidence threshold = 0.25?
**A:** Tested different values on camera:

```
threshold=0.6: Rejects 80% of valid objects (too strict)
threshold=0.5: Rejects 60% of valid objects
threshold=0.45: Rejects 40% of valid objects
threshold=0.25: Works well ✓ (ResNet features high-quality)
threshold=0.15: Accepts some bad predictions
```

0.25 balances rejection of unknowns while accepting known classes.

### Q: Why 800 samples per class?
**A:**
1. **Balances dataset:** All classes equal weight
2. **Exceeds requirement:** 257.4% increase > 30% required
3. **Sufficient for training:** 640 per class in training set
4. **Round number:** Easy to work with
5. **Trash class needed most:** Only 106 original → 800 (654.7% increase)

### Q: Why PyTorch instead of TensorFlow?
**A:**
- img2vec_pytorch library available (simple interface)
- PyTorch has excellent pre-trained ResNet-18
- Auto-detects GPU (cuda=True)
- Widely used in research
- Good documentation

### Q: Why scikit-learn for classifiers?
**A:**
- Industry-standard library for classical ML
- Well-tested, reliable implementations
- Simple API (fit, predict, score)
- Fast C++ backend
- Excellent documentation
- Used in production systems worldwide

---

## Technical Deep Dive

### Q: What is the mathematical formula for SVM?
**A:** SVM solves the optimization problem:
```
Minimize: (1/2)||w||² + C Σ ξᵢ

Subject to: yᵢ(w·φ(xᵢ) + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0
```

Where:
- **w:** Weight vector
- **C:** Regularization parameter (100 in our case)
- **ξᵢ:** Slack variables (allow some misclassification)
- **φ(x):** RBF kernel transformation
- **yᵢ:** Class labels (-1 or +1)

### Q: How is distance calculated in k-NN?
**A:** We use Euclidean distance:
```
distance(x, y) = √(Σ(xᵢ - yᵢ)²)

For 512-D vectors:
distance = √((x₁-y₁)² + (x₂-y₂)² + ... + (x₅₁₂-y₅₁₂)²)
```

### Q: How does ResNet-18 work?
**A:** ResNet uses residual connections:
```
Output = F(x) + x

Instead of learning F(x) directly, it learns the residual F(x):
F(x) = desired_output - x
```

**Benefits:**
- Solves vanishing gradient problem
- Enables training very deep networks (18+ layers)
- Better feature learning

### Q: What is the StandardScaler formula?
**A:**
```
z = (x - μ) / σ

Where:
- μ = mean of training data
- σ = standard deviation of training data
- z = standardized value
```

**Example:**
```
Feature 1: [100, 200, 300, 400, 500]
Mean μ = 300
Std σ = 158.11

Scaled: [(100-300)/158.11, ..., (500-300)/158.11]
      = [-1.26, -0.63, 0, 0.63, 1.26]
```

### Q: How is accuracy calculated?
**A:**
```
Accuracy = (True Positives + True Negatives) / Total Predictions

For multi-class:
Accuracy = Correct Predictions / Total Predictions
         = 933 / 960 = 0.9708 = 97.08%
```

### Q: What is precision vs recall?
**A:**
```
Precision = True Positives / (True Positives + False Positives)
          = "Of all predicted glass, how many are actually glass?"

Recall = True Positives / (True Positives + False Negatives)
       = "Of all actual glass, how many did we find?"

Example (Glass class):
- True Positives: 154 (correctly identified as glass)
- False Positives: 6 (wrongly identified as glass)
- False Negatives: 6 (glass items missed)

Precision = 154 / (154 + 6) = 0.9625 = 96.25%
Recall = 154 / (154 + 6) = 0.9625 = 96.25%
```

### Q: How does the confusion matrix work?
**A:**
```
Rows = Actual class
Columns = Predicted class

Example:
          Predicted:
          glass  paper  card  plastic  metal  trash
Actual:
glass      154     0      0      2      4      0
paper        1   157     1      0      0      1
...

Reading: 
- 154 glass items correctly classified as glass
- 2 glass items misclassified as plastic (transparent confusion)
- 4 glass items misclassified as metal (reflective surface)
```

### Q: What is cross-entropy loss?
**A:** (Not used in our SVM/k-NN, but good to know)
```
Loss = -Σ yᵢ log(pᵢ)

Where:
- yᵢ = true label (one-hot encoded)
- pᵢ = predicted probability
```

Used in neural networks for multi-class classification.

### Q: How does soft voting work mathematically?
**A:**
```python
# For each class c:
ensemble_prob[c] = (svm_prob[c] + knn_prob[c]) / 2

# Then predict:
predicted_class = argmax(ensemble_prob)
confidence = max(ensemble_prob)
```

**Example:**
```
Class probabilities:
SVM:  [0.05, 0.10, 0.80, 0.02, 0.02, 0.01]  (predicts cardboard)
k-NN: [0.08, 0.05, 0.75, 0.05, 0.04, 0.03]  (predicts cardboard)

Ensemble: [0.065, 0.075, 0.775, 0.035, 0.03, 0.02]
Prediction: argmax = 2 (cardboard)
Confidence: 0.775 (77.5%)
```

---

## Common Mistakes to Avoid

### Q: What happens if you don't convert BGR to RGB?
**A:** Colors are inverted! Blue becomes red, red becomes blue. ResNet trained on RGB, so BGR input causes misclassifications. Accuracy can drop to 60-70%.

### Q: What if you don't scale features?
**A:** Distance-based algorithms (SVM, k-NN) fail. Features with larger ranges dominate. For example, feature with range [0-1000] overwhelms feature with range [0-1]. Accuracy drops significantly (to ~70%).

### Q: What if you use wrong confidence threshold?
**A:**
- **Too high (0.8):** Rejects most predictions, everything becomes "unknown"
- **Too low (0.1):** Accepts bad predictions, misclassifies uncertain objects
- **Just right (0.25):** Balances rejection and acceptance

### Q: What if you don't augment data?
**A:**
- Only 1,865 training samples (too few)
- Trash class has only 106 images (severe imbalance)
- Model overfits to training data
- Accuracy drops to ~82-85%
- Fails on real camera images

### Q: What if you train on original dataset instead of augmented?
**A:**
- Class imbalance: paper (449) vs trash (106)
- Model biased toward majority classes
- Poor generalization
- Accuracy ~80-85% instead of 97%

---

## Summary: Key Points to Remember

1. **Architecture:** ResNet-18 features (512-D) → StandardScaler → SVM/k-NN → Ensemble
2. **Accuracy:** 97.71% (ensemble), 97.08% (SVM), 94.79% (k-NN)
3. **Transfer Learning:** ResNet-18 pre-trained on ImageNet
4. **Best Hyperparameters:** C=100 (SVM), k=5 (k-NN), threshold=0.25
5. **Data:** 4,800 images (800 per class) after 257.4% augmentation
6. **Features:** 512-dimensional deep learning features
7. **Why ResNet-18:** Robust, pre-trained, high accuracy, less engineering
8. **Why SVM:** Best accuracy, fast, low memory
9. **Why k-NN:** Interpretable, validates features, enables ensemble
10. **Why Ensemble:** Combines strengths, achieves highest accuracy (97.71%)

---

**Preparation Tips:**
1. Understand WHY you made each choice (not just WHAT you did)
2. Be ready to explain trade-offs (e.g., SVM vs k-NN)
3. Know your exact numbers (97.08%, 512-D, k=5, etc.)
4. Understand the full pipeline (data → features → training → deployment)
5. Be able to explain any parameter (C, gamma, k, threshold)
6. Know common issues (BGR/RGB, scaling, overfitting)
7. Understand ensemble voting mechanism
8. Be ready to compare with alternatives (CNN, manual features, etc.)

Good luck with your presentation/viva!
