# 🐱🐶 Task-03: Image Classification — Cats vs Dogs (SVM)

Classify images of cats and dogs using a **Support Vector Machine (SVM)** with **HOG (Histogram of Oriented Gradients)** feature extraction and **PCA** dimensionality reduction.

---

## 📂 Dataset
[Kaggle — Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

Download and unzip `train.zip`. You will get a `train/` folder with files named:
```
cat.0.jpg  cat.1.jpg  ...  dog.0.jpg  dog.1.jpg  ...
```
Place the `train/` folder in the same directory as the script.

---

## 🔧 Pipeline

```
Raw Image (JPG)
     ↓  Resize to 64×64
     ↓  Convert to Grayscale
     ↓  HOG Feature Extraction  →  ~1764-dim vector
     ↓  StandardScaler
     ↓  PCA (150 components)    →  ~90% variance retained
     ↓  SVM — RBF Kernel (C=10)
     ↓  Prediction: Cat (0) / Dog (1)
```

---

## ⚙️ Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| Image size | 64 × 64 | Resize before feature extraction |
| Images per class | 1000 | Increase for higher accuracy (slower) |
| HOG orientations | 9 | Gradient direction bins |
| HOG pixels/cell | 8 | Spatial granularity |
| HOG cells/block | 2 | Normalization block size |
| PCA components | 150 | ~90% variance retained |
| SVM kernel | RBF | Best for non-linear image features |
| SVM C | 10 | Regularization parameter |

---

## 🗂️ Output Format

### Console Output
```
Found 1000 cat images and 1000 dog images.
HOG feature vector length : 1764
Total samples             : 2000

Train: 1600  |  Test: 400
PCA: 150 components explain 91.3% of variance

Training SVM (RBF kernel) …
Training complete.

── Evaluation Metrics ──
  Accuracy : 78.50%

── Classification Report ──
              precision  recall  f1-score  support
         Cat       0.79    0.78      0.79      200
         Dog       0.78    0.79      0.79      200
    accuracy                         0.79      400

── Sample Predictions (first 10 rows) ──
  Filename  True_Label  Predicted_Label  P_Dog  Correct
cat.0.jpg         Cat              Cat  0.112     True
dog.0.jpg         Dog              Dog  0.891     True
...
```

### svm_predictions.csv
```
Filename, True_Label, Predicted_Label, P_Dog, Correct
cat.0.jpg, Cat, Cat, 0.112, True
cat.1.jpg, Cat, Dog, 0.623, False
dog.0.jpg, Dog, Dog, 0.891, True
...
```

### Saved Plots
| File | Description |
|---|---|
| `plot1_confusion_matrix.png` | Confusion matrix (TP/FP/TN/FN) |
| `plot2_pca_variance.png` | Cumulative explained variance vs components |
| `plot3_sample_predictions.png` | 3×6 grid of test images with ✔/✘ labels |
| `plot4_hog_visualisation.png` | Side-by-side: original image vs HOG gradient map |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn scikit-image Pillow matplotlib

# 2. Place the train/ folder here, then:
python cats_dogs_svm.py
```

> **Speed tip:** The default uses 1000 images/class (~2 min on CPU).  
> Set `MAX_PER_CLASS = 500` in the config block for a faster run (~45 sec).

---

## 📦 File Structure

```
├── cats_dogs_svm.py                  # Main script
├── train/                            # Kaggle dataset folder (download separately)
│   ├── cat.0.jpg ... cat.N.jpg
│   └── dog.0.jpg ... dog.N.jpg
├── svm_predictions.csv               # Generated: per-image predictions
├── plot1_confusion_matrix.png        # Generated
├── plot2_pca_variance.png            # Generated
├── plot3_sample_predictions.png      # Generated
├── plot4_hog_visualisation.png       # Generated
└── README.md
```

---

## 📊 Why HOG + SVM?

| Component | Purpose |
|---|---|
| **HOG** | Captures edge/gradient structure — the "shape" of cats vs dogs — without needing a deep network |
| **PCA** | Compresses the ~1764-dim HOG vector to 150 dims, removes noise, speeds up SVM training |
| **SVM RBF** | Finds a non-linear decision boundary in the reduced feature space; performs well on mid-size datasets |

> Expected accuracy with 1000 images/class: **~75–82%**  
> For higher accuracy, consider CNN (e.g. transfer learning with ResNet).

---

## 🛠️ Dependencies

```
numpy
pandas
scikit-learn
scikit-image
Pillow
matplotlib
```
