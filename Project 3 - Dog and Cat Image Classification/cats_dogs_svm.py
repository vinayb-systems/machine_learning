# ============================================================
# Task-03: SVM Image Classifier — Cats vs Dogs
# Dataset: https://www.kaggle.com/c/dogs-vs-cats/data
#
# Expected folder layout:
#   Dataset/
#     train/  cats/  *.jpg
#             dogs/  *.jpg
#     val/    cats/  *.jpg
#             dogs/  *.jpg
#     test/   cats/  *.jpg
#             dogs/  *.jpg
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────
DATASET_DIR   = "Dataset"         # root folder (same level as this script)
IMG_SIZE      = (64, 64)          # resize all images to this
MAX_PER_CLASS = 1000              # images per class (lower = faster)
HOG_ORIENT    = 9
HOG_PIXELS    = 8                 # pixels per cell
HOG_CELLS     = 2                 # cells per block
PCA_COMPONENTS = 150
RANDOM_STATE  = 42

# ── 1. LOAD FILE PATHS FROM FOLDER STRUCTURE ──────────────────
def get_files(split: str, label: str) -> list:
    """Return sorted image paths for Dataset/<split>/<label>/"""
    pattern = os.path.join(DATASET_DIR, split, label, "*.jpg")
    files   = sorted(glob.glob(pattern))
    # also pick up .jpeg / .png just in case
    for ext in ("*.jpeg", "*.png"):
        files += sorted(glob.glob(os.path.join(DATASET_DIR, split, label, ext)))
    return files

train_cats = get_files("train", "cats")[:MAX_PER_CLASS]
train_dogs = get_files("train", "dogs")[:MAX_PER_CLASS]
val_cats   = get_files("val",   "cats")[:MAX_PER_CLASS]
val_dogs   = get_files("val",   "dogs")[:MAX_PER_CLASS]
test_cats  = get_files("test",  "cats")[:MAX_PER_CLASS]
test_dogs  = get_files("test",  "dogs")[:MAX_PER_CLASS]

if not train_cats or not train_dogs:
    raise FileNotFoundError(
        f"No images found under '{DATASET_DIR}/train/cats' or '.../dogs'.\n"
        f"Current working directory: {os.getcwd()}\n"
        "Make sure the script is run from the same folder that contains 'Dataset/'."
    )

print("── Image counts ──")
print(f"  Train : {len(train_cats)} cats | {len(train_dogs)} dogs")
print(f"  Val   : {len(val_cats)}   cats | {len(val_dogs)}   dogs")
print(f"  Test  : {len(test_cats)}  cats | {len(test_dogs)}  dogs")

# ── 2. HOG FEATURE EXTRACTION ─────────────────────────────────
def extract_hog(filepath: str) -> np.ndarray:
    """Load → resize → grayscale → HOG feature vector."""
    img  = Image.open(filepath).convert("RGB").resize(IMG_SIZE)
    gray = rgb2gray(np.array(img))
    feat = hog(
        gray,
        orientations=HOG_ORIENT,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        block_norm="L2-Hys",
        feature_vector=True
    )
    return feat

def build_dataset(cat_files, dog_files, desc=""):
    print(f"  Extracting HOG features: {desc} …")
    X_c = np.array([extract_hog(f) for f in cat_files])
    X_d = np.array([extract_hog(f) for f in dog_files])
    X   = np.vstack([X_c, X_d])
    y   = np.array([0]*len(cat_files) + [1]*len(dog_files))
    files = cat_files + dog_files
    return X, y, files

print("\nExtracting HOG features (this takes ~1–2 min) …")
X_train, y_train, train_files = build_dataset(train_cats, train_dogs, "train")
X_val,   y_val,   val_files   = build_dataset(val_cats,   val_dogs,   "val")
X_test,  y_test,  test_files  = build_dataset(test_cats,  test_dogs,  "test")

print(f"\nHOG vector length : {X_train.shape[1]}")
print(f"Train samples     : {X_train.shape[0]}")
print(f"Val   samples     : {X_val.shape[0]}")
print(f"Test  samples     : {X_test.shape[0]}")

# ── 3. SCALING ────────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ── 4. PCA ────────────────────────────────────────────────────
# n_components must be <= min(n_samples, n_features)-1; auto-cap to be safe
max_components = min(X_train_sc.shape[0], X_train_sc.shape[1]) - 1
n_components   = min(PCA_COMPONENTS, max_components)
if n_components < PCA_COMPONENTS:
    print(f"  PCA_COMPONENTS capped {PCA_COMPONENTS} -> {n_components} "
          f"(only {X_train_sc.shape[0]} training samples available)")

pca         = PCA(n_components=n_components, whiten=True, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_sc)
X_val_pca   = pca.transform(X_val_sc)
X_test_pca  = pca.transform(X_test_sc)

explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
print(f"\nPCA: {n_components} components -> {explained:.1f}% variance retained")

# ── 5. TRAIN SVM ──────────────────────────────────────────────
print("\nTraining SVM (RBF kernel, C=10) …")
svm = SVC(kernel="rbf", C=10, gamma="scale",
          class_weight="balanced", probability=True,
          random_state=RANDOM_STATE)
svm.fit(X_train_pca, y_train)
print("Training complete.")

# ── 6. EVALUATE ───────────────────────────────────────────────
label_names = ["Cat", "Dog"]

# Validation set
val_pred  = svm.predict(X_val_pca)
val_acc   = accuracy_score(y_val, val_pred)

# Test set
y_pred    = svm.predict(X_test_pca)
y_prob    = svm.predict_proba(X_test_pca)[:, 1]
test_acc  = accuracy_score(y_test, y_pred)
report    = classification_report(y_test, y_pred, target_names=label_names)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n── Evaluation Metrics ──")
print(f"  Validation Accuracy : {val_acc  * 100:.2f}%")
print(f"  Test Accuracy       : {test_acc * 100:.2f}%")
print(f"\n── Classification Report (Test Set) ──\n{report}")

# ── 7. SAVE PREDICTIONS CSV ───────────────────────────────────
results_df = pd.DataFrame({
    "Filename"        : [os.path.basename(f) for f in test_files],
    "True_Label"      : [label_names[l] for l in y_test],
    "Predicted_Label" : [label_names[p] for p in y_pred],
    "P_Dog"           : np.round(y_prob, 4),
    "Correct"         : y_test == y_pred
})
results_df.to_csv("svm_predictions.csv", index=False)
print("\nPredictions saved → svm_predictions.csv")
print(results_df.head(10).to_string(index=False))

# ── 8. PLOT 1 — Confusion Matrix ──────────────────────────────
fig1, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set",
             fontsize=13, fontweight="bold", pad=10)
fig1.tight_layout()
fig1.savefig("plot1_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → plot1_confusion_matrix.png")

# ── 9. PLOT 2 — PCA Explained Variance ────────────────────────
fig2, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.cumsum(pca.explained_variance_ratio_) * 100,
        color="#457B9D", lw=2)
ax.axhline(90, color="gray", lw=1, linestyle="--", label="90% threshold")
ax.axvline(n_components, color="#E63946", lw=1, linestyle=":",
           label=f"{n_components} components selected")
ax.set_xlabel("Number of PCA Components")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_title("PCA — Explained Variance", fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("plot2_pca_variance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → plot2_pca_variance.png")

# ── 10. PLOT 3 — Sample Predictions Grid ──────────────────────
n_samples = min(18, len(test_files))
sample_idx = np.random.RandomState(0).choice(len(test_files), n_samples, replace=False)

fig3, axes3 = plt.subplots(3, 6, figsize=(13, 6))
fig3.suptitle("Sample Predictions   ✔ Correct   ✘ Wrong",
              fontsize=13, fontweight="bold")

for ax_i, idx in zip(axes3.flat, sample_idx):
    img     = Image.open(test_files[idx]).convert("RGB").resize((96, 96))
    correct = bool(results_df.iloc[idx]["Correct"])
    pred    = results_df.iloc[idx]["Predicted_Label"]
    true    = results_df.iloc[idx]["True_Label"]
    color   = "#2A9D8F" if correct else "#E63946"
    symbol  = "✔" if correct else "✘"
    ax_i.imshow(img)
    ax_i.axis("off")
    ax_i.set_title(f"{symbol} {pred}\n(true: {true})",
                   fontsize=7, color=color, pad=3)

plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.88)
fig3.savefig("plot3_sample_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → plot3_sample_predictions.png")

# ── 11. PLOT 4 — HOG Visualisation ───────────────────────────
from skimage.feature import hog as hog_vis
from skimage import exposure

sample_vis = train_cats[:3] + train_dogs[:3]
fig4, axes4 = plt.subplots(2, 6, figsize=(13, 5))
fig4.suptitle("HOG Feature Visualisation  (top: original | bottom: HOG map)",
              fontsize=12, fontweight="bold")

for col, fpath in enumerate(sample_vis):
    img_arr = np.array(Image.open(fpath).convert("RGB").resize(IMG_SIZE))
    gray    = rgb2gray(img_arr)
    _, hog_img = hog_vis(
        gray, orientations=HOG_ORIENT,
        pixels_per_cell=(HOG_PIXELS, HOG_PIXELS),
        cells_per_block=(HOG_CELLS, HOG_CELLS),
        block_norm="L2-Hys", visualize=True, feature_vector=True
    )
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    lbl = "Cat" if col < 3 else "Dog"
    axes4[0, col].imshow(img_arr);  axes4[0, col].set_title(lbl, fontsize=9)
    axes4[0, col].axis("off")
    axes4[1, col].imshow(hog_img, cmap="inferno")
    axes4[1, col].axis("off")

plt.subplots_adjust(hspace=0.15, wspace=0.05, top=0.88)
fig4.savefig("plot4_hog_visualisation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → plot4_hog_visualisation.png")

# ── 12. FINAL SUMMARY ─────────────────────────────────────────
print("\n" + "="*55)
print("FINAL SUMMARY")
print("="*55)
print(f"  Train images (per class) : {MAX_PER_CLASS}")
print(f"  HOG vector length        : {X_train.shape[1]}")
print(f"  PCA components           : {PCA_COMPONENTS}  ({explained:.1f}% variance)")
print(f"  SVM kernel               : RBF  (C=10, gamma=scale)")
print(f"  Validation Accuracy      : {val_acc  * 100:.2f}%")
print(f"  Test Accuracy            : {test_acc * 100:.2f}%")
print(f"  Test Correct / Total     : {int(test_acc * len(y_test))} / {len(y_test)}")
print("="*55)