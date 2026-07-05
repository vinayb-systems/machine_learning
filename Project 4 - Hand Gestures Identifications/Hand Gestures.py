# ==========================================================
# Hand Gesture Recognition using CNN
# Dataset: LeapGestRecog (Kaggle)
# Author: Prodigy InfoTech Task-04
# ==========================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# Dataset Path
# ==========================================================

DATASET_PATH = "leapGestRecog"      # <-- Change this
IMG_SIZE = 64

# ==========================================================
# Load Dataset
# ==========================================================

images = []
labels = []


IMG_SIZE = 64
DATASET_PATH = "leapGestRecog"

print("Loading Dataset...")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):

            img_path = os.path.join(root, file)

            label = os.path.basename(os.path.dirname(img_path))

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)
            labels.append(label)

print("Images Loaded:", len(images))
print("Classes:", sorted(set(labels)))


# ==========================================================
# Preprocessing
# ==========================================================

X = np.array(images, dtype="float32") / 255.0

X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

y_cat = to_categorical(y)

print("Images Shape :", X.shape)
print("Classes :", len(encoder.classes_))

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# CNN Model
# ==========================================================

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu',
                 input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(len(encoder.classes_),activation='softmax'))

model.summary()

# ==========================================================
# Compile
# ==========================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================================
# Train
# ==========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# ==========================================================
# Test Accuracy
# ==========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy :", accuracy)

# ==========================================================
# Predictions
# ==========================================================

predictions = model.predict(X_test)

predicted = np.argmax(predictions, axis=1)

actual = np.argmax(y_test, axis=1)

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(classification_report(
    actual,
    predicted,
    target_names=encoder.classes_
))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(actual, predicted)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ==========================================================
# Accuracy & Loss Curves
# ==========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(history.history['accuracy'], label='Train')

plt.plot(history.history['val_accuracy'], label='Validation')

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.subplot(1,2,2)

plt.plot(history.history['loss'], label='Train')

plt.plot(history.history['val_loss'], label='Validation')

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()

# ==========================================================
# Show Sample Predictions
# ==========================================================

plt.figure(figsize=(15,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[i].reshape(64,64), cmap='gray')

    actual_label = encoder.inverse_transform([actual[i]])[0]

    predicted_label = encoder.inverse_transform([predicted[i]])[0]

    plt.title(f"A:{actual_label}\nP:{predicted_label}")

    plt.axis("off")

plt.tight_layout()

plt.show()

# ==========================================================
# Save Model
# ==========================================================

model.save("HandGestureRecognition.h5")

print("\nModel Saved Successfully!")