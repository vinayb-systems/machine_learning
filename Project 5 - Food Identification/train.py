import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools

# 1. Setup generators (use a small subset of Food-101, e.g. 3-5 categories)
train_dir = "food_subset/train"   # folder structure: train/class_name/images...
test_dir  = "food_subset/test"

img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

# 2. Define CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10,
    verbose=1
)

# 4. Plot training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 5. Predictions
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# 6. Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 7. Show sample predictions
plt.figure(figsize=(12,6))
for i in range(10):
    img, label = test_gen[i][0][0], np.argmax(test_gen[i][1][0])
    pred = np.argmax(model.predict(img[np.newaxis,...]))
    plt.subplot(2,5,i+1)
    plt.imshow(img)
    plt.title(f"T:{class_labels[label]} P:{class_labels[pred]}")
    plt.axis('off')
plt.suptitle("Sample Predictions")
plt.show()

# 8. Misclassified examples
mis_idx = np.where(y_pred != y_true)[0][:10]
plt.figure(figsize=(12,6))
for i, idx in enumerate(mis_idx):
    img, label = test_gen[idx][0][0], y_true[idx]
    pred = y_pred[idx]
    plt.subplot(2,5,i+1)
    plt.imshow(img)
    plt.title(f"T:{class_labels[label]} P:{class_labels[pred]}")
    plt.axis('off')
plt.suptitle("Misclassified Examples")
plt.show()
