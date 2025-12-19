"""
Q2: Section 11.8 Application: Handwritten Digit Recognition from Digital Images
Multilayer Neural Network for MNIST Dataset
Classes: 0-9 (10 classes - multi-class classification)
"""

import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mnist import load_images, load_labels
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

# MNIST dataset paths
train_img_path = os.path.join("MNIST-dataset", "train-images.idx3-ubyte")
train_label_path = os.path.join("MNIST-dataset", "train-labels.idx1-ubyte")
test_img_path = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")
test_label_path = os.path.join("MNIST-dataset", "t10k-labels.idx1-ubyte")

# Load MNIST data
print("Loading MNIST dataset...")
train_images = load_images(train_img_path)
train_labels = load_labels(train_label_path)
test_images = load_images(test_img_path)
test_labels = load_labels(test_label_path)
print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")

# Extract Hu moments features
print("\nExtracting Hu moments features...")
train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True)
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True)
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

print("Feature extraction completed!")

# Normalize features (optional but recommended)
print("\nNormalizing features...")
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)
train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# Create multilayer neural network model
print("\nCreating multilayer neural network model...")
model = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=[7], activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Get unique categories
categories = np.unique(test_labels)
print(f"Categories: {categories}")
print(f"Number of classes: {len(categories)}")

# Compile the model
print("\nCompiling model...")
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(1e-4)
)

# Print model summary
print("\nModel Summary:")
model.summary()

# Setup callbacks
print("\nSetting up callbacks...")
mc_callback = ModelCheckpoint("mlp_mnist_model.h5", monitor='loss', save_best_only=True, verbose=1)
es_callback = EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model
print("\nTraining the model...")
print("This may take a while. Training will stop early if loss doesn't improve for 5 epochs.")
history = model.fit(
    train_huMoments, 
    train_labels, 
    epochs=1000, 
    verbose=1, 
    callbacks=[mc_callback, es_callback]
)

# Make predictions
print("\nMaking predictions on test set...")
nn_preds = model.predict(test_huMoments)
predicted_classes = np.argmax(nn_preds, axis=1)

# Calculate confusion matrix
print("\nCalculating confusion matrix...")
conf_matrix = confusion_matrix(test_labels, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Display confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
cm_display.plot()
cm_display.ax_.set_title("Neural Network Confusion Matrix")
plt.tight_layout()
plt.savefig("q2_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'q2_confusion_matrix.png'")
plt.show()

# Print per-class accuracy
print("\nPer-class Accuracy:")
for i, category in enumerate(categories):
    class_correct = conf_matrix[i, i]
    class_total = np.sum(conf_matrix[i, :])
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    print(f"Class {category}: {class_correct}/{class_total} = {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

print(f"\nModel saved as 'mlp_mnist_model.h5'")
print("Training completed!")

