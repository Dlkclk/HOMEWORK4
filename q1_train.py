"""
Q1: Section 10.9 Application: Handwritten Digit Recognition from Digital Images
Single Neuron Classifier for MNIST Dataset
Classes: "zero" vs "not zero"
"""

import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from mnist import load_images, load_labels
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

# Normalize features
print("\nNormalizing features...")
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)
train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# Create single neuron model
print("\nCreating single neuron model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[7], activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

# Prepare labels: 0 -> 0, others -> 1
print("\nPreparing labels (0 vs not-zero)...")
train_labels_binary = train_labels.copy()
test_labels_binary = test_labels.copy()
train_labels_binary[train_labels_binary != 0] = 1
test_labels_binary[test_labels_binary != 0] = 1

print(f"Train labels distribution: {np.bincount(train_labels_binary)}")
print(f"Test labels distribution: {np.bincount(test_labels_binary)}")

# Train the model
print("\nTraining the model...")
history = model.fit(
    train_huMoments,
    train_labels_binary,
    batch_size=128,
    epochs=50,
    class_weight={0: 8, 1: 1},
    verbose=1
)

# Make predictions
print("\nMaking predictions on test set...")
perceptron_preds = model.predict(test_huMoments)

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels_binary, perceptron_preds > 0.5)
print("\nConfusion Matrix:")
print(conf_matrix)

# Display confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot()
cm_display.ax_.set_title("Single Neuron Classifier Confusion Matrix")
plt.tight_layout()
plt.savefig("q1_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'q1_confusion_matrix.png'")
plt.show()

# Save the model
model_path = "mnist_single_neuron.h5"
model.save(model_path)
print(f"\nModel saved as '{model_path}'")

# Print final metrics
test_loss, test_accuracy = model.evaluate(test_huMoments, test_labels_binary, verbose=0)
print(f"\nFinal Test Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

