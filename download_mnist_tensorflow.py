"""
MNIST Dataset Download using TensorFlow
Alternative method to download MNIST dataset
"""

import os
import numpy as np
import struct

def save_mnist_to_idx_format():
    """Download MNIST using TensorFlow and save in IDX format"""
    
    try:
        import tensorflow as tf
        print("Downloading MNIST using TensorFlow...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print("Download completed!")
    except ImportError:
        print("ERROR: TensorFlow not installed. Please install it with: pip install tensorflow")
        return False
    
    # Create folder
    dataset_dir = "MNIST-dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Folder created: {dataset_dir}")
    
    def save_images(filename, images):
        """Save images in IDX format"""
        with open(filename, 'wb') as f:
            # Write header
            f.write(struct.pack('>IIII', 2051, len(images), 28, 28))
            # Write image data
            images = images.astype(np.uint8)
            f.write(images.tobytes())
        print(f"Saved: {filename}")
    
    def save_labels(filename, labels):
        """Save labels in IDX format"""
        with open(filename, 'wb') as f:
            # Write header
            f.write(struct.pack('>II', 2049, len(labels)))
            # Write label data
            labels = labels.astype(np.uint8)
            f.write(labels.tobytes())
        print(f"Saved: {filename}")
    
    # Save files
    print("\nSaving files in IDX format...")
    save_images(os.path.join(dataset_dir, "train-images.idx3-ubyte"), x_train)
    save_labels(os.path.join(dataset_dir, "train-labels.idx1-ubyte"), y_train)
    save_images(os.path.join(dataset_dir, "t10k-images.idx3-ubyte"), x_test)
    save_labels(os.path.join(dataset_dir, "t10k-labels.idx1-ubyte"), y_test)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All files saved!")
    print("=" * 60)
    print(f"\nFiles are in: {os.path.abspath(dataset_dir)}")
    
    return True

if __name__ == "__main__":
    success = save_mnist_to_idx_format()
    if success:
        print("\n[SUCCESS] You can now run 'python q1_train.py' to start training!")
    else:
        print("\n[ERROR] Download failed.")

