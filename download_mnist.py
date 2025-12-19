"""
MNIST Dataset Auto Download Script
Downloads MNIST dataset from official website and extracts to MNIST-dataset folder
"""

import os
import sys
import urllib.request
import gzip
import shutil

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def download_mnist():
    """Download and extract MNIST dataset"""
    
    # Create folder
    dataset_dir = "MNIST-dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Folder created: {dataset_dir}")
    
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        ("train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte")
    ]
    
    print("=" * 60)
    print("Downloading MNIST Dataset...")
    print("=" * 60)
    
    for gz_file, extracted_file in files:
        gz_path = os.path.join(dataset_dir, gz_file)
        extracted_path = os.path.join(dataset_dir, extracted_file)
        
        # Skip if file already exists
        if os.path.exists(extracted_path):
            print(f"[OK] {extracted_file} already exists, skipping...")
            continue
        
        try:
            print(f"\n[DOWNLOAD] {gz_file}...")
            urllib.request.urlretrieve(base_url + gz_file, gz_path)
            print(f"   [OK] Download completed")
            
            print(f"[EXTRACT] {extracted_file}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"   [OK] Extraction completed")
            
            # Delete gzip file
            os.remove(gz_path)
            print(f"   [OK] Temporary file deleted")
            
        except Exception as e:
            print(f"   [ERROR] {e}")
            if os.path.exists(gz_path):
                os.remove(gz_path)
            return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All files downloaded and extracted!")
    print("=" * 60)
    print(f"\nFiles are in: {os.path.abspath(dataset_dir)}")
    
    return True

if __name__ == "__main__":
    success = download_mnist()
    if success:
        print("\n[SUCCESS] You can now run 'python q1_train.py' to start training!")
    else:
        print("\n[ERROR] Download failed. Please check your internet connection.")

