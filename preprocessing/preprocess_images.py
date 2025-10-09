import os
import cv2
import csv
import random
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ========== CONFIG ==========
INPUT_DIR = "data/single_pieces"
LABELS_FILE = "data/labels/pieces_labels.csv"
OUTPUT_DIR = "data/processed"
IMG_SIZE = 128  # resize dimension
TEST_SPLIT = 0.2  # 20% for testing

# ============================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_labels(csv_path):
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                labels[row[0]] = row[1]
    return labels

def preprocess_and_save(image_path, label, subset):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize 0–1 and save as npy
    img = img.astype(np.float32) / 255.0
    label_dir = os.path.join(OUTPUT_DIR, subset, label)
    ensure_dir(label_dir)

    filename = os.path.basename(image_path).split('.')[0] + ".npy"
    save_path = os.path.join(label_dir, filename)
    np.save(save_path, img)
    return True

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "train"))
    ensure_dir(os.path.join(OUTPUT_DIR, "test"))

    labels_dict = read_labels(LABELS_FILE)
    image_paths, labels = [], []

    # collect image paths and labels
    for img_name, label in labels_dict.items():
        img_path = os.path.join(INPUT_DIR, img_name)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(label)

    print(f"Found {len(image_paths)} labeled images.")

    # split data
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=TEST_SPLIT, stratify=labels, random_state=42
    )

    print(f"→ {len(train_paths)} for training, {len(test_paths)} for testing")

    # process train set
    print("\n📦 Preprocessing training images...")
    for img, lbl in tqdm(zip(train_paths, train_labels), total=len(train_paths)):
        preprocess_and_save(img, lbl, "train")

    # process test set
    print("\n📦 Preprocessing test images...")
    for img, lbl in tqdm(zip(test_paths, test_labels), total=len(test_paths)):
        preprocess_and_save(img, lbl, "test")

    print("\n✅ Preprocessing complete!")
    print(f"Processed data saved in: {OUTPUT_DIR}/train and {OUTPUT_DIR}/test")

if __name__ == "__main__":
    main()
# Preprocessing script
