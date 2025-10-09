import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from pillow_heif import register_heif_opener

# Enable HEIC support
register_heif_opener()

# ========== CONFIG ==========
MODEL_PATH = "models/trained_models/piece_classifier.h5"
LABEL_ENCODER_PATH = "models/trained_models/label_encoder_classes.npy"
IMG_SIZE = 128
# ============================


def load_model_and_labels():
    """Load trained CNN and label encoder."""
    print("📦 Loading model and label classes...")
    model = load_model(MODEL_PATH)
    class_names = np.load(LABEL_ENCODER_PATH)
    print(f"✅ Loaded model with {len(class_names)} classes.")
    return model, class_names


def detect_and_warp_board(image):
    """Detect chessboard corners and warp image to top view."""
    print("📐 Detecting board corners...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    board_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:  # Found a quadrilateral
            board_contour = approx
            break

    if board_contour is None:
        print("⚠️ Chessboard corners not found. Using full image.")
        return image

    # Sort points (top-left, top-right, bottom-right, bottom-left)
    pts = board_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    print("✅ Board corrected to top view.")
    return warped


def split_board_into_cells(board_img):
    """Divide a top-view chessboard image into 64 equal cells."""
    h, w, _ = board_img.shape
    cell_h, cell_w = h // 8, w // 8
    cells = []

    for i in range(8):
        row = []
        for j in range(8):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = board_img[y1:y2, x1:x2]
            row.append(cell)
        cells.append(row)
    return cells


def preprocess_cell(cell):
    """Resize and normalize a single cell."""
    cell = cv2.resize(cell, (IMG_SIZE, IMG_SIZE))
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    cell = cell.astype(np.float32) / 255.0
    return np.expand_dims(cell, axis=0)


def predict_board_state(board_img_path):
    """Predict the piece in each of the 64 cells of the board."""
    model, class_names = load_model_and_labels()

    # Load image (supports JPG, PNG, HEIC)
    try:
        img = Image.open(board_img_path).convert("RGB")
        board_img = np.array(img)
        board_img = cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Error: Could not load board image ({e})")
        return

    # Detect and warp board
    board_img = detect_and_warp_board(board_img)

    # Split into 64 cells
    cells = split_board_into_cells(board_img)
    print(f"♟️ Predicting pieces on {os.path.basename(board_img_path)}...\n")

    board_predictions = []
    for i, row in enumerate(cells):
        row_preds = []
        for j, cell in enumerate(row):
            preprocessed = preprocess_cell(cell)
            pred = model.predict(preprocessed, verbose=0)
            label = class_names[np.argmax(pred)]
            row_preds.append(label)
        board_predictions.append(row_preds)

    # Print results in board form
    print("🧩 Predicted Board Layout (Top View):\n")
    for row in board_predictions:
        print(" | ".join(f"{p:>12}" for p in row))
        print("-" * 120)
    print("\n")

    return board_predictions


if __name__ == "__main__":
    TEST_DIR = "test_images"

    if not os.path.exists(TEST_DIR):
        print("⚠️ Please create a folder named 'test_images' and add your test images there.")
    else:
        images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
        if not images:
            print("⚠️ No test images found in 'test_images' folder.")
        else:
            print(f"🧩 Found {len(images)} test images.\n")
            for img_name in images:
                img_path = os.path.join(TEST_DIR, img_name)
                print(f"🔍 Testing on: {img_name}\n")
                predict_board_state(img_path)
                print("\n" + "=" * 120 + "\n")
