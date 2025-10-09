import os
import csv

# === Folder hierarchy ===
structure = {
    "data": [
        "single_pieces",
        "gameplay/game1",
        "gameplay/game2",
        "labels",
        "raw"
    ],
    "preprocessing": [],
    "models": ["trained_models"],
    "utils": [],
    "notebooks": [],
    "results": ["predictions"]
}

# === CSV label files ===
labels = {
    "data/labels/pieces_labels.csv": ["image", "label"],
    "data/labels/game_labels.csv": ["image", "from", "to"]
}

# === Placeholder scripts & files ===
placeholder_files = {
    "preprocessing/preprocess_images.py": "# Preprocessing script\n",
    "preprocessing/data_split.py": "# Split dataset into train/test\n",
    "models/piece_classifier.py": "# CNN model for piece classification\n",
    "models/move_detector.py": "# Logic for move detection\n",
    "utils/board_utils.py": "# Helper functions for board mapping\n",
    "utils/image_utils.py": "# Helper functions for OpenCV operations\n",
    "utils/visualization.py": "# For plotting predictions and results\n",
    "notebooks/piece_training.ipynb": "",
    "notebooks/move_detection.ipynb": "",
    "requirements.txt": "opencv-python\ntensorflow\nnumpy\nmatplotlib\npandas\n",
    "README.md": "# Chess Vision Project\n\nThis project detects chess moves and recognizes board states using computer vision.\n"
}

# === Create folders ===
print("📂 Creating project structure...\n")
for root, subs in structure.items():
    os.makedirs(root, exist_ok=True)
    for sub in subs:
        os.makedirs(os.path.join(root, sub), exist_ok=True)
print("✅ Folder structure created.\n")

# === Create label CSVs ===
for path, headers in labels.items():
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"[+] Created {path}")

# === Create placeholder files ===
for file_path, content in placeholder_files.items():
    with open(file_path, "w") as f:
        f.write(content)
    print(f"[+] Created {file_path}")

print("\n🎉 All done! Your chess_vision_project is ready.")
