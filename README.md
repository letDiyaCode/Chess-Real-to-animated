♟️ Chess Real-to-Animated

A Computer Vision System that Detects Real-World Chess Moves and Recreates Them Digitally

⚡ Quickstart Setup
# 1️⃣ Clone the Repository
git clone https://github.com/diya173/Chess-Real-to-animated.git
cd Chess-Real-to-animated

# 2️⃣ Create & Activate Conda Environment
conda env create -f environment.yml
conda activate chess_vision

# 3️⃣ (Optional) Run Labeling Scripts
python data/labels/label_single_pieces.py
python data/labels/label_game_moves.py


That’s it! ✅
You’re ready to preprocess images, train the piece classifier, and test move detection.

📘 Overview

This project aims to bridge real chess play and digital animation.
Using a top-down camera view, we capture real chessboard states, detect piece movements, and replicate them on a virtual/animated chessboard in real time.

It combines computer vision, deep learning, and chess logic to automate move recognition and visualization.

🧠 Project Pipeline
1️⃣ Data Collection

Images captured using a fixed top-view camera setup.

~400 images collected:

Single-piece dataset → for piece classification.

Gameplay dataset → for move detection between board states.

Data includes:

Varying lighting conditions.

Multiple board states and game progressions.

2️⃣ Data Organization

Project structured into clean modules:

chess_vision_project/
├── data/
│   ├── single_pieces/           # images of individual chess pieces
│   ├── gameplay/                # game sequences (multiple frames)
│   ├── labels/                  # labeling CSVs
│   └── raw/                     # backup/raw data
├── preprocessing/               # scripts for resizing, normalization, etc.
├── models/                      # CNNs for piece and move detection
├── utils/                       # helper functions (OpenCV, mapping)
├── notebooks/                   # Jupyter notebooks for experiments
├── results/                     # model outputs, metrics, visuals
├── environment.yml              # conda environment setup
└── README.md                    # project documentation

🧩 Methodology
Stage	Description	Tools
Piece Detection	Classify each cell of the chessboard into piece type (or empty)	TensorFlow / Keras CNN
Board Segmentation	Detect 8×8 grid using OpenCV corner detection	OpenCV
Move Detection	Compare consecutive board states to find moved piece	Numpy + Board State Diff
Animated Visualization	Display detected moves on a digital chessboard	Python / Pygame (future scope)
⚙️ Tech Stack
Category	Tools
Languages	Python
Libraries	OpenCV, TensorFlow, NumPy, Pandas, Matplotlib
Environment	Conda (Python 3.10)
Data Handling	CSV, JSON
IDE	VS Code / Jupyter Notebook
Version Control	Git + GitHub
🧰 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/diya173/Chess-Real-to-animated.git
cd Chess-Real-to-animated

2️⃣ Create the Conda Environment
conda env create -f environment.yml
conda activate chess_vision

3️⃣ (Optional) Install Manually
pip install opencv-python tensorflow numpy pandas matplotlib scikit-learn

🧾 Dataset Labeling
Single Piece Labeling

Run:

python data/labels/label_single_pieces.py


You’ll be shown each image and asked to type the correct piece name.

Gameplay Move Labeling

Run:

python data/labels/label_game_moves.py


You’ll input the from and to squares (e.g., e2 e4) for each board image.

🚀 Future Goals

 Automate board detection from any angle.

 Integrate YOLO-based piece localization.

 Create real-time move-to-animation pipeline.

 Build a React or Godot-based front-end for visual replay.

 Add sound, timestamps, and player tracking.# Chess Vision Project

This project detects chess moves and recognizes board states using computer vision.
