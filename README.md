# ♟️ Chess Real-to-Animated

Turn a recording of a **real over-the-board chess game** into a **digital
animated board** that plays alongside the video, with a live move indicator and
material score.

The system takes top-down photos/frames of a physical game, detects the pieces
with a trained YOLOv8 model, reconstructs the board position for each frame, and
renders a synced side-by-side view: **real video on the left, animated board on
the right.**

---

## 1. What it does (pipeline overview)

```
photos ─▶ make video ─▶ calibrate board ─▶ dedupe + stabilize ─▶
YOLO piece detection ─▶ board reconstruction ─▶ side-by-side web replay
```

| Stage | Script | What it does |
|-------|--------|--------------|
| Video | `preprocessing/make_video.py` | Stitches the still frames into a game video |
| Calibrate | `preprocessing/calibrate_board.py` | You click the 4 board corners once per game (fixed camera) |
| Dedupe | `preprocessing/dedupe_frames.py` | Drops near-duplicate frames, ORB-stabilizes drift, builds a clean video |
| Detect + reconstruct | `inference/reconstruct_game_yolo.py` | Runs YOLO on each frame and reads the board position (FEN) |
| Train | `detection/train_detector.py` | Trains the YOLOv8 piece detector on the annotated dataset |
| Replay UI | `web/build_replay.py` | Generates a self-contained HTML page: video + animated board + score |

Helper/debug tools:
- `inference/show_detection.py` — draw the detector's boxes on one frame.
- `inference/detect_pieces.py` — print one frame's board as ASCII.
- `preprocessing/heic_to_jpg_games.py` — convert HEIC frames to JPG for annotation.

---

## 2. How it works (the approach)

**Fixed-camera calibration.** The camera stays put during a game, so the board's
4 corners are marked once and reused to flatten (perspective-warp) every frame.

**Frame alignment + deduplication.** Small camera drift between shots is corrected
with ORB feature registration, and frames where nothing changed are dropped, so
we keep one clean, stabilized frame per distinct position.

**YOLOv8 piece detection.** A YOLOv8 model (trained on a Roboflow-annotated
dataset of the pieces) detects each piece and its class. Because the pieces are
tall 3-D objects seen at an angle, each detection is mapped to a square using the
**bottom-center** of its box (where the piece meets the board), passed through the
calibration transform.

**Board reconstruction.** For each frame the detected pieces are placed on their
squares to produce a position (FEN). Board orientation is auto-detected by
matching an early frame to the standard starting position.

**Replay UI.** A self-contained HTML page plays the real video and drives an
animated board from the reconstructed positions, synced by time, with a move
indicator and a material-score bar.

---

## 3. Honest accuracy & limitations

- **Detector:** ~**0.89 mAP50** (precision ≈ recall ≈ 0.89) on the validation
  set. Most piece classes score 0.95-0.99; the weakest is black_rook (~0.82).
- **What that means:** on a ~30-piece board it reads roughly 3-5 squares
  wrong/missing per frame. So the animated board **tracks the game well on clear
  and mid-game positions but is not a frame-perfect match**, especially in
  crowded, heavily-occluded openings.
- **Root causes:** the camera is at an angle (not perfectly top-down) and pieces
  occlude each other; validation numbers are also somewhat optimistic because the
  fixed camera makes frames similar.
- The current reconstruction mode reads each frame independently (no
  chess-legality smoothing), so occasional flicker/impossible reads can appear.

**Ideas to improve:** legality-constrained tracking with re-anchoring (smooths
flicker, fixes impossible reads), more fully-annotated frames, or a truly
top-down recapture.

---

## 4. Setup

Requires Python 3.11 (tested). From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs OpenCV, PyTorch + Ultralytics (YOLOv8), python-chess, pillow-heif
(for HEIC), etc. On Apple Silicon, training/inference uses the GPU via MPS
automatically.

---

## 5. How to run

Activate the environment first: `source .venv/bin/activate`

### Full pipeline (game1)

```bash
# 1. Calibrate the 4 board corners (interactive; click TL, TR, BR, BL, press y)
python preprocessing/calibrate_board.py game1

# 2. Dedupe + build the clean, stabilized video
python preprocessing/dedupe_frames.py game1 --threshold 22

# 3. Detect pieces on every frame and reconstruct positions
python inference/reconstruct_game_yolo.py game1 detected

# 4. Build and open the side-by-side interface
python web/build_replay.py game1
```

Calibration is saved to `data/calibration/game1_corners.json`, so step 1 only
needs to be done once per game.

### Just view the current result

```bash
source .venv/bin/activate
python inference/reconstruct_game_yolo.py game1 detected
python web/build_replay.py game1        # opens web/replay_game1.html
```

### Debug a single frame

```bash
python inference/show_detection.py data/gameplay/game1/IMG_1373.HEIC 0.12
python inference/detect_pieces.py  data/gameplay/game1/IMG_1373.HEIC 180
```

---

## 6. Training the detector

Dataset (YOLOv8 format, exported from Roboflow) lives in `data/roboflow_v2/`
with 13 classes (12 pieces + `empty_board`, which is ignored downstream).

```bash
python detection/train_detector.py \
    --data data/roboflow_v2/data.yaml \
    --model yolov8s.pt --epochs 150 --name chess_detector_v2

# then promote the best weights for inference
cp runs/detect/chess_detector_v2/weights/best.pt models/trained_models/best.pt
```

**Annotation guidelines** (if you re-annotate in Roboflow):
- On full-board photos, draw a tight box around **every** piece (all ~32),
  labeled with the correct color+type.
- Single-piece photos get **one** box.
- Do **not** label empty squares or the board itself.
- Prefer variety (openings, mid-game, endgames) over near-duplicate frames.

---

## 7. Project structure

```
preprocessing/
  calibrate_board.py     # click 4 corners -> calibration
  dedupe_frames.py       # ORB align + drop duplicates + clean video
  make_video.py          # photos -> video
  extract_cells.py       # warp/crop helpers (shared library)
  heic_to_jpg_games.py   # HEIC -> JPG for annotation
detection/
  train_detector.py      # YOLOv8 training
models/
  move_detector.py       # square-mapping + diff helpers (shared library)
  trained_models/best.pt # trained detector weights (not committed)
inference/
  reconstruct_game_yolo.py  # main: detect -> reconstruct positions
  detect_pieces.py          # single-frame board (ASCII)
  show_detection.py         # single-frame detection viz
web/
  build_replay.py        # generate the side-by-side HTML
  replay_game1.html       # generated interface
data/
  gameplay/game1, game2  # raw frames (HEIC)
  single_pieces/         # single-piece photos
  roboflow_v2/           # annotated YOLOv8 dataset
  calibration/           # per-game corner calibration
  labels/                # keyframes + reconstructed states
  videos/                # generated videos (not committed)
```

---

## 8. Data notes

- **Gameplay frames:** game1 = ~106, game2 = ~196 (fixed top-down phone camera).
- **Single-piece photos:** 174.
- **Annotated dataset:** 314 images / ~2,872 boxes across train/valid/test.
- Generated artifacts (videos, cell crops, training runs, model weights, the
  dataset zip, the virtualenv) are git-ignored and regenerable.

---

## 9. Tech stack

Python 3.11 · OpenCV · PyTorch + Ultralytics YOLOv8 · python-chess · Pillow /
pillow-heif · NumPy. Web UI is plain HTML/CSS/JS (no server needed).
