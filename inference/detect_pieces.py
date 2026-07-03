"""Detect pieces on a board photo with the trained YOLO model and map them to
chess squares, producing a board state (and FEN).

Key idea for angled photos: a piece's bounding box floats above its base
because pieces are tall 3-D objects. So we map the **bottom-center** of each
box (where the piece meets the board) through the calibration homography to
decide which square it stands on.

Usage (once weights exist at models/trained_models/best.pt):
    python inference/detect_pieces.py data/gameplay/game1/IMG_1332.HEIC
"""

import os
import sys

import chess
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.extract_cells import load_bgr, load_corners  # noqa: E402
from models.move_detector import square_rowcol  # noqa: E402

BOARD_SIZE = 800
DEFAULT_WEIGHTS = "models/trained_models/best.pt"
NON_PIECE_CLASSES = {"empty_board"}


def build_rowcol_to_square(rotate):
    return {square_rowcol(sq, rotate): sq for sq in chess.SQUARES}


def image_to_board_transform(corners):
    """Homography mapping raw-image coords -> flat 800x800 board coords."""
    src = np.array(corners, dtype="float32")
    dst = np.array([[0, 0], [BOARD_SIZE - 1, 0],
                    [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]],
                   dtype="float32")
    return cv2.getPerspectiveTransform(src, dst)


def map_point(M, x, y):
    pt = np.array([[[x, y]]], dtype="float32")
    out = cv2.perspectiveTransform(pt, M)[0][0]
    return float(out[0]), float(out[1])


def detect_board_state(model, image_path, corners, rotate, conf=0.25,
                       aligner=None, class_names=None):
    """Return {square_name: (piece_class, confidence)} for one frame."""
    bgr = load_bgr(image_path)
    if aligner is not None:
        bgr = aligner.align(bgr)

    results = model.predict(source=bgr, conf=conf, verbose=False)
    names = class_names or results[0].names
    M = image_to_board_transform(corners)
    rc_to_sq = build_rowcol_to_square(rotate)

    state = {}
    if results[0].boxes is None:
        return state
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, cf, cls = box[:6]
        piece = names[int(cls)]
        if piece in NON_PIECE_CLASSES:
            continue
        # bottom-center of the box = where the piece stands
        bx, by = (x1 + x2) / 2.0, y2
        u, v = map_point(M, bx, by)
        col = int(u // (BOARD_SIZE / 8))
        row = int(v // (BOARD_SIZE / 8))
        if not (0 <= row < 8 and 0 <= col < 8):
            continue
        sq = rc_to_sq.get((row, col))
        if sq is None:
            continue
        name = chess.square_name(sq)
        # keep the higher-confidence detection if two land on one square
        if name not in state or cf > state[name][1]:
            state[name] = (piece, float(cf))
    return state


def state_to_board(state):
    """Build a chess.Board from a {square: (piece, conf)} placement."""
    board = chess.Board(None)  # empty board
    sym = {
        "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
        "white_rook": "R", "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
        "black_rook": "r", "black_queen": "q", "black_king": "k",
    }
    for name, (piece, _) in state.items():
        if piece in sym:
            board.set_piece_at(chess.parse_square(name),
                               chess.Piece.from_symbol(sym[piece]))
    return board


def print_ascii(state):
    board = state_to_board(state)
    print(board)


if __name__ == "__main__":
    from ultralytics import YOLO

    weights = DEFAULT_WEIGHTS
    if not os.path.exists(weights):
        sys.exit(f"❌ Weights not found at {weights} (training may still be running).")
    img = sys.argv[1] if len(sys.argv) > 1 else "data/gameplay/game1/IMG_1332.HEIC"
    rotate = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    model = YOLO(weights)
    corners = load_corners("game1")
    st = detect_board_state(model, img, corners, rotate)
    print(f"Detected {len(st)} pieces on {os.path.basename(img)} (rotate={rotate}):")
    print_ascii(st)
