"""Detect chess moves from a sequence of board frames (no ML required).

Approach (works because the camera is fixed and the game starts from the
standard position):
  1. Warp every frame to a flat 800x800 board using the saved calibration.
  2. For each consecutive pair of frames, measure how much each of the 64
     squares changed (pixel difference) -> the moved piece's from/to squares
     change the most.
  3. Among the *legal* moves in the current position (python-chess), pick the
     one whose involved squares best match the changed squares.
  4. Push it, and repeat -> full game as a sequence of FENs + a PGN.

Board orientation (which corner is a1) is auto-detected by trying all four
rotations and keeping the one whose detected moves match the pixel changes
best.
"""

import json
import os
import sys

import chess
import chess.pgn
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.extract_cells import load_bgr, load_corners, warp_board  # noqa: E402
from preprocessing.dedupe_frames import Aligner  # noqa: E402

GAMEPLAY_DIR = "data/gameplay"
LABELS_DIR = "data/labels"
IMG_EXTS = (".heic", ".jpg", ".jpeg", ".png")
BOARD_SIZE = 800
NO_MOVE_THRESHOLD = 12.0  # mean-abs-diff below this on the busiest square => no move


def square_rowcol(square, rotate):
    """Map a python-chess square (0..63) to (row, col) in the warped board."""
    f = chess.square_file(square)
    r = chess.square_rank(square)
    row, col = 7 - r, f  # rotate 0: white at bottom, a-file left
    if rotate == 90:
        row, col = col, 7 - row
    elif rotate == 180:
        row, col = 7 - row, 7 - col
    elif rotate == 270:
        row, col = 7 - col, row
    return row, col


def warp_gray(path, corners, aligner=None):
    bgr = load_bgr(path)
    if aligner is not None:
        bgr = aligner.align(bgr)
    warped = warp_board(bgr, corners, BOARD_SIZE)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


def square_diff_scores(prev_gray, cur_gray, inset_frac=0.18):
    """8x8 array of mean abs-difference per square (inset to skip grid lines)."""
    step = BOARD_SIZE / 8.0
    inset = int(step * inset_frac)
    d = cv2.absdiff(prev_gray, cur_gray)
    scores = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            y1, y2 = int(r * step) + inset, int((r + 1) * step) - inset
            x1, x2 = int(c * step) + inset, int((c + 1) * step) - inset
            scores[r, c] = float(d[y1:y2, x1:x2].mean())
    return scores


def involved_squares(board, move):
    """Squares whose contents change when `move` is played."""
    sqs = {move.from_square, move.to_square}
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            sqs |= {chess.square(7, rank), chess.square(5, rank)}  # king-side rook
        else:
            sqs |= {chess.square(0, rank), chess.square(3, rank)}  # queen-side rook
    if board.is_en_passant(move):
        sqs.add(chess.square(chess.square_file(move.to_square),
                             chess.square_rank(move.from_square)))
    return sqs


def pick_move(board, scores, rotate):
    """Return (best_move, confidence) for the legal move matching the changes."""
    best, best_score = None, -1.0
    for m in board.legal_moves:
        vals = [scores[square_rowcol(sq, rotate)] for sq in involved_squares(board, m)]
        s = sum(vals) / len(vals)
        if s > best_score:
            best, best_score = m, s
    return best, best_score


def list_frames(game):
    """Prefer the deduplicated keyframes if available, else all frames."""
    game_dir = os.path.join(GAMEPLAY_DIR, game)
    kf_path = os.path.join(LABELS_DIR, f"{game}_keyframes.json")
    if os.path.exists(kf_path):
        with open(kf_path) as f:
            frames = json.load(f)["kept"]
        print(f"📑 Using {len(frames)} deduplicated keyframes.")
        return game_dir, frames
    frames = sorted(f for f in os.listdir(game_dir) if f.lower().endswith(IMG_EXTS))
    if not frames:
        sys.exit(f"❌ No frames in {game_dir}")
    return game_dir, frames


def reconstruct(game, rotate, grays=None, frames=None, verbose=False):
    """Reconstruct the game for a given rotation.

    Returns dict with board, fens (per frame), moves (uci), and mean confidence.
    """
    if grays is None:
        corners = load_corners(game)
        game_dir, frames = list_frames(game)
        grays = [warp_gray(os.path.join(game_dir, f), corners) for f in frames]

    board = chess.Board()
    fens = [board.fen()]
    moves = []
    confidences = []

    for i in range(1, len(grays)):
        scores = square_diff_scores(grays[i - 1], grays[i])
        if scores.max() < NO_MOVE_THRESHOLD:
            fens.append(board.fen())  # duplicate / no change
            continue
        move, conf = pick_move(board, scores, rotate)
        if move is None:  # no legal moves (game over) -> stop
            break
        board.push(move)
        fens.append(board.fen())
        moves.append(move.uci())
        confidences.append(conf)
        if verbose:
            print(f"  frame {i}: {move.uci()} (conf {conf:.1f})")

    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return {"board": board, "fens": fens, "moves": moves,
            "mean_conf": mean_conf, "frames": frames}


def detect_game(game, verbose=True):
    """Auto-detect orientation and reconstruct the full game."""
    corners = load_corners(game)
    game_dir, frames = list_frames(game)
    print(f"🎞️  Aligning + warping {len(frames)} frames for {game} ...")
    aligner = Aligner(load_bgr(os.path.join(game_dir, frames[0])))
    grays = [warp_gray(os.path.join(game_dir, f), corners, aligner) for f in frames]

    # Probe the 4 rotations on the first chunk; keep the most confident.
    probe_n = min(len(grays), 25)
    best_rot, best_conf = 0, -1.0
    for rot in (0, 90, 180, 270):
        r = reconstruct(game, rot, grays=grays[:probe_n], frames=frames[:probe_n])
        print(f"   rotation {rot:3d}: mean confidence {r['mean_conf']:.1f}")
        if r["mean_conf"] > best_conf:
            best_rot, best_conf = rot, r["mean_conf"]
    print(f"✅ Using rotation {best_rot}")

    result = reconstruct(game, best_rot, grays=grays, frames=frames, verbose=verbose)
    result["rotate"] = best_rot
    return result


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "game1"
    res = detect_game(g, verbose=True)
    print(f"\n♟️  Detected {len(res['moves'])} moves (rotation {res['rotate']}).")
    game_pgn = chess.pgn.Game() if False else None
    board = chess.Board()
    san = []
    for uci in res["moves"]:
        mv = chess.Move.from_uci(uci)
        san.append(board.san(mv))
        board.push(mv)
    # number the moves
    out = []
    for idx, s in enumerate(san):
        if idx % 2 == 0:
            out.append(f"{idx // 2 + 1}.{s}")
        else:
            out.append(s)
    print(" ".join(out))
