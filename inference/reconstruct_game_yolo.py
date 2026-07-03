"""Reconstruct a game from frames using YOLO detections + chess rules.

We do NOT rely on perfect full-board detection. Instead we track the position
from the standard start and, at each keyframe, pick the single legal move whose
resulting position best matches whatever the detector saw. Missed pieces simply
don't vote; chess legality + the known previous position do the rest.

Prerequisites:
    - models/trained_models/best.pt (trained YOLO detector)
    - data/calibration/<game>_corners.json (calibration)
    - data/labels/<game>_keyframes.json (deduplicated frames) [optional]

Usage:
    python inference/reconstruct_game_yolo.py game1
Output:
    data/labels/<game>_states_yolo.json  (per-frame FEN + reconstructed PGN)
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
from models.move_detector import square_rowcol, square_diff_scores  # noqa: E402

GAMEPLAY_DIR = "data/gameplay"
LABELS_DIR = "data/labels"
BOARD_SIZE = 800
WEIGHTS = "models/trained_models/best.pt"
NON_PIECE = {"empty_board"}

SYM = {"white_pawn": "P", "white_knight": "N", "white_bishop": "B",
       "white_rook": "R", "white_queen": "Q", "white_king": "K",
       "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
       "black_rook": "r", "black_queen": "q", "black_king": "k"}
CLASS_OF = {v: k for k, v in SYM.items()}


def piece_class_at(board, square):
    p = board.piece_at(square)
    return CLASS_OF.get(p.symbol()) if p else None


def frames_for(game):
    gd = os.path.join(GAMEPLAY_DIR, game)
    kf = os.path.join(LABELS_DIR, f"{game}_keyframes.json")
    if os.path.exists(kf):
        frames = json.load(open(kf))["kept"]
    else:
        frames = sorted(f for f in os.listdir(gd) if f.lower().endswith(
            (".heic", ".jpg", ".jpeg", ".png")))
    # keep only frames that still exist on disk
    existing = [f for f in frames if os.path.exists(os.path.join(gd, f))]
    dropped = len(frames) - len(existing)
    if dropped:
        print(f"⚠️  Skipping {dropped} missing frame(s).")
    return existing


def calib_matrix(corners):
    dst = np.array([[0, 0], [BOARD_SIZE - 1, 0],
                    [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]], "float32")
    return cv2.getPerspectiveTransform(np.array(corners, "float32"), dst)


def detect_rowcol(model, bgr, aligner, M, conf=0.12):
    """Return list of (row, col, piece_class, conf) for one raw frame."""
    res = model.predict(source=bgr, conf=conf, verbose=False, device="cpu")
    names = res[0].names
    if res[0].boxes is None:
        return []
    H = aligner.homography(bgr)  # frame -> reference; None if ref itself
    out = []
    for b in res[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, cf, cls = b[:6]
        piece = names[int(cls)]
        if piece in NON_PIECE:
            continue
        pt = np.array([[[(x1 + x2) / 2.0, y2]]], "float32")  # bottom-center
        if H is not None:
            pt = cv2.perspectiveTransform(pt, H)
        pt = cv2.perspectiveTransform(pt, M)[0][0]
        col, row = int(pt[0] // (BOARD_SIZE / 8)), int(pt[1] // (BOARD_SIZE / 8))
        if 0 <= row < 8 and 0 <= col < 8:
            out.append((row, col, piece, float(cf)))
    return out


def state_for_rotation(dets, rotate):
    """Map (row,col) detections to {square: (piece, conf)} for a rotation."""
    rc_to_sq = {square_rowcol(sq, rotate): sq for sq in chess.SQUARES}
    state = {}
    for row, col, piece, cf in dets:
        sq = rc_to_sq.get((row, col))
        if sq is None:
            continue
        if sq not in state or cf > state[sq][1]:
            state[sq] = (piece, cf)
    return state


def score_board(board, state):
    """How many detected pieces agree with this board position."""
    s = 0.0
    for sq, (piece, cf) in state.items():
        actual = piece_class_at(board, sq)
        if actual == piece:
            s += 1.0
        elif actual is not None:
            s -= 0.5  # detected a different piece here
    return s


def change_bonus(board, move, change_grid, rotate):
    """Reward moves whose from/to squares actually changed in the pixels."""
    if change_grid is None:
        return 0.0
    fr = square_rowcol(move.from_square, rotate)
    to = square_rowcol(move.to_square, rotate)
    return (change_grid[fr] + change_grid[to]) / 2.0


def track(all_dets, changes, rotate, alpha=0.2):
    """Reconstruct moves for a rotation using detections + change region."""
    board = chess.Board()
    fens = [board.fen()]
    moves = []
    total = 0.0
    for idx in range(1, len(all_dets)):
        state = state_for_rotation(all_dets[idx], rotate)
        chg = changes[idx - 1] if changes else None
        best_move, best_score = None, -1e9
        for m in board.legal_moves:
            board.push(m)
            sc = score_board(board, state)
            board.pop()
            sc += alpha * change_bonus(board, m, chg, rotate)
            if sc > best_score:
                best_move, best_score = m, sc
        if best_move is None:
            break
        board.push(best_move)
        moves.append(best_move.uci())
        fens.append(board.fen())
        total += best_score
    return moves, fens, total


def board_from_dets(dets, rotate):
    """Build a placement-only board directly from detections (best piece/square)."""
    rc_to_sq = {square_rowcol(sq, rotate): sq for sq in chess.SQUARES}
    best = {}  # square -> (piece, conf)
    for row, col, piece, cf in dets:
        sq = rc_to_sq.get((row, col))
        if sq is None:
            continue
        if sq not in best or cf > best[sq][1]:
            best[sq] = (piece, cf)
    board = chess.Board(None)
    for sq, (piece, _) in best.items():
        if piece in SYM:
            board.set_piece_at(sq, chess.Piece.from_symbol(SYM[piece]))
    return board


def plausibility(board):
    """Higher = more like a real chess position (used to pick orientation)."""
    wk = len(board.pieces(chess.KING, chess.WHITE))
    bk = len(board.pieces(chess.KING, chess.BLACK))
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    score = 0
    score += 3 if wk == 1 else -2
    score += 3 if bk == 1 else -2
    score += 1 if wp <= 8 else -1
    score += 1 if bp <= 8 else -1
    # pawns not on rank 1 or 8
    for sq in list(board.pieces(chess.PAWN, chess.WHITE)) + list(board.pieces(chess.PAWN, chess.BLACK)):
        r = chess.square_rank(sq)
        if r in (0, 7):
            score -= 1
    return score


def reconstruct_detected(game):
    """Read each frame's board directly from detections (no fragile tracking)."""
    from ultralytics import YOLO
    if not os.path.exists(WEIGHTS):
        sys.exit(f"❌ Weights not found: {WEIGHTS}")
    model = YOLO(WEIGHTS)
    corners = load_corners(game)
    M = calib_matrix(corners)
    frames = frames_for(game)
    gd = os.path.join(GAMEPLAY_DIR, game)
    aligner = Aligner(load_bgr(os.path.join(gd, frames[0])))

    print(f"🔎 Detecting pieces on {len(frames)} frames ...")
    all_dets = []
    for i, f in enumerate(frames):
        all_dets.append(detect_rowcol(model, load_bgr(os.path.join(gd, f)), aligner, M))
        if (i + 1) % 25 == 0:
            print(f"   {i + 1}/{len(frames)}")

    # pick orientation by matching early full-board frames to the standard start
    start = chess.Board()
    early = [d for d in all_dets[1:15] if len(d) > 15][:8]
    best_rot, best_score = 0, -1e9
    for rot in (0, 90, 180, 270):
        s = 0
        for d in early:
            b = board_from_dets(d, rot)
            s += sum(1 for sq in chess.SQUARES if b.piece_at(sq) == start.piece_at(sq))
        print(f"   rotation {rot:3d}: start-match {s}")
        if s > best_score:
            best_rot, best_score = rot, s
    print(f"✅ Using rotation {best_rot}")

    # anchor first frame to the standard start (it's crowded/hard to detect)
    fens = []
    for i, dets in enumerate(all_dets):
        if i == 0:
            fens.append(chess.Board().fen())
            continue
        b = board_from_dets(dets, best_rot)
        fens.append(b.fen() if b.piece_map() else fens[-1])

    os.makedirs(LABELS_DIR, exist_ok=True)
    path = os.path.join(LABELS_DIR, f"{game}_states_yolo.json")
    json.dump({"game": game, "rotate": best_rot, "mode": "detected",
               "frames": [{"stem": os.path.splitext(f)[0], "fen": fen}
                          for f, fen in zip(frames, fens)],
               "pgn": ""}, open(path, "w"), indent=2)
    print(f"💾 Saved {path} ({len(fens)} detected positions)")


def reconstruct(game):
    from ultralytics import YOLO
    if not os.path.exists(WEIGHTS):
        sys.exit(f"❌ Weights not found: {WEIGHTS}")
    model = YOLO(WEIGHTS)
    corners = load_corners(game)
    M = calib_matrix(corners)
    frames = frames_for(game)
    gd = os.path.join(GAMEPLAY_DIR, game)
    aligner = Aligner(load_bgr(os.path.join(gd, frames[0])))

    print(f"🔎 Detecting pieces on {len(frames)} frames ...")
    all_dets = []
    grays = []
    for i, f in enumerate(frames):
        bgr = load_bgr(os.path.join(gd, f))
        all_dets.append(detect_rowcol(model, bgr, aligner, M))
        warped = warp_board(aligner.align(bgr), corners, BOARD_SIZE)
        grays.append(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
        if (i + 1) % 20 == 0:
            print(f"   {i + 1}/{len(frames)}")

    # per-transition pixel-change grids (8x8), reliable after alignment
    changes = [square_diff_scores(grays[i - 1], grays[i]) for i in range(1, len(grays))]

    best = None
    for rot in (0, 90, 180, 270):
        moves, fens, score = track(all_dets, changes, rot)
        print(f"   rotation {rot:3d}: score {score:.0f}, {len(moves)} moves")
        if best is None or score > best["score"]:
            best = {"rotate": rot, "moves": moves, "fens": fens, "score": score}

    # SAN move list
    board = chess.Board()
    san = []
    for uci in best["moves"]:
        mv = chess.Move.from_uci(uci)
        san.append(board.san(mv))
        board.push(mv)
    out = " ".join((f"{i//2+1}." if i % 2 == 0 else "") + s for i, s in enumerate(san))
    print(f"\n✅ rotation {best['rotate']}, {len(best['moves'])} moves:\n{out}\n")

    os.makedirs(LABELS_DIR, exist_ok=True)
    path = os.path.join(LABELS_DIR, f"{game}_states_yolo.json")
    json.dump({"game": game, "rotate": best["rotate"],
               "frames": [{"stem": os.path.splitext(f)[0], "fen": fen}
                          for f, fen in zip(frames, best["fens"])],
               "pgn": out}, open(path, "w"), indent=2)
    print(f"💾 Saved {path}")
    return best


if __name__ == "__main__":
    args = sys.argv[1:]
    game = "game1"
    mode = "detected"
    for a in args:
        if a in ("detected", "tracked"):
            mode = a
        else:
            game = a
    if mode == "detected":
        reconstruct_detected(game)
    else:
        reconstruct(game)
