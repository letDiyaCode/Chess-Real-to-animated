"""Warp every frame of a game using calibrated corners and slice into 64 cells.

Prerequisite:
    Run calibrate_board.py for the game first to create
    data/calibration/<game>_corners.json

Usage:
    python preprocessing/extract_cells.py game1
    python preprocessing/extract_cells.py game1 --board-size 800 --pad 0.0

Output:
    data/cells/<game>/<frame_stem>/r{row}c{col}.jpg
      row 0 = top of the warped board, col 0 = left of the warped board.

A manifest is also written to data/cells/<game>/manifest.json listing all
frames and the warped board size, so the labeling step can map (row,col)
cells to chess squares.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()

GAMEPLAY_DIR = "data/gameplay"
CALIB_DIR = "data/calibration"
CELLS_DIR = "data/cells"
IMG_EXTS = (".heic", ".jpg", ".jpeg", ".png")


def load_corners(game):
    path = os.path.join(CALIB_DIR, f"{game}_corners.json")
    if not os.path.exists(path):
        sys.exit(f"❌ No calibration found at {path}. Run calibrate_board.py first.")
    with open(path) as f:
        data = json.load(f)
    pts = np.array(data["corners"], dtype="float32")  # TL, TR, BR, BL
    if pts.shape != (4, 2):
        sys.exit("❌ Calibration file is malformed (need 4 corner points).")
    return pts


def load_bgr(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def warp_board(img, src_pts, board_size):
    dst = np.array(
        [[0, 0], [board_size - 1, 0],
         [board_size - 1, board_size - 1], [0, board_size - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_pts, dst)
    return cv2.warpPerspective(img, M, (board_size, board_size))


def slice_cells(board_img, pad):
    """Yield (row, col, cell_img). pad shrinks each cell toward its center."""
    size = board_img.shape[0]
    step = size / 8.0
    inset = int(step * pad)
    for r in range(8):
        for c in range(8):
            y1, y2 = int(r * step), int((r + 1) * step)
            x1, x2 = int(c * step), int((c + 1) * step)
            cell = board_img[y1 + inset:y2 - inset, x1 + inset:x2 - inset]
            yield r, c, cell


def extract(game, board_size, pad):
    src_pts = load_corners(game)
    game_dir = os.path.join(GAMEPLAY_DIR, game)
    frames = sorted(f for f in os.listdir(game_dir) if f.lower().endswith(IMG_EXTS))
    if not frames:
        sys.exit(f"❌ No frames in {game_dir}")

    out_root = os.path.join(CELLS_DIR, game)
    os.makedirs(out_root, exist_ok=True)

    print(f"🔪 Slicing {len(frames)} frames for {game} "
          f"(board={board_size}px, pad={pad}) ...")
    for frame in tqdm(frames):
        stem = os.path.splitext(frame)[0]
        img = load_bgr(os.path.join(game_dir, frame))
        board = warp_board(img, src_pts, board_size)
        frame_out = os.path.join(out_root, stem)
        os.makedirs(frame_out, exist_ok=True)
        for r, c, cell in slice_cells(board, pad):
            cv2.imwrite(os.path.join(frame_out, f"r{r}c{c}.jpg"), cell)

    manifest = {
        "game": game,
        "frames": [os.path.splitext(f)[0] for f in frames],
        "frame_files": frames,
        "board_size": board_size,
        "pad": pad,
        "cell_naming": "r{row}c{col}.jpg  (row 0=top, col 0=left of warped board)",
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Done. Cells in {out_root}/<frame>/  | manifest written.")


def main():
    ap = argparse.ArgumentParser(description="Warp + slice game frames into 64 cells.")
    ap.add_argument("game", help="game folder name, e.g. game1")
    ap.add_argument("--board-size", type=int, default=800,
                    help="warped board side length in px (default 800 -> 100px cells)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="fraction of each cell to trim inward (0.0-0.3)")
    args = ap.parse_args()
    extract(args.game, args.board_size, args.pad)


if __name__ == "__main__":
    main()
