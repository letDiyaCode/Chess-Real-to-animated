"""Calibrate the chessboard corners for a game (fixed-camera setup).

Because the camera is fixed for an entire game, the board's four corners sit
at the same pixel location in every frame. So we mark them ONCE on the first
frame and reuse that warp for all frames of the game.

Usage:
    python preprocessing/calibrate_board.py game1
    python preprocessing/calibrate_board.py game1 --frame IMG_1332.HEIC

Workflow:
    1. A window shows the first frame. Click the 4 corners of the PLAYING AREA
       in this order:  top-left -> top-right -> bottom-right -> bottom-left.
       (u = undo last point, r = reset, q/ESC = quit)
    2. A verification window then shows the board warped flat with an 8x8 grid.
       Press 'y' to accept and save, or 'r' to redo the clicks.

Output:
    data/calibration/<game>_corners.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

GAMEPLAY_DIR = "data/gameplay"
CALIB_DIR = "data/calibration"
IMG_EXTS = (".heic", ".jpg", ".jpeg", ".png")
DISPLAY_H = 900            # on-screen height of the calibration image
WARP_SIZE = 800           # verification warp size
CORNER_ORDER = ["top-left", "top-right", "bottom-right", "bottom-left"]


def list_frames(game):
    game_dir = os.path.join(GAMEPLAY_DIR, game)
    if not os.path.isdir(game_dir):
        sys.exit(f"❌ No such game folder: {game_dir}")
    frames = sorted(f for f in os.listdir(game_dir) if f.lower().endswith(IMG_EXTS))
    if not frames:
        sys.exit(f"❌ No image frames found in {game_dir}")
    return game_dir, frames


def load_bgr(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def warp(full, pts, size=WARP_SIZE):
    src = np.array(pts, dtype="float32")
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(full, M, (size, size))


def clicks_too_clustered(full_pts, W, H):
    xs = [p[0] for p in full_pts]
    ys = [p[1] for p in full_pts]
    span_x = (max(xs) - min(xs)) / W
    span_y = (max(ys) - min(ys)) / H
    return span_x < 0.2 or span_y < 0.2


def collect_corners(disp):
    points = []

    def redraw():
        canvas = disp.copy()
        for idx, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(canvas, str(idx + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if len(points) > 1:
            cv2.polylines(canvas, [np.array(points, np.int32)],
                          len(points) == 4, (0, 200, 0), 2)
        hint = (f"Click: {CORNER_ORDER[len(points)]}"
                if len(points) < 4 else "ENTER=continue  r=reset  u=undo  q=quit")
        cv2.putText(canvas, hint, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("calibrate", canvas)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            redraw()

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            return None
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key == ord("r"):
            points.clear()
            redraw()
        if key in (13, 10) and len(points) == 4:
            return points


def verify(warped):
    grid = warped.copy()
    step = warped.shape[0] // 8
    for k in range(9):
        cv2.line(grid, (k * step, 0), (k * step, warped.shape[0]), (0, 255, 0), 1)
        cv2.line(grid, (0, k * step), (warped.shape[1], k * step), (0, 255, 0), 1)
    cv2.putText(grid, "Grid aligns with squares? y=accept  r=redo",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.namedWindow("verify", cv2.WINDOW_NORMAL)
    cv2.imshow("verify", grid)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("y"):
            cv2.destroyWindow("verify")
            return True
        if key in (ord("r"), ord("q"), 27):
            cv2.destroyWindow("verify")
            return False


def calibrate(game, frame=None):
    game_dir, frames = list_frames(game)
    frame = frame or frames[0]
    frame_path = os.path.join(game_dir, frame)
    print(f"📷 Calibrating on frame: {frame_path}")

    full = load_bgr(frame_path)
    H, W = full.shape[:2]
    scale = DISPLAY_H / H
    disp = cv2.resize(full, (int(W * scale), DISPLAY_H))

    while True:
        pts = collect_corners(disp)
        if pts is None:
            cv2.destroyAllWindows()
            print("🚪 Quit without saving.")
            return None

        full_pts = [[round(x / scale, 2), round(y / scale, 2)] for (x, y) in pts]

        if clicks_too_clustered(full_pts, W, H):
            print("⚠️  Those 4 points are too close together to be board corners. "
                  "Click the FOUR far-apart corners of the playing area.")
            continue

        warped = warp(full, full_pts)
        if verify(warped):
            break
        print("🔁 Redoing calibration...")

    cv2.destroyAllWindows()

    os.makedirs(CALIB_DIR, exist_ok=True)
    out_path = os.path.join(CALIB_DIR, f"{game}_corners.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "game": game,
                "reference_frame": frame,
                "image_size": [W, H],
                "corner_order": CORNER_ORDER,
                "corners": full_pts,
            },
            f,
            indent=2,
        )
    print(f"✅ Saved corners to {out_path}")
    print(f"   corners (full-res px): {full_pts}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Calibrate board corners for a game.")
    ap.add_argument("game", help="game folder name under data/gameplay/, e.g. game1")
    ap.add_argument("--frame", help="specific frame filename to calibrate on")
    args = ap.parse_args()
    calibrate(args.game, args.frame)


if __name__ == "__main__":
    main()
