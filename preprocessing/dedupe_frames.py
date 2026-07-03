"""Remove duplicate (no-change) frames from a game and build a clean video.

Many photos in a game are near-identical (no move happened between them) and
the camera drifts slightly. This tool:
  1. Aligns every frame to the reference frame (ORB homography) so drift is
     not mistaken for a move.
  2. Warps to the flat board and compares each frame to the last KEPT frame.
  3. Keeps a frame only if some square changed more than --threshold
     (i.e. a move actually happened).
  4. Writes a deduplicated, stabilized video of the kept positions and saves
     the list of kept frames.

Usage:
    python preprocessing/dedupe_frames.py game1
    python preprocessing/dedupe_frames.py game1 --threshold 22 --hold 1.0

Outputs:
    data/labels/<game>_keyframes.json   (kept frame filenames, in order)
    data/videos/<game>_clean.mp4        (stabilized, deduplicated)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.extract_cells import load_bgr, load_corners, warp_board  # noqa: E402

GAMEPLAY_DIR = "data/gameplay"
LABELS_DIR = "data/labels"
VIDEOS_DIR = "data/videos"
IMG_EXTS = (".heic", ".jpg", ".jpeg", ".png")
BOARD_SIZE = 800


class Aligner:
    """Aligns frames to a reference image using ORB feature matching."""

    def __init__(self, ref_bgr, n_features=4000):
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.kp0, self.des0 = self.orb.detectAndCompute(self.ref_gray, None)
        self.size = (ref_bgr.shape[1], ref_bgr.shape[0])

    def homography(self, bgr):
        """Return the homography mapping this frame's coords -> reference coords
        (or None if it can't be estimated)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 12:
            return None
        matches = sorted(self.bf.match(des, self.des0), key=lambda m: m.distance)[:400]
        if len(matches) < 12:
            return None
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([self.kp0[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H

    def align(self, bgr):
        H = self.homography(bgr)
        if H is None:
            return bgr
        return cv2.warpPerspective(bgr, H, self.size)


def max_square_change(prev_gray, cur_gray, inset_frac=0.2):
    """Largest per-square mean-abs-difference (skips grid-line borders)."""
    step = BOARD_SIZE / 8.0
    inset = int(step * inset_frac)
    d = cv2.absdiff(prev_gray, cur_gray)
    best = 0.0
    for r in range(8):
        for c in range(8):
            y1, y2 = int(r * step) + inset, int((r + 1) * step) - inset
            x1, x2 = int(c * step) + inset, int((c + 1) * step) - inset
            best = max(best, float(d[y1:y2, x1:x2].mean()))
    return best


def list_frames(game):
    game_dir = os.path.join(GAMEPLAY_DIR, game)
    if not os.path.isdir(game_dir):
        sys.exit(f"❌ No such game folder: {game_dir}")
    frames = sorted(f for f in os.listdir(game_dir) if f.lower().endswith(IMG_EXTS))
    if not frames:
        sys.exit(f"❌ No frames in {game_dir}")
    return game_dir, frames


def dedupe(game, threshold, hold, fps, width, stabilize):
    corners = load_corners(game)
    game_dir, frames = list_frames(game)

    ref_bgr = load_bgr(os.path.join(game_dir, frames[0]))
    aligner = Aligner(ref_bgr)

    def warped_of(bgr):
        aligned = aligner.align(bgr) if stabilize else bgr
        return aligned, warp_board(aligned, corners, BOARD_SIZE)

    kept = [frames[0]]
    kept_full = []
    a0, w0 = warped_of(ref_bgr)
    kept_full.append(a0)
    last_gray = cv2.cvtColor(w0, cv2.COLOR_BGR2GRAY)

    print(f"🔍 Scanning {len(frames)} frames (threshold={threshold}) ...")
    for fname in tqdm(frames[1:]):
        bgr = load_bgr(os.path.join(game_dir, fname))
        aligned, warped = warped_of(bgr)
        cur_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        change = max_square_change(last_gray, cur_gray)
        if change > threshold:
            kept.append(fname)
            kept_full.append(aligned)
            last_gray = cur_gray

    print(f"✅ Kept {len(kept)} distinct positions out of {len(frames)} frames.")

    # Save keyframe list
    os.makedirs(LABELS_DIR, exist_ok=True)
    kf_path = os.path.join(LABELS_DIR, f"{game}_keyframes.json")
    with open(kf_path, "w") as f:
        json.dump({"game": game, "threshold": threshold,
                   "kept": kept, "total": len(frames)}, f, indent=2)
    print(f"💾 Keyframes: {kf_path}")

    # Build the clean (stabilized) video from kept frames
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    out = os.path.join(VIDEOS_DIR, f"{game}_clean.mp4")
    H0, W0 = kept_full[0].shape[:2]
    out_w = width
    out_h = int(H0 * width / W0)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    per = max(1, int(round(fps * hold)))
    for img in kept_full:
        frame = cv2.resize(img, (out_w, out_h))
        for _ in range(per):
            writer.write(frame)
    writer.release()
    print(f"🎬 Clean video: {out}  ({len(kept)} positions, ~{len(kept)*hold:.0f}s)")
    return kf_path, out


def main():
    ap = argparse.ArgumentParser(description="Dedupe game frames and build clean video.")
    ap.add_argument("game")
    ap.add_argument("--threshold", type=float, default=22.0,
                    help="min per-square change to count as a new position")
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--no-stabilize", action="store_true",
                    help="skip ORB alignment (faster, but keeps camera drift)")
    args = ap.parse_args()
    dedupe(args.game, args.threshold, args.hold, args.fps, args.width,
           stabilize=not args.no_stabilize)


if __name__ == "__main__":
    main()
