"""Build a game video from a folder of sequential board photos.

Each photo is one settled position (one per move). We hold each photo on
screen for a short time so the result plays like a real game recording that
the video pipeline can later analyze for moves.

Usage:
    python preprocessing/make_video.py --src data/gameplay/game1 --out data/videos/game1.mp4
    python preprocessing/make_video.py --src data/gameplay/game1 --hold 1.0 --fps 30 --width 1280

Supports .heic/.jpg/.jpeg/.png. Frames are sorted by filename (play order).
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()

IMG_EXTS = (".heic", ".jpg", ".jpeg", ".png")


def list_images(src):
    if not os.path.isdir(src):
        sys.exit(f"❌ Source folder not found: {src}")
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(IMG_EXTS))
    if not files:
        sys.exit(f"❌ No images found in {src}")
    return files


def load_resized(path, width):
    rgb = np.array(Image.open(path).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if width and w != width:
        height = int(h * width / w)
        bgr = cv2.resize(bgr, (width, height))
    return bgr


def make_video(src, out, fps, hold, width):
    files = list_images(src)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Use the first image to fix the video frame size.
    first = load_resized(os.path.join(src, files[0]), width)
    H, W = first.shape[:2]
    frames_per_photo = max(1, int(round(fps * hold)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, (W, H))
    if not writer.isOpened():
        sys.exit("❌ Could not open VideoWriter (codec issue).")

    print(f"🎬 {len(files)} photos -> {out}  ({W}x{H}, {fps}fps, {hold}s/photo)")
    for fname in tqdm(files):
        frame = load_resized(os.path.join(src, fname), width)
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        for _ in range(frames_per_photo):
            writer.write(frame)

    writer.release()
    dur = len(files) * hold
    print(f"✅ Wrote {out}  (~{dur:.0f}s, {len(files)} positions)")


def main():
    ap = argparse.ArgumentParser(description="Make a game video from board photos.")
    ap.add_argument("--src", required=True, help="folder of sequential photos")
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=1.0,
                    help="seconds to hold each photo (default 1.0)")
    ap.add_argument("--width", type=int, default=1280,
                    help="output width in px; height keeps aspect (default 1280)")
    args = ap.parse_args()
    make_video(args.src, args.out, args.fps, args.hold, args.width)


if __name__ == "__main__":
    main()
