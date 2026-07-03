"""Show the YOLO detector's output on a single frame (draws boxes + labels).

Usage:
    python inference/show_detection.py                         # a default frame
    python inference/show_detection.py data/gameplay/game1/IMG_1373.HEIC
    python inference/show_detection.py <image> 0.1             # custom confidence
"""

import os
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.extract_cells import load_bgr  # noqa: E402

WEIGHTS = "models/trained_models/best.pt"


def main():
    from ultralytics import YOLO
    if not os.path.exists(WEIGHTS):
        sys.exit(f"❌ Weights not found: {WEIGHTS}")

    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/gameplay/game1/IMG_1373.HEIC"
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.12

    model = YOLO(WEIGHTS)
    res = model.predict(source=load_bgr(img_path), conf=conf, device="cpu", verbose=False)
    n = 0 if res[0].boxes is None else len(res[0].boxes)

    annotated = res[0].plot()  # BGR image with boxes + class labels
    os.makedirs("results/diag", exist_ok=True)
    out = os.path.join("results/diag",
                       f"detect_{os.path.splitext(os.path.basename(img_path))[0]}.jpg")
    cv2.imwrite(out, annotated)
    print(f"🔎 {n} pieces detected on {os.path.basename(img_path)} (conf={conf})")
    print(f"🖼️  Saved: {out}")
    if sys.platform == "darwin":
        os.system(f'open "{out}"')


if __name__ == "__main__":
    main()
