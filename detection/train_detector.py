"""Train a YOLOv8 chess-piece detector on the Roboflow export.

Dataset: data/roboflow_yolov8/ (YOLOv8 format, 13 classes incl. empty_board).
Runs on Apple Silicon GPU (MPS) if available, else CPU.

Usage:
    python detection/train_detector.py
    python detection/train_detector.py --epochs 150 --model yolov8s.pt

Output weights: runs/detect/chess_detector/weights/best.pt
"""

import argparse
import os

import torch
from ultralytics import YOLO

DATA_YAML = "data/roboflow_yolov8/data.yaml"


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8 chess-piece detector.")
    ap.add_argument("--model", default="yolov8n.pt", help="base weights to fine-tune")
    ap.add_argument("--data", default=DATA_YAML, help="path to dataset data.yaml")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="chess_detector")
    args = ap.parse_args()
    data_yaml = args.data

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Training on device: {device}")

    if not os.path.exists(data_yaml):
        raise SystemExit(f"❌ Dataset yaml not found: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        patience=30,          # early-stop if no improvement
        plots=True,
    )
    print("✅ Training complete. Best weights: "
          f"runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
