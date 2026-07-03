"""Convert the HEIC gameplay frames to JPG into per-game folders for annotation.

Output:
    data/annotate/game1_jpg/*.jpg
    data/annotate/game2_jpg/*.jpg
"""

import os

from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()

JOBS = {
    "data/gameplay/game1": "data/annotate/game1_jpg",
    "data/gameplay/game2": "data/annotate/game2_jpg",
}
IMG_EXTS = (".heic", ".jpg", ".jpeg", ".png")


def convert(src, dst):
    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(IMG_EXTS))
    count = 0
    for fname in tqdm(files, desc=os.path.basename(dst)):
        stem = os.path.splitext(fname)[0]
        out = os.path.join(dst, stem + ".jpg")
        try:
            Image.open(os.path.join(src, fname)).convert("RGB").save(
                out, "JPEG", quality=95)
            count += 1
        except Exception as e:
            print(f"  skip {fname}: {e}")
    return count, len(files)


def main():
    for src, dst in JOBS.items():
        done, total = convert(src, dst)
        print(f"✅ {src} -> {dst}: {done}/{total} converted")


if __name__ == "__main__":
    main()
