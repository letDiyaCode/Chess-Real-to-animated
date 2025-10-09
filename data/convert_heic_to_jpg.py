import os
from pillow_heif import register_heif_opener
from PIL import Image
from tqdm import tqdm

# Enable HEIC support for PIL
register_heif_opener()

# Input and output directories
input_dir = "data/single_pieces"
output_dir = "data/single_pieces_converted"

os.makedirs(output_dir, exist_ok=True)

count = 0
for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(input_dir, filename)
        jpg_name = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(output_dir, jpg_name)
        try:
            img = Image.open(heic_path)
            img = img.convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)
            count += 1
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print(f"\n✅ Conversion complete! {count} HEIC images converted to JPG.")
print(f"Converted images saved in: {output_dir}")
