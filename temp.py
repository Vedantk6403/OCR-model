import os
import cv2

image_folder = "temp"

# Get all image file paths sorted
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")])

# Loop through each image path
for i, img_path in enumerate(image_files):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    print(f"Processing {img_path}, shape: {img.shape}")
    # You can now use img and img_path as needed
