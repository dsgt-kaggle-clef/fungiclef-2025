# import cv2
# import os

# def find_corrupt_images(image_dir):
#     corrupt_images = []
#     for filename in os.listdir(image_dir):
#         img_path = os.path.join(image_dir, filename)
#         img = cv2.imread(img_path)
#         if img is None:
#             corrupt_images.append(img_path)
#         print(corrupt_images)

#     return corrupt_images

# image_dir = os.path.join(os.environ["HOME"], "scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/fullsize")
# corrupt_images = find_corrupt_images(image_dir)

# if corrupt_images:
#     print("Found corrupt images:", corrupt_images)
#     for img in corrupt_images:
#         os.remove(img)  # Remove corrupt files if needed
# else:
#     print("No corrupt images found.")

import os
import cv2
from PIL import Image


def find_corrupt_images(image_dir):
    corrupt_images = []

    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)

        # Try loading with OpenCV
        # print(f"image being read is {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"OpenCV failed to load: {img_path}")
            corrupt_images.append(img_path)
            continue  # Skip further checks

        # Try loading with PIL (better for detecting JPEG issues)
        try:
            with Image.open(img_path) as img:
                img.verify()  # Detect truncated files
        except (IOError, OSError) as e:
            print(f"PIL failed to load: {img_path} -> {e}")
            corrupt_images.append(img_path)

    return corrupt_images


image_dir = os.path.join(
    os.environ["HOME"],
    "scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/720p",
)
corrupt_images = find_corrupt_images(image_dir)

if corrupt_images:
    print("Found corrupt images:", corrupt_images)

    # Optional: Remove corrupt files
    for img in corrupt_images:
        os.remove(img)
    print("Corrupt images removed.")
else:
    print("No corrupt images found.")
