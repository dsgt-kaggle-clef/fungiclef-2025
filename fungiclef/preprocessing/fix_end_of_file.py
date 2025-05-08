"""
From https://github.com/Poulinakis-Konstantinos/ML-util-functions/blob/master/scripts/Img_Premature_Ending-Detect_Fix.py
This script will detect and also fix the problem of Premature Ending in images.
This is caused when the image is corrupted in such a way that their hex code does not end with the default
D9. Opening the image with opencv and other image libraries is usually still possible, but the images might
produce errors during DL training or other tasks.
  Loading such an image with opencv and then saving it again can solve the problem. You can manually inspect
,using a notepad, that the image's hex finishes with D9 after the script has finished.
"""

import cv2
from pathlib import Path
import argparse
### example usage python fix_image_endings.py --base-dir /scratch/fungiclef/dataset/images/FungiTastic-FewShot


def create_paths(base_dir):
    dir_paths = []
    image_size = ["300p", "500p", "720p", "fullsize"]
    datasets = ["train", "val", "test"]
    base_dir = Path(base_dir)
    for dataset in datasets:
        for size in image_size:
            path = base_dir / dataset / size
            dir_paths.append(path)
    return dir_paths


def detect_and_fix(img_path, img_name):
    # detect for premature ending
    try:
        with open(img_path, "rb") as im:
            im.seek(-2, 2)
            if im.read() == b"\xff\xd9":
                # print('Image OK :', img_name)
                pass
            else:
                # fix image
                img = cv2.imread(img_path)
                print(f"I'm trying to write to path {img_path}")
                cv2.imwrite(img_path, img)
                print("FIXED corrupted image :", img_name)
    except (IOError, SyntaxError) as e:
        print(e)
        print(
            "Unable to load/write Image : {} . Image might be destroyed".format(
                img_path
            )
        )


def main():
    # dir_paths = [os.path.join(
    #     os.environ["HOME"],
    #     "scratch/fungiclef/dataset/images/FungiTastic-FewShot/val/images_need_fixing",
    # )]
    parser = argparse.ArgumentParser(description="Fix premature ending in JPG images")
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="directory containing the dataset (e.g., /scratch/fungiclef/dataset/images/FungiTastic-FewShot)",
    )

    args = parser.parse_args()
    dir_paths = create_paths(args.base_dir)

    # skip images to fix, this file is straight up corrupted. No images in the fullsize directories need to be skipped.
    skip_path = Path(
        "/scratch/fungiclef/dataset/images/FungiTastic-FewShot/val/720p/3-4100094035.JPG"
    )

    for dir_path in dir_paths:
        print(f"We are working with this dataset: {dir_path}")

        for img_file in dir_path.glob("*.JPG"):
            result = extract_path_from_scratch(img_file)
            # print(result)
            result = Path(result)
            # Make sure to change the extension if it is not 'JPG' ( for example 'jpg','PNG' etc..)
            if result == skip_path:
                # print(result)
                pass
            else:
                detect_and_fix(img_path=str(img_file), img_name=img_file.name)

    print("Process Finished")


def extract_path_from_scratch(full_path):
    """
    Extract the portion of the path starting with '/scratch' or 'scratch'
    regardless of what comes before it.

    Args:
        full_path (str): The full file path

    Returns:
        str: The path portion starting with '/scratch' or 'scratch'
    """
    # Convert to string if it's a Path object
    if isinstance(full_path, Path):
        full_path = str(full_path)

    # Method 2: Split the path and look for 'scratch'
    parts = full_path.split("/")
    try:
        # Find the index of 'scratch' in the path parts
        for i, part in enumerate(parts):
            if part == "scratch":
                # Join from 'scratch' to the end with a leading slash
                return "/" + "/".join(parts[i:])
    except ValueError:
        # 'scratch' not found in the path
        pass

    # If all methods fail, return the original
    return full_path


if __name__ == "__main__":
    main()
