"""
From https://github.com/Poulinakis-Konstantinos/ML-util-functions/blob/master/scripts/Img_Premature_Ending-Detect_Fix.py
This script will detect and also fix the problem of Premature Ending in images.
This is caused when the image is corrupted in such a way that their hex code does not end with the default
D9. Opening the image with opencv and other image libraries is usually still possible, but the images might
produce errors during DL training or other tasks.
  Loading such an image with opencv and then saving it again can solve the problem. You can manually inspect
,using a notepad, that the image's hex finishes with D9 after the script has finished.
"""

import os
import cv2

# Directory to search for images
# dir_path = r'/home/...'
# train_path1 = os.path.join(
#     os.environ["HOME"],
#     "scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/300p",
# )
# train_path2 = os.path.join(
#     os.environ["HOME"],
#     "scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/500p",
# )
# train_path3 = os.path.join(
#     os.environ["HOME"],
#     "scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/720p",
# )
# train_path4 = os.path.join(
#     os.environ["HOME"],
#     "scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/fullsize",
# )

# # dir_paths = [train_path1, train_path2, train_path3, train_path4]


def create_paths():
    dir_paths = []
    image_size = ["300p", "500p", "720p", "fullsize"]
    datasets = ["train", "val", "test"]
    for dataset in datasets:
        for size in image_size:
            path = os.path.join(
                os.environ["HOME"],
                f"scratch/fungiclef/dataset/images/FungiTastic-FewShot/{dataset}/{size}",
            )
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
    dir_paths = create_paths()
    # dir_paths = [os.path.join(
    #     os.environ["HOME"],
    #     "scratch/fungiclef/dataset/images/FungiTastic-FewShot/val/images_need_fixing",
    # )]
    for dir_path in dir_paths:
        print(f"We are working with this dataset: {dir_path}")
        for path in os.listdir(dir_path):
            # Make sure to change the extension if it is nor 'jpg' ( for example 'JPG','PNG' etc..)
            if path.endswith(".JPG"):
                img_path = os.path.join(dir_path, path)
                if (
                    img_path
                    == "/storage/home/hcoda1/5/jtam30/scratch/fungiclef/dataset/images/FungiTastic-FewShot/val/720p/3-4100094035.JPG"
                ):
                    pass
                else:
                    detect_and_fix(img_path=img_path, img_name=path)

    print("Process Finished")


if __name__ == "__main__":
    main()
