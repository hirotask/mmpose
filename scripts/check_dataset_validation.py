import argparse
import glob
import os

#############################
#
# データセットに重複がないかチェックするためのプログラム
#
# 実行方法：python check_dataset_validation.py [学習用データセット] [評価用データセット]
#
#############################

def valid_duplication(train_images, val_images):
    duplicates = []
    for train_image in train_images:
        for val_image in val_images:
            train_base_name = os.path.basename(train_image)
            val_base_name = os.path.basename(val_image)
            if train_base_name == val_base_name:
                duplicates.append(train_base_name)
    
    print(f"重複しているデータ数： {len(duplicates)}")

def get_images(path):
    return glob.glob(f"{path}\\*.jpg")

def parseArgs():
    parser = argparse.ArgumentParser(description="A program that takes in three directories as arguments.")
    parser.add_argument("train_img_dir")
    parser.add_argument("val_img_dir")

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    train_images = get_images(args.train_img_dir)
    val_images = get_images(args.val_img_dir)

    print(f"学習用データセット数: {len(train_images)}")
    print(f"評価用データセット数： {len(val_images)}")

    valid_duplication(train_images, val_images)

if __name__ == "__main__":
    main()
