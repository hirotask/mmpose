import argparse
import glob
from PIL import Image
import os

def resize_images(dir, out_dir):
    jpg_files = glob.glob(f"{dir}\\*.jpg")

    for f in jpg_files:
        filename = os.path.basename(f)
        image = Image.open(f)
        resized_image = image.resize((640, 360))
        resized_file_path = f"{out_dir}\\{filename}"
        resized_image.save(resized_file_path)


def parseArgs():
    parser = argparse.ArgumentParser(description="A program that takes in three directories as arguments.")
    parser.add_argument("img_dir", help="The directory containing the images.")
    parser.add_argument("img_out_dir")
    
    args = parser.parse_args()

    return args

def main():
    args = parseArgs()

    resize_images(args.img_dir, args.img_out_dir)

if __name__ == "__main__":
    main()
