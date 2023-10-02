from PIL import Image
import os
import glob
from datetime import datetime
import argparse
import collections as cl
import json
import re

class COCODatasets():
    def __init__(self, img_dir, annotation_dir, pose_2d_dir, bbox_dir) -> None:
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.pose_2d_dir = pose_2d_dir
        self.bbox_dir = bbox_dir
        self.base_image_id = 2000


    def get_image_files(self):
        # Get a list of all files and directories in the specified directory
        jpg_files = glob.glob(f"{self.img_dir}\\*.jpg")

        print(f"read image files: {jpg_files[:10]}...")

        return jpg_files
    
    def get_bachi_annotations(self):
        # Unixでこのスクリプトを動かす場合、\\を/に変えること
        ann_files = glob.glob(f"{self.annotation_dir}\\*json")

        print(f"read annotation files: {ann_files[:10]}...")

        results = []
        for file_path in ann_files:
            with open(file_path,"r") as f:
                filename = os.path.basename(file_path)
                id = re.findall(r'\d+', filename)
                tmp = cl.OrderedDict()
                tmp["id"] = int(id[0])
                tmp["data"] = json.load(f)
                results.append(tmp)

        return results

    def get_2d_pose_annotations(self):
        ann_files = glob.glob(f"{self.pose_2d_dir}\\**\\*json")

        print(f"read 2D pose annotation files: {ann_files[:10]}...")

        results = []
        for file_path in ann_files:
            with open(file_path,"r") as f:
                filename = os.path.basename(file_path)
                id = re.findall(r'\d+', filename)
                tmp = cl.OrderedDict()
                tmp["id"] = int(id[0])
                tmp["data"] = json.load(f)
                results.append(tmp)

        return results


    def get_image_info(self, image_path):
        # Open the image file
        with Image.open(image_path) as img:
            width, height = img.size

        # Get the filename
        filename = os.path.basename(image_path)

        # Get the last modified time
        timestamp = os.path.getmtime(image_path)
        last_modified_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        numbers = re.findall(r'\d+', filename)

        return {
            'filename': filename,
            'width': width,
            'height': height,
            'last_modified_time': last_modified_time,
            'image_id': self.base_image_id + int(numbers[0]) if numbers else None
        }
    
    def get_bboxes(self):
        # predフォルダを指定していることが前提
        json_files = glob.glob(f"{self.bbox_dir}\\*json")

        print(f"read bboxes annotation files: {json_files[:10]}...")

        results = []
        for file_path in json_files:
            with open(file_path,"r") as f:
                filename = os.path.basename(file_path)
                id = re.findall(r'\d+', filename)
                tmp = cl.OrderedDict()
                tmp["id"] = int(id[0])
                data = json.load(f)

                person_bboxes = []
                if data["labels"] and data["bboxes"]:
                    for i,label in enumerate(data["labels"]):
                        if int(label) == 1: # label==1はPersonの検出結果
                            person_bboxes.append({"score": data["scores"][i], "bbox": data["bboxes"][i]})
    
                if len(person_bboxes) > 0:
                    max_score_data  = max(person_bboxes, key=lambda x: x["score"]) #スコアが最も高いデータを取得
                    tmp["bbox"] = max_score_data["bbox"]
                else:
                    tmp["bbox"] = []
                
                results.append(tmp)

        return results

    ## COCO Datasetsを作る関数たち
    ## 意味はここを参照:https://qiita.com/kHz/items/8c06d0cb620f268f4b3e#%E3%81%AF%E3%81%98%E3%82%81%E3%81%AB
    def make_info(self):
        tmp = cl.OrderedDict()
        tmp["description"] = "Test"
        tmp["url"] = "https://test"
        tmp["version"] = "0.01"
        tmp["year"] = 2023
        tmp["contributor"] = "hashi"
        tmp["data_created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return tmp
    
    def make_licenses(self):
        tmp = cl.OrderedDict()
        tmp["id"] = 1
        tmp["url"] = ""
        tmp["name"] = ""
        return tmp

    def make_images(self):
        image_files = self.get_image_files()
        tmps = []
        for image_file in image_files:
            image_info = self.get_image_info(image_path=image_file)
            tmp = cl.OrderedDict()
            tmp["license"] = 1
            tmp["id"] = image_info["image_id"]
            tmp["file_name"] = image_info["filename"]
            tmp["width"] = image_info["width"]
            tmp["height"] = image_info["height"]
            tmp["date_captured"] = image_info["last_modified_time"]
            tmps.append(tmp)
        
        return tmps

    def make_annotations(self):
        # TODO: バチデータがないときはnum_keypointsが16になるようなデータを作成
        image_files = self.get_image_files()
        annotations = self.get_bachi_annotations()
        poses = self.get_2d_pose_annotations()
        bboxes = self.get_bboxes()

        tmps = []

        for image_file in image_files:
            image_info = self.get_image_info(image_path=image_file)

            keypoints = []
            for pose in poses:
                if pose["id"] == (image_info["image_id"] - self.base_image_id):
                    keypoints = pose["data"]["instance_info"][0]["keypoints"]
            
            for annotation in annotations:
                if annotation["id"] == (image_info["image_id"] - self.base_image_id):
                    # left_wrist = 9
                    # right_wrist = 10
                    for ann in annotation["data"]["shapes"]:
                        if ann["label"] == "left_wrist":
                            keypoints[9] = ann["points"][0]
                        elif ann["label"] == "right_wrist":
                            keypoints[10] = ann["points"][0]
                
            for keypoint in keypoints:
                keypoint.append(2)

            bbox_res = []
            for bbox in bboxes:
                if bbox["id"] == (image_info["image_id"] - self.base_image_id):
                    bbox_res = bbox["bbox"]

            if len(keypoints) == 0 or len(bbox_res) == 0:
                continue

            tmp = cl.OrderedDict()
            tmp["segmentation"] = []
            tmp["id"] = 1000 +  image_info["image_id"] #アノテーションID
            tmp["num_keypoints"] = 17 # キーポイント数
            tmp["image_id"] = image_info["image_id"] # イメージID
            tmp["category_id"] = 1 # カテゴリーID
            tmp["area"] = image_info["width"] * image_info["height"] #ピクセル数
            tmp["iscrowd"] = 0
            tmp["bbox"] =  bbox_res
            tmp["keypoints"] = keypoints
            tmps.append(tmp)

        return tmps
    
    def make_categories(self):
        tmps = []
        tmp = cl.OrderedDict()
        tmp["id"] = "1"
        tmp["supercategory"] = "person"
        tmp["name"] = "person"
        tmp["keypoints"] = [
            "nose","left_eye","right_eye","left_ear","right_ear",
			"left_shoulder","right_shoulder",
			"left_elbow","right_elbow","left_wrist","right_wrist",
			"left_hip","right_hip",
			"left_knee","right_knee","left_ankle","right_ankle"
        ]
        tmp["skeleton"] = [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
			[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
        ]
        tmps.append(tmp)
            
        return tmps

def parseArgs():
    parser = argparse.ArgumentParser(description="A program that takes in three directories as arguments.")
    parser.add_argument("img_dir", help="The directory containing the images.")
    parser.add_argument("annotation_dir", help="The directory containing the annotations.")
    parser.add_argument("pose_2d_dir", help="The directory containing the 2D pose information.")
    parser.add_argument("bbox_dir",help="The directory containing the bbox." )

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()

    js = cl.OrderedDict()
    coco = COCODatasets(img_dir=args.img_dir, annotation_dir=args.annotation_dir, pose_2d_dir=args.pose_2d_dir, bbox_dir=args.bbox_dir)
    js["info"] = coco.make_info()
    js["licenses"] = coco.make_licenses()
    js["images"] = coco.make_images()
    js["annotations"] = coco.make_annotations()
    js["categories"] = coco.make_categories()

    # write
    fw = open('dataset.json','w')
    json.dump(js,fw,indent=2)


if __name__ == "__main__":
    main()
