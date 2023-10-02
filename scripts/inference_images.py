from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.apis import MMPoseInferencer
import glob
import os

register_all_modules()

config_file = 'rtmpose-m_8xb256-420e_coco-256x192.py'
checkpoint_file = 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'

inferencer = MMPoseInferencer(
    pose2d=config_file,
    pose2d_weights=checkpoint_file
)

image_paths = glob.glob("/mmpose/data/mydataset_640x360_fixed/train/*.jpg")

for image_path in image_paths:
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    result_generator = inferencer(image_path, out_dir=f"vis_result_{file_name}")
    result = next(result_generator)
