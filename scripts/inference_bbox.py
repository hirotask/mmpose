from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv
import glob

config_file = '/mmdet/checkpoints/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = '/mmdet/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cpu'
files = glob.glob("/mmdet/data/mydataset_640x360_fixed/train/*.jpg")

for f in files:
    img = mmcv.imread(f, channel_order='rgb')
    result = inference_detector(model, img)
    print(result)


