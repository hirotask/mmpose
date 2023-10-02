from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS

@DATASETS.register_module(name='CustomDataset')
class CustomDataset(BaseDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/custom_dataset.py')
