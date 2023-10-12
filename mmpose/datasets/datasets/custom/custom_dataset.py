from ..base import BaseCocoStyleDataset
from mmpose.registry import DATASETS

@DATASETS.register_module(name='CustomDataset')
class CustomDataset(BaseCocoStyleDataset):
    METAINFO: dict = dict(from_file='configs/_base_/datasets/custom_dataset.py')
