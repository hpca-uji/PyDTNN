import importlib
import sys

from .dataset import Dataset
from .cifar10 import CIFAR10
from .custom_dataset import CustomDataset
from .imagenet import ImageNet
from .mnist import MNIST
from .folder_loader import DatasetFolderLoader

CustomImport = CustomDataset.import_

def get_dataset(model) -> Dataset:
    try:
        dataset_name = {"mnist": "MNIST", "cifar10": "CIFAR10", "imagenet": "ImageNet", "raw": "CustomImport", "folder": "DatasetFolderLoader"}
        dataset_mod = importlib.import_module("pydtnn.datasets")
        dataset_cls = getattr(dataset_mod, dataset_name[model.dataset_name])
        dataset = dataset_cls(model)
    except Exception:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return dataset
