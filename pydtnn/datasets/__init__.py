import importlib
import sys

from pydtnn.datasets.dataset import Dataset
from pydtnn.datasets.cifar10 import CIFAR10
from pydtnn.datasets.custom_dataset import CustomDataset
from pydtnn.datasets.imagenet import ImageNet
from pydtnn.datasets.mnist import MNIST
from pydtnn.datasets.folder_loader import DatasetFolderLoader
from pydtnn.datasets.chest_xray14 import ChestXRay14

CustomImport = CustomDataset.import_


# TODO: REMOVE imports and use proper import_module path
def get_dataset(model) -> Dataset:
    try:
        dataset_name = {"mnist": "MNIST",
                        "cifar10": "CIFAR10",
                        "imagenet": "ImageNet",
                        "archive": "CustomImport",
                        "folder": "DatasetFolderLoader",
                        "chestxray14": "ChestXRay14"
                        }
        dataset_mod = importlib.import_module("pydtnn.datasets")
        dataset_cls = getattr(dataset_mod, dataset_name[model.dataset_name])
        dataset = dataset_cls(model)
    except Exception:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return dataset
