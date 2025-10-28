import importlib
import sys

from pydtnn.datasets.dataset import Dataset
from pydtnn.datasets.cifar10 import CIFAR10
from pydtnn.datasets.custom_dataset import CustomDataset
from pydtnn.datasets.imagenet import ImageNet
from pydtnn.datasets.mnist import MNIST
from pydtnn.datasets.folder import Folder
from pydtnn.datasets.chestxray import ChestXRay
from pydtnn.datasets.synthetic import Synthetic

CustomImport = CustomDataset.import_


__all__ = (
    "get_dataset",
)


# TODO: REMOVE imports and use proper import_module path
def get_dataset(model) -> Dataset:
    try:
        dataset_name = {"mnist": "MNIST",
                        "cifar10": "CIFAR10",
                        "imagenet": "ImageNet",
                        "archive": "CustomImport",
                        "folder": "DatasetFolderLoader",
                        "chestxray": "ChestXRay",
                        "synthetic": "Synthetic"
                        }
        dataset_mod = importlib.import_module("pydtnn.datasets")
        dataset_cls = getattr(dataset_mod, dataset_name[model.dataset_name])
        dataset = dataset_cls(model)
    except Exception:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return dataset
