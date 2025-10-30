from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydtnn.model import Model as _Model
    from pydtnn.datasets.dataset import Dataset as _Dataset


# TODO: REMOVE imports and use proper import_module path
def select(model: "_Model") -> "_Dataset":
    from pydtnn.datasets.cifar10 import CIFAR10
    from pydtnn.datasets.custom_dataset import CustomDataset
    from pydtnn.datasets.imagenet import ImageNet
    from pydtnn.datasets.mnist import MNIST
    from pydtnn.datasets.folder import Folder
    from pydtnn.datasets.chestxray import ChestXRay
    from pydtnn.datasets.synthetic import Synthetic

    dataset = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "imagenet": ImageNet,
        "archive": CustomDataset.import_,
        "folder": Folder,
        "chestxray": ChestXRay,
        "synthetic": Synthetic
    }

    try:
        cls = dataset[model.dataset_name]
    except KeyError:
        raise ValueError(f"Dataset {model.dataset_name!r} not found!") from None

    return cls(model)
