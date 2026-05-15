"""
Dataset module for PyDTNN.

Provides the base Dataset class and utility functions for managing,
transforming, and generating data batches for machine learning models.
"""

from pydtnn.datasets.dataset.transform import Transform
from pydtnn.utils import find_component


__all__ = (
    "Dataset",
    "select"
)


class Dataset(Transform):
    """
    Base class for handling datasets in PyDTNN.

    This class provides a framework for loading, transforming, and batching data
    for machine learning models. It supports various data augmentation techniques,
    normalization, and distributed data loading.

    NOTE
    - input_shape is expected to be in NCHW format
    - data_generator() is expected to be in model.dtype, normalized to [0, 1]
    - data_generator(x) is expected to be in model.tensor_format format
    - data_generator(y) is expected to be in NC format
    """


def select(name: str) -> type[Dataset]:
    """
    Select a dataset class by name.

    This function dynamically imports and returns a dataset class based on its
    string name. It searches within the current package for the specified class.

    Args:
        name: The string name of the dataset class to select.

    Returns:
        The dataset class type.

    Raises:
        AssertionError: If the package context cannot be determined.
    """
    assert __package__, "Package not found!"
    # NOTE: Going to parent package:
    package = ".".join(__package__.split(".")[:-1])
    return find_component(package, name)