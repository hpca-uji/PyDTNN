from typing import TYPE_CHECKING

import numpy as np

from pydtnn.datasets.custom_dataset import CustomDataset
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.constants import ArrayShape

if TYPE_CHECKING:
    from pydtnn.model import Model


def archive(model: "Model", force_test_as_validation=False, debug=False) -> "CustomDataset":
    """
    Archived Dataset

    Load from a NPZ with x_train, y_train, x_test, y_test attributes.
    Train and Test must have matching types, shapes and dtypes.
    X must be in a NDArray with NCHW shape and float64 dtype.
    Y must be in a NDArray with N (or more) and float64 dtype.
    """

    with np.load(model.dataset_path) as data:
        data: dict[str, np.ndarray]
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
        input_shape: ArrayShape = x_train.shape[1:]

    # Ensure dataset is in model.tensor_format
    x_train = model.encode_tensor(x_train)
    x_test = model.encode_tensor(x_test)

    # Ensure dataset is in model.dtype
    match model.dtype:
        case np.float64:
            pass
        case np.float32:
            x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
            x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)
        case _:
            raise NotImplementedError(f"Unsupported model dtype {model.dtype}")

    # Ensure dataset transformations are applied
    x_train, y_train = x_train.copy(), y_train.copy()
    x_test, y_test = x_test.copy(), y_test.copy()

    # Create dataset
    dataset = CustomDataset(
        model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_shape=input_shape,
        force_test_as_validation=force_test_as_validation,
        debug=debug
    )

    # Debug information
    if dataset.debug:
        print(f"Import: {dataset.model.dataset_raw_path}")
        print(f"x_train: {x_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"y_test: {y_test.shape}")

    return dataset
