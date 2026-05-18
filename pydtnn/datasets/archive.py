"""
Module for handling archived datasets stored in NPZ format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from pydtnn.datasets.abstract import Dataset
from pydtnn.utils import get_npz_shape

__all__ = ("Archive",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class Archive(Dataset):
    """
    Archived Dataset

    Load from a NPZ with x_train, y_train, x_test, y_test attributes.
    Train and Test must have matching types, shapes and dtypes.
    X must be in a NDArray with NCHW shape and float64 dtype.
    Y must be in a NDArray with N (or more) and float64 dtype.
    """

    def __init__(self, model: Model, force_test_as_validation=False, debug=False):
        """
        Initialize the Archive dataset by inspecting the NPZ file structure.

        Args:
            model: The model instance associated with this dataset.
            force_test_as_validation: Whether to use test data as validation data.
            debug: Enable debug logging.
        """
        shapes = get_npz_shape(model.dataset_path)
        x_train = shapes["x_train"]
        y_train = shapes["y_train"]
        x_test = shapes["x_test"]
        y_test = shapes["y_test"]

        debug_str = list[str]()
        debug_str.append(f"Import: {model.dataset_path}")
        debug_str.append(f"x_train: {x_train}")
        debug_str.append(f"y_train: {y_train}")
        debug_str.append(f"x_test: {x_test}")
        debug_str.append(f"y_test: {y_test}")
        logger.debug("\n".join(debug_str))

        super().__init__(model, x_train[0], x_test[0], x_train[1:], y_train[1:], force_test_as_validation=force_test_as_validation, debug=debug)

    def _ensure_data_init(self):
        """
        Lazy load and process dataset from disk if not already initialized.
        """
        if len(self._x[Dataset.Part.TRAIN]):
            return

        with np.load(self.model.dataset_path) as data:
            x_train = data["x_train"]
            y_train = data["y_train"]
            x_test = data["x_test"]
            y_test = data["y_test"]

        # Ensure dataset is in model.tensor_format
        x_train = self.model.encode_tensor(x_train)
        x_test = self.model.encode_tensor(x_test)

        # Ensure dataset is in model.dtype
        match self.model.dtype:
            case np.float64:
                pass
            case np.float32:
                x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
                x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)
            case _:
                raise NotImplementedError(f"Unsupported model dtype {self.model.dtype}")

        # Ensure dataset transformations are applied
        x_train, y_train = np.ascontiguousarray(x_train), np.ascontiguousarray(y_train)
        x_test, y_test = np.ascontiguousarray(x_test), np.ascontiguousarray(y_test)

        self._x[Dataset.Part.TRAIN] = x_train
        self._y[Dataset.Part.TRAIN] = y_train

        self._x[Dataset.Part.TEST] = x_test
        self._y[Dataset.Part.TEST] = y_test

        self._x[Dataset.Part.VAL] = x_test if self.test_as_validation else x_train
        self._y[Dataset.Part.VAL] = y_test if self.test_as_validation else y_train

    def _actual_data_generator(self, part: Dataset.Part):
        """
        Generator that yields data batches for a specific dataset part.

        Args:
            part: The dataset partition to generate data from.
        """
        self._ensure_data_init()
        yield from super()._actual_data_generator(part)
