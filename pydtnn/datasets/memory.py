"""In-memory dataset implementation for PyDTNN."""

from __future__ import annotations

import logging
import operator
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np

from pydtnn.datasets.abstract import Dataset
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Memory",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model

TENSOR_ASSERT = {TensorFormat.NCHW: operator.lt, TensorFormat.NHWC: operator.gt}


class Memory(Dataset):
    """
    Custom Dataset

    In-memory dataset.
    Train and Test must have matching types, shapes and dtypes.
    Input must be in NCHW format, output in N (or more) format.
    X must be in a NDArray with `model.tensor_shape` shape and `model.dtype` dtype.
    Y must be in a NDArray with N (or more) and `model.dtype` dtype.
    """

    def __init__(
        self,
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        input_shape: ArrayShape | None = None,
        output_shape: ArrayShape | None = None,
        force_test_as_validation: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize the in-memory dataset.

        Args:
            model: The model instance.
            x_train: Training input data.
            y_train: Training target data.
            x_test: Testing input data.
            y_test: Testing target data.
            input_shape: Expected shape of input data.
            output_shape: Expected shape of output data.
            force_test_as_validation: Whether to force test data as validation.
            debug: Enable debug mode.
        """
        if x_test is None or y_test is None:
            if x_test is None and y_test is None:
                x_test = x_train
                y_test = y_train
            else:
                raise ValueError(
                    "Both x_test and y_test must be provided or, alternatively, none of them!"
                )

        if input_shape is None:
            input_shape = x_train.shape[1:]

        if output_shape is None:
            output_shape = y_train.shape[1:]

        if len(x_train.shape) == 3 and not TENSOR_ASSERT[self.model.tensor_format](
            x_train.shape[0], x_train.shape[2]
        ):
            logger.warning(
                f"Dataset x_train.shape {x_train.shape} may not be in {
                    self.model.tensor_format.upper()
                } format, following the model format!"
            )

        if len(x_test.shape) == 3 and not TENSOR_ASSERT[self.model.tensor_format](
            x_test.shape[0], x_test.shape[2]
        ):
            logger.warning(
                f"Dataset x_test.shape {x_test.shape} may not be in {
                    self.model.tensor_format.upper()
                } format, following the model format!"
            )

        test_as_validation = model.test_as_validation or force_test_as_validation

        # Mix train and validation
        idx = np.arange(x_train.shape[0])
        model.random.shuffle(idx)
        x_train = np.ascontiguousarray(x_train[idx])
        y_train = np.ascontiguousarray(y_train[idx])

        self.__x_source: list[np.ndarray] = []
        self.__y_source: list[np.ndarray] = []
        # Sources for the training part
        self.__x_source.append(x_train)
        self.__y_source.append(y_train)
        # Sources for the validation part
        if test_as_validation:
            self.__x_source.append(x_test)
            self.__y_source.append(y_test)
        else:
            self.__x_source.append(x_train)
            self.__y_source.append(y_train)
        # Sources for the test part
        self.__x_source.append(x_test)
        self.__y_source.append(y_test)

        super().__init__(
            model,
            x_train.shape[0],
            x_test.shape[0],
            input_shape,
            output_shape,
            force_test_as_validation=force_test_as_validation,
            debug=debug,
        )

    def _data_generator(self, part: Dataset.Part) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Yield raw data from the dataset partition."""
        x = self.__x_source[part]
        y = self.__y_source[part]

        if part is Dataset.Part.TRAIN and self.model.augment_shuffle:
            idx = np.arange(x.shape[0])
            self.model.random.shuffle(idx[: self.train_nsamples])
            x = x[idx]
            y = y[idx]

        local_offset = self._local_offset[part]
        local_nsamples = self._local_nsamples[part]
        local_slice = slice(local_offset, local_offset + local_nsamples)
        x = x[local_slice, ...]
        y = y[local_slice, ...]
        yield x, y
