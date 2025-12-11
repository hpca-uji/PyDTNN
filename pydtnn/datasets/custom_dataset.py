import operator
import warnings
import numpy as np
from pydtnn.datasets.dataset import Dataset
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.constants import ArrayShape

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model

TENSOR_ASSERT = {
    TensorFormat.NCHW: operator.lt,
    TensorFormat.NHWC: operator.gt
}


class CustomDataset(Dataset):
    """
    Custom Dataset

    In-memory dataset.
    Train and Test must have matching types, shapes and dtypes.
    Input must be in NCHW format, output in N (or more) format.
    X must be in a NDArray with `model.tensor_shape` shape and `model.dtype` dtype.
    Y must be in a NDArray with N (or more) and `model.dtype` dtype.
    """

    def __init__(self, model: "Model", x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray | None = None, y_test: np.ndarray | None = None,
                 input_shape: ArrayShape | None = None, output_shape: ArrayShape | None = None,
                 force_test_as_validation=False, debug=False):
        if x_test is None or y_test is None:
            if x_test is None and y_test is None:
                x_test = x_train
                y_test = y_train
            else:
                raise ValueError("Both x_test and y_test must be provided or, alternatively, none of them!")

        if input_shape is None:
            _input_shape: ArrayShape = x_train.shape[1:]

        if output_shape is None:
            _output_shape: ArrayShape = y_train.shape[1:]

        if len(x_train.shape) == 3 and not TENSOR_ASSERT[self.model.tensor_format](x_train.shape[0], x_train.shape[2]):
            warnings.warn(f"Dataset x_train.shape {x_train.shape} may not be in {self.model.tensor_format.upper()} format, following the model format!", RuntimeWarning)

        if len(x_test.shape) == 3 and not TENSOR_ASSERT[self.model.tensor_format](x_test.shape[0], x_test.shape[2]):
            warnings.warn(f"Dataset x_test.shape {x_test.shape} may not be in {self.model.tensor_format.upper()} format, following the model format!", RuntimeWarning)

        self.__x_source: list[np.ndarray] = []
        self.__y_source: list[np.ndarray] = []
        # Sources for the training part
        self.__x_source.append(x_train)
        self.__y_source.append(y_train)
        # Sources for the validation part
        if force_test_as_validation:
            self.__x_source.append(x_test)
            self.__y_source.append(y_test)
        else:
            self.__x_source.append(x_train)
            self.__y_source.append(y_train)
        # Sources for the test part
        self.__x_source.append(x_test)
        self.__y_source.append(y_test)

        super().__init__(model,
                         x_train.shape[0],
                         x_test.shape[0],
                         _input_shape,
                         _output_shape,
                         force_test_as_validation=force_test_as_validation,
                         debug=debug)

    def _init_actual_data(self):
        for part in Dataset.Part:
            local_offset = self._local_offset[part]
            local_nsamples = self._local_nsamples[part]
            local_slice = slice(local_offset, local_offset + local_nsamples)
            self._x[part] = self.__x_source[part][local_slice, ...]
            self._y[part] = self.__y_source[part][local_slice, ...]
