import operator
import warnings
import numpy as np
from pydtnn.datasets.dataset import Dataset
from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn.utils.types import shape_t

from typing import TYPE_CHECKING, Self
if TYPE_CHECKING:
    from pydtnn.model import Model

TENSOR_ASSERT = {
    PYDTNN_TENSOR_FORMAT.NCHW: operator.lt,
    PYDTNN_TENSOR_FORMAT.NHWC: operator.gt
}


class CustomDataset(Dataset):

    def __init__(self, model: "Model", x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray | None = None, y_test: np.ndarray | None = None,
                 input_shape: shape_t | None = None, output_shape: shape_t | None = None,
                 force_test_as_validation=False, debug=False):
        if x_test is None or y_test is None:
            if x_test is None and y_test is None:
                x_test = x_train
                y_test = y_train
            else:
                raise SystemExit("Both x_test and y_test must be provided or, alternatively, none of them!")

        if input_shape is None:
            input_shape: shape_t = x_train.shape[1:]

        if output_shape is None:
            output_shape: shape_t = y_train.shape[1:]

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
                         input_shape,
                         output_shape,
                         force_test_as_validation=force_test_as_validation,
                         debug=debug)

    def _init_actual_data(self):
        for part in Dataset.Part:
            local_offset = self._local_offset[part]
            local_nsamples = self._local_nsamples[part]
            local_slice = slice(local_offset, local_offset + local_nsamples)
            self._x[part] = self.__x_source[part][local_slice, ...]
            self._y[part] = self.__y_source[part][local_slice, ...]

    @classmethod
    def import_(cls: "type[CustomDataset]", model: "Model", force_test_as_validation=False, debug=False) -> "CustomDataset":
        """Import dataset (rank specific)"""
        with np.load(model.dataset_path) as data:
            data: dict[str, np.ndarray]
            x_train = data["x_train"]
            y_train = data["y_train"]
            x_test = data["x_test"]
            y_test = data["y_test"]
            input_shape: shape_t = x_train.shape[1:]

            # Ensure dataset is in model.tensor_format
            match model.tensor_format:
                case PYDTNN_TENSOR_FORMAT.NCHW:
                    pass
                case PYDTNN_TENSOR_FORMAT.NHWC:
                    x_train = cls._nchw2nhwc(x_train)
                    x_test = cls._nchw2nhwc(x_test)
                case _:
                    raise NotImplementedError(f"Unsupported tensor format {model.tensor_format}")

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
            self = cls(
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
            if self.debug:
                print(f"Import: {self.model.dataset_raw_path}")
                print(f"x_train: {x_train.shape}")
                print(f"y_train: {y_train.shape}")
                print(f"x_test: {x_test.shape}")
                print(f"y_test: {y_test.shape}")

            return self
