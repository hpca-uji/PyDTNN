from pydtnn.datasets.dataset import Dataset
import typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model

from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils import random
from pydtnn.utils.archive import load_archive
import numpy as np

class DatasetFromTar(Dataset):

    def __init__(self, model: "Model", train_nsamples: int, test_nsamples: int, input_shape: tuple[int, ...], output_shape: tuple[int, ...], max_prefetch=1, force_test_as_validation=False, debug=False):
        super().__init__(model, train_nsamples, test_nsamples, input_shape, output_shape, max_prefetch, force_test_as_validation, debug)
        self._xy_filenames: list[list[tuple[tuple[str, ...], np.ndarray]]]

    @typing.override
    def _actual_data_generator(self, part):
        offset = self._local_offset[part]
        nsamples = self._local_nsamples[part]
        xy_filenames = self._xy_filenames[part]

        if part is Dataset.Part.TRAIN:
            random.shuffle(xy_filenames)  # type: ignore (numpy shuffle's typing wasn't well defined.)

        xy_filenames = xy_filenames[offset:offset + nsamples]

        for path, y in xy_filenames:
            with load_archive(*path) as fp:
                x = self._load_image(fp)

            # Add N dimension
            x = x[None, ...]
            y = y[None, ...]

            # Set tensor format
            match self.model.tensor_format:
                case TensorFormat.NHWC:
                    x = self._nchw2nhwc(x)
                case TensorFormat.NCHW:
                    pass
                case _:
                    raise ValueError("Unsupported tensor format")

            # Set dtype and order
            x = x.astype(dtype=self.model.dtype, order="C")
            y = y.astype(dtype=self.model.dtype, order="C")

            # Inplace normalization
            x /= 255.0

            yield x, y
        #---