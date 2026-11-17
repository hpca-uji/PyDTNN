"""CuPy-based implementation of layerable components for the PyDTNN framework."""
import math
from typing import TYPE_CHECKING

import cupy.cuda as cupy_driver

from pydtnn.libs import numpy as np
from pydtnn.backends.cupy.abstract.base import BaseCupy
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.utils.constants import ArrayShape

__all__ = ("LayerableCupy",)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class LayerableCupy(LayerableNumpy, BaseCupy):
    """Abstract base class for layers utilizing CuPy as the backend."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None) -> None:
        """Init method of LayerableCupy."""
        super()._model_init(prev_shape, x)

        if self.id != self.model.backend.id:
            return None

        # Calculate the kernel sizes
        attrs = cupy_driver.runtime.getDeviceProperties(self.model.cupy_device)

        max_threads = attrs["maxThreadsDim"][0]
        min_threads = cupy_driver.runtime.deviceGetAttribute(cupy_driver.runtime.cudaDevAttrWarpSize, self.model.cupy_device)

        block_size = max(min_threads, min(self.model.batch_size, max_threads))
        grid_size = math.ceil(self.model.batch_size / block_size)

        self.cuda_grid = (grid_size, 1, 1)
        self.cuda_block = (block_size, 1, 1)
