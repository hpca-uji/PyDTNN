import logging

import numpy as np

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array

__all__ = ("Input",)

logger = logging.getLogger(__name__)


class Input[T: Array](Layer[T]):
    def __init__(self, shape: tuple = (1,)):
        super().__init__(shape)

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[T, T]:
        return (x_batch, y_batch)  # type: ignore (It's fine)
