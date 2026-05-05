import logging
from abc import abstractmethod

import numpy as np

from pydtnn.abstract.base import Base
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array

__all__ = (
    "Metric",
    "select",
)

logger = logging.getLogger(__name__)


class Metric[T: Array](Base):
    format = ""
    order = 0  # No need of special order.

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _model_init(self) -> None:
        self.dtype: np.dtype = np.dtype(np.float32) if np.issubdtype(self.model.dtype, np.int32) else self.model.dtype
        # NOTE: self.dtype is necessary before calling super.
        super()._model_init()
        self.shape = (self.model.batch_size, *self.model.output_shape)

    @abstractmethod
    def compute(self, y_pred: T, y_targ: T) -> float | np.ndarray:
        pass


def select(name: str) -> type[Metric]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
