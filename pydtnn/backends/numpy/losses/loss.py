import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.libs import numpy as np
from pydtnn.losses.loss import Loss

__all__ = ("LossNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class LossNumpy(Loss[np.ndarray], BaseNumpy):
    """
    Extends a Loss class with the attributes and methods required by CPU Losses.
    """

    def _model_init(self) -> None:
        super()._model_init()
        self.dx = np.ndarray(self.shape, dtype=self.model.dtype)
        self.memory_used += self.dx.nbytes
