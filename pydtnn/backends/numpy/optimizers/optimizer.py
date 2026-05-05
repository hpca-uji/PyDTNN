import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.libs import numpy as np
from pydtnn.optimizers.optimizer import Optimizer

__all__ = ("OptimizerNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class OptimizerNumpy(Optimizer[np.ndarray], BaseNumpy):
    """
    Extends an Optimizer class with the attributes and methods required by CPU Optimizers.
    """

    def are_all_zeros(self, ndarray: np.ndarray) -> bool:
        return not ndarray.any()
