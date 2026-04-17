import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class OptimizerNumpy(Optimizer[np.ndarray], BaseNumpy):
    """
    Extends an Optimizer class with the attributes and methods required by CPU Optimizers.
    """

    def are_all_zeros(self, ndarray: np.ndarray) -> bool:
        return not ndarray.any()
