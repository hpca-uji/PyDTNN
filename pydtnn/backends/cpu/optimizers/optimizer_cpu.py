from abc import ABC

from pydtnn.optimizers import Optimizer
from numpy import ndarray

class OptimizerCPU(Optimizer, ABC):
    """
    Extends an Optimizer class with the attributes and methods required by CPU Optimizers.
    """

    def are_all_zeros(self, ndarray: ndarray) -> bool:
        return not ndarray.any()
