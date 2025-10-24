from pydtnn.optimizers.optimizer import Optimizer
from numpy import ndarray


class OptimizerCPU(Optimizer):
    """
    Extends an Optimizer class with the attributes and methods required by CPU Optimizers.
    """

    def are_all_zeros(self, ndarray: ndarray) -> bool:
        return not ndarray.any()
