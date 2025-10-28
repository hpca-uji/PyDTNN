from pydtnn.optimizers.optimizer import Optimizer
import numpy as np


class OptimizerCPU(Optimizer[np.ndarray]):
    """
    Extends an Optimizer class with the attributes and methods required by CPU Optimizers.
    """

    def are_all_zeros(self, ndarray: np.ndarray) -> bool:
        return not ndarray.any()
