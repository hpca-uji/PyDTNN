from pydtnn.losses.loss import Loss
from pydtnn.libs import libnumpy as np


class LossCPU(Loss[np.ndarray]):
    """
    Extends a Loss class with the attributes and methods required by CPU Losses.
    """

    def initialize(self) -> None:
        super().initialize()
        self.dx = np.ndarray(self.shape, dtype=self.model.dtype)
        self.real_memory_size += self.dx.nbytes
