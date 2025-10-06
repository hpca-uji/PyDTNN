import numpy as np

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Dropout
from pydtnn.model import ModelModeEnum


class DropoutCPU(LayerCPU, Dropout):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: np.ndarray = None

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

    def forward(self, x: np.ndarray) -> np.ndarray:

        match self.model.mode:
            case ModelModeEnum.TRAIN:
                # NOTE: Remember, it's necessary a new random mask every training's forward call.
                # self.mask = np.random.binomial(1, (1 - self.rate), size=self.shape).astype(self.model.dtype) / (1 - self.rate)
                self.mask = np.random.binomial(n=1, p=(1 - self.rate), size=self.shape).astype(dtype=self.model.dtype)
                self.mask /= (1 - self.rate)
                return x * self.mask
            case ModelModeEnum.EVALUATE:
                return x
            case _:
                raise RuntimeError(f"Unexpected model mode \'{self.model.mode}\'.")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dy *= self.mask
        return dy
