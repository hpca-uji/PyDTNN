import numpy as np

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Dropout
from pydtnn.model import Model


class DropoutCPU(LayerCPU, Dropout[np.ndarray]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:

        match self.model.mode:
            case Model.Mode.TRAIN:
                # NOTE: Remember, it's necessary a new random mask every training's forward call.
                # self.mask = np.random.binomial(1, (1 - self.rate), size=self.shape).astype(self.model.dtype) / (1 - self.rate)
                self.mask = np.random.binomial(n=1, p=(1 - self.rate), size=self.shape).astype(dtype=self.model.dtype, order="C", copy=None)
                np.divide(self.mask, (1 - self.rate), out=self.mask, dtype=self.model.dtype)
                np.multiply(x, self.mask, out=x, order="C", dtype=self.model.dtype)
            case Model.Mode.EVALUATE:
                pass # Just returns x.
            case _:
                raise RuntimeError(f"Unexpected model mode \'{self.model.mode}\'.")
        return x
    # ----

    def backward(self, dy: np.ndarray) -> np.ndarray:
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype, order="C")
        return dy
