import numpy as np

from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.dropout import Dropout
from pydtnn.model import Model
from pydtnn.utils import random


class DropoutCPU(Dropout[np.ndarray], LayerCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: np.ndarray = None  # type: ignore (It will be initalized later.)

    def forward(self, x: np.ndarray) -> np.ndarray:

        match self.model.mode:
            case Model.Mode.TRAIN:
                # NOTE: Remember, it's necessary a new random mask every training's forward call.
                # self.mask = random.binomial(1, (1 - self.rate), size=self.shape).astype(self.model.dtype) / (1 - self.rate)
                self.mask = np.asarray(random.binomial(n=1, p=(1 - self.rate), size=self.shape), dtype=self.model.dtype, order="C", copy=None)
                np.divide(self.mask, (1 - self.rate), out=self.mask, dtype=self.model.dtype)
                np.multiply(x, self.mask, out=x, order="C", dtype=self.model.dtype)
            case Model.Mode.EVALUATE:
                pass  # Just returns x.
            case _:
                raise RuntimeError(f"Unexpected model mode \'{self.model.mode}\'.")
        return x
    # ----

    def backward(self, dy: np.ndarray) -> np.ndarray:
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype, order="C")
        return dy
