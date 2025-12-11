import numpy as np

from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.multiplication import Multiplication
from pydtnn.model import Model


class MultiplicationCPU(Multiplication[np.ndarray], LayerCPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x1 = None
        self.x2 = None

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        # Performance model
        self.fwd_time = None
        self.bwd_time = None

    def transpose(self, x):
        return x.swapaxes(-2,-1)

    def forward(self, x1, x2):
        if self.model.mode == Model.Mode.TRAIN:
            self.x1 = x1
            self.x2 = x2
        return self.model.matmul(x1, x2)

    def backward(self, dy):
        dx1 = self.model.matmul(dy, self.transpose(self.x2))
        dx2 = self.model.matmul(self.transpose(self.x1), dy)
        return dx1, dx2
