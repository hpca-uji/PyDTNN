import numpy as np

from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.scalar import Scalar


class ScalarCPU(Scalar[np.ndarray], LayerCPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        # Performance model
        self.fwd_time = None  # Not yet
        self.bwd_time = self.fwd_time

    def forward(self, x):
        return x * self.scale

    def backward(self, dy):
        return dy * self.scale
