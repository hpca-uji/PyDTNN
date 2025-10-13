from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Input
from numpy import ndarray, asarray


class InputCPU(LayerCPU, Input):

    def forward(self, x: ndarray) -> ndarray:
        return asarray(x, dtype=self.model.dtype, order="C", copy=None)

    def backward(self, dy: ndarray) -> ndarray:
        return asarray(dy, dtype=self.model.dtype, order="C", copy=None)
