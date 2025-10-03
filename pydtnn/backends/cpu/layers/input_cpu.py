from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Input
from numpy import ndarray


class InputCPU(LayerCPU, Input):

    def forward(self, x: ndarray) -> ndarray:
        # NOTE: This layer is never called.
        return x

    def backward(self, dy: ndarray) -> ndarray:
        # NOTE: This layer is never called.
        return dy
