from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.libs import libnumpy as np


class AbstractBlockLayerCPU(AbstractBlockLayer[np.ndarray], LayerCPU):
    pass
