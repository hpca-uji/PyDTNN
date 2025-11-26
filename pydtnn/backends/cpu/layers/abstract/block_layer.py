from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
import numpy as np

class AbstractBlockLayerCPU(LayerCPU, AbstractBlockLayer[np.ndarray]):
    pass
