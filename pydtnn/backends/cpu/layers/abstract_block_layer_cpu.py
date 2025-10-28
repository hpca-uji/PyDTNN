from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.layers.abstract_block_layer import AbstractBlockLayer
import numpy as np

class AbstractBlockLayerCPU(LayerCPU, AbstractBlockLayer[np.ndarray]):
    pass
