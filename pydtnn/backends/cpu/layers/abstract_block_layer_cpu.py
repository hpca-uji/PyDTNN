from abc import ABC

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers.abstract_block_layer import AbstractBlockLayer
from pydtnn.layers.layer import LayerError

class AbstractBlockLayerCPU(LayerCPU, AbstractBlockLayer, ABC):
    pass