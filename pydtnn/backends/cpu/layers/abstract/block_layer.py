from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
import numpy as np

class AbstractBlockLayerCPU(AbstractBlockLayer[np.ndarray], LayerCPU):
    
    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        for p in self.paths:
            for layer in p:
                self.actual_size += layer.actual_size
