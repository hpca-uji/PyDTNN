from abc import ABC

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers.abstract_block_layer import AbstractBlockLayer


class AbstractBlockLayerCPU(LayerCPU, AbstractBlockLayer, ABC):

    def initialize_block_layer(self) -> None:
        for p in self.paths:
            prev_shape = self.prev_shape
            for layer in p:
                layer.set_model(self.model)
                layer.initialize(prev_shape)
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)
