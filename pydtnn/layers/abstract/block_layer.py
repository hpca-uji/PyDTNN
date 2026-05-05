import logging

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = ("AbstractBlockLayer",)

logger = logging.getLogger(__name__)


class AbstractBlockLayer[T: Array](Layer[T]):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.paths = []
        for path in args:
            self.paths.append(path)
        self.is_block_layer = True
        self.out_shapes: list[ArrayShape] = []

    def _model_init(self, prev_shape, x):
        super()._model_init(prev_shape, x)
        self.initialize_block_layer()

        temp_memory_size = []
        for p in self.paths:
            for layer in p:
                self.memory_used += layer.memory_used
                temp_memory_size.append(layer.tmp_memory_used)
        self.tmp_memory_used += self.model.memory_cls._total(*temp_memory_size)

    def initialize_block_layer(self):
        for p_i, p in enumerate(self.paths):
            prev_shape = self.prev_shape
            x = self.x
            for i, layer in enumerate(p):
                layer._init_backend_with_model(self.model)
                layer.parent_layer = self
                layer._model_init(prev_shape, x)
                x = layer.y
                if p_i == 0 and (len(p) - 1) == i:
                    self.y = x
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)
        self.shape = self.out_shapes[0]

    def update_weights(self, optimizer):
        for p in self.paths:
            for layer in p:
                layer.update_weights(optimizer)

    def reduce_weights_async(self, gradient=True):
        for p in self.paths:
            for layer in p:
                layer.reduce_weights_async(gradient=gradient)

    def wait_allreduce_async(self, gradient=True):
        for p in self.paths:
            for layer in p:
                layer.wait_allreduce_async(gradient=gradient)

    def reduce_weights_sync(self, gradient=True):
        for p in self.paths:
            for layer in p:
                layer.reduce_weights_sync(gradient=gradient)

    def print_in_convdirect_format(self):
        for p in self.paths:
            for layer in p:
                layer.print_in_convdirect_format()
