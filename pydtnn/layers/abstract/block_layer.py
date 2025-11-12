from copy import deepcopy
from typing import Self
from pydtnn.layers.layer import Layer
from pydtnn.utils.types import Array

class AbstractBlockLayer[T: Array](Layer[T]):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.paths = []
        for p in args:
            self.paths.append(p)
        self.is_block_layer = True
        self.out_shapes: list[tuple[int, ...]] = []

    def initialize(self, prev_shape, x=None):
        super().initialize(prev_shape, x)
        self.initialize_block_layer()

    def initialize_block_layer(self):
        for p_i, p in enumerate(self.paths):
            prev_shape = self.prev_shape
            x = self.x
            for i, layer in enumerate(p):
                layer.set_backend(self.model._backend)
                layer.set_model(self.model)
                layer.initialize(prev_shape, x)
                x = layer.y
                if p_i == 0 and (len(p) - 1) == i:
                    self.y = x
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)
        self.shape = self.out_shapes[0]
    
    def copy_from(self, other: Self) -> None:
        super().copy_from(other)
        
        num_paths = len(self.paths)
        assert num_paths == len(other.paths), f"Both layers must have the same number of paths (self: {num_paths}, other: {len(other.paths)})"
        for p in range(num_paths):
            path = self.paths[p]
            other_path = other.paths[p]

            num_layers = len(path)

            assert num_layers == len(other_path), f"Both paths must have the same number of layers (self: {num_layers}, other: {len(other_path)})"
            for l in range(num_layers):
                layer = path[l]
                other_layer = other_path[l]
                layer.copy_from(other_layer)

        self.out_shapes = deepcopy(other.out_shapes)
     # ----


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
