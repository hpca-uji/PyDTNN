from abc import ABC, abstractmethod

from pydtnn.layers.layer import Layer


class AbstractBlockLayer(Layer, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.paths = []
        for p in args:
            self.paths.append(p)
        self.is_block_layer = True
        self.out_shapes: list[tuple[int, ...]] = []

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.initialize_block_layer()

    @abstractmethod
    def initialize_block_layer(self):
        pass

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
