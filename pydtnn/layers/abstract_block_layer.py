from abc import ABC

from pydtnn.layers.layer import Layer


class AbstractBlockLayer(Layer, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.paths = []
        for p in args:
            self.paths.append(p)
        self.is_block_layer = True
        self.out_shapes: list[tuple[int, ...]] = []

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self.initialize_block_layer()

    def initialize_block_layer(self):
        for p in self.paths:
            prev_shape = self.prev_shape
            x = self.x
            for layer in p:
                layer.set_model(self.model)
                layer.initialize(prev_shape, x)
                x = layer.y
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)

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
