import numpy as np

from pydtnn.backends.cpu.layers.abstract.block_layer import AbstractBlockLayerCPU
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.activations.relu import Relu
from pydtnn.layers.feed_forward import FeedForward


class FeedForwardCPU(FeedForward[np.ndarray], AbstractBlockLayerCPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FC_1 = FC(shape=(self.d_ff,))
        self.relu = Relu()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.FC_2 = FC(shape=(self.shape[-1],))
        self.paths = [[self.FC_1, self.relu, self.dropout, self.FC_2]]

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)

        # Initialize all sublayers
        for layer in self.children:
            layer.init_backend_from_model(self.model)

        self.FC_1.initialize(prev_shape=(self.shape[-1],), x=x)
        self.relu.initialize(prev_shape=(self.d_ff,), x=self.FC_1.y)
        self.dropout.initialize(prev_shape=(self.d_ff,), x=self.relu.y)
        self.FC_2.initialize(prev_shape=(self.d_ff,), x=self.dropout.y)

        self.y = self.FC_2.y
        self.dx = self.FC_1.dx

        for layer in self.children:
            self.fwd_time += layer.fwd_time
            self.bwd_time += layer.bwd_time
            self.nparams += layer.nparams

    def initialize_block_layer(self):
        pass

    def forward(self, x):
        x = self.FC_1.forward(x)
        x = self.relu.forward(x)
        x = self.dropout.forward(x)
        x = self.FC_2.forward(x)
        return x

    def backward(self, dy):
        dx = self.FC_2.backward(dy)
        dx = self.dropout.backward(dx)
        dx = self.relu.backward(dx)
        dx = self.FC_1.backward(dx)
        return dx
