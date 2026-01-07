from pydtnn.backends.gpu.layers.abstract.block_layer import AbstractBlockLayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.layers.fc import FC
from pydtnn.layers.dropout import Dropout
from pydtnn.activations.relu import Relu
from pydtnn.layers.feed_forward import FeedForward


class FeedForwardGPU(FeedForward[TensorGPU], AbstractBlockLayerGPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FC_1 = FC(shape=(self.d_ff,), use_bias=False)
        self.relu = Relu()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.FC_2 = FC(shape=(self.shape[-1],), use_bias=False)
        self.paths = [[self.FC_1, self.relu, self.dropout, self.FC_2]]

        # The next attributes will be initialized later
        self.y = self.dx = None

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        self.shape = prev_shape

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

    # Need to flatten and unflatten after the operation in order to maintain the shape it recieves from pre and post layers
    def forward(self, x):
        self.FC_1.forward(x)
        self.relu.forward(self.FC_1.y)
        self.dropout.forward(self.relu.y)
        self.FC_2.forward(self.dropout.y)
        return self.y

    def backward(self, dy):
        self.FC_2.backward(dy)
        self.dropout.backward(self.FC_2.dx)
        self.relu.backward(self.dropout.dx)
        self.FC_1.backward(self.relu.dx)
        return self.dx
