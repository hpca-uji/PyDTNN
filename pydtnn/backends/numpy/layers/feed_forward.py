import logging
from typing import TYPE_CHECKING

from pydtnn.activations.relu import Relu
from pydtnn.backends.numpy.layers.abstract.block_layer import \
    AbstractBlockLayerNumpy
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.libs import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class FeedForwardNumpy(FeedForward[np.ndarray], AbstractBlockLayerNumpy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FC_1 = FC(shape=(self.d_ff,))
        self.relu = Relu()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.FC_2 = FC(shape=(self.shape[-1],))
        self.paths = [[self.FC_1, self.relu, self.dropout, self.FC_2]]

    def _model_init(self, prev_shape, x):
        super()._model_init(prev_shape, x)

        # Initialize all sublayers
        for layer in self.children:
            layer._init_backend_with_model(self.model)

        self.FC_1._model_init(prev_shape=(self.shape[-1],), x=x)
        self.relu._model_init(prev_shape=(self.d_ff,), x=self.FC_1.y)
        self.dropout._model_init(prev_shape=(self.d_ff,), x=self.relu.y)
        self.FC_2._model_init(prev_shape=(self.d_ff,), x=self.dropout.y)

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
