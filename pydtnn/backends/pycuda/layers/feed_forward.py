"""PyCUDA implementation of the FeedForward layer for the PyDTNN framework."""

import logging
from typing import Any

from pydtnn.activations.relu import Relu
from pydtnn.backends.pycuda.layers.abstract.block_layer import AbstractBlockLayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.feed_forward import FeedForward
from pydtnn.utils.constants import ArrayShape

__all__ = ("FeedForwardPycuda",)

logger = logging.getLogger(__name__)


class FeedForwardPycuda(FeedForward[TensorArray], AbstractBlockLayerPycuda):
    """PyCUDA-accelerated FeedForward layer implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the FeedForwardPycuda layer with sublayers."""
        super().__init__(*args, **kwargs)
        self.FC_1 = FC(shape=(self.d_ff,), use_bias=False)
        self.relu = Relu()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.FC_2 = FC(shape=(self.shape[-1],), use_bias=False)
        self.paths = [[self.FC_1, self.relu, self.dropout, self.FC_2]]

        # The next attributes will be initialized later
        self.y: TensorArray = None  # type: ignore
        self.dx: TensorArray = None  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize model parameters and sublayers for the PyCUDA backend."""
        super()._model_init(prev_shape, x)
        self.shape = prev_shape

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

    def initialize_block_layer(self) -> None:
        """Initialize block-specific configurations."""
        pass

    # Need to flatten and unflatten after the operation in order to maintain
    # the shape it recieves from pre and post layers
    def forward(self, x: TensorArray) -> TensorArray:
        """Perform the forward pass through the feed-forward network."""
        self.FC_1.forward(x)
        self.relu.forward(self.FC_1.y)
        self.dropout.forward(self.relu.y)
        self.FC_2.forward(self.dropout.y)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform the backward pass through the feed-forward network."""
        self.FC_2.backward(dy)
        self.dropout.backward(self.FC_2.dx)
        self.relu.backward(self.dropout.dx)
        self.FC_1.backward(self.relu.dx)
        return self.dx
