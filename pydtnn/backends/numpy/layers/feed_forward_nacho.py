"""Numpy backend implementation of the FeedForward layer."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.activations.relu import Relu
from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.feed_forward_nacho import FeedForwardNacho
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("FeedForwardNumpyNacho",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class FeedForwardNumpyNacho(FeedForwardNacho[np.ndarray], AbstractBlockLayerNumpy):
    """Numpy-based FeedForward layer implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the FeedForwardNumpy layer with sublayers."""
        super().__init__(*args, **kwargs)
        self.FC_1 = FC(shape=(self.d_ff,))
        self.relu = Relu()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.FC_2 = FC(shape=(self.shape[-1],))
        self.paths = [[self.FC_1, self.relu, self.dropout, self.FC_2]]

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initializes sublayers and model parameters."""
        super()._model_init(prev_shape, x)

        # Initialize all sublayers
        for layer in self.children:
            layer._init_backend_with_model(self.model)

        # breakpoint()
        self.FC_1._model_init(prev_shape=(self.shape[-1],), x=x)
        self.relu._model_init(prev_shape=(self.d_ff,), x=self.FC_1.y)
        self.dropout._model_init(prev_shape=(self.d_ff,), x=self.relu.y)
        self.FC_2._model_init(prev_shape=(self.d_ff,), x=self.dropout.y)
        # breakpoint()

        # NOTE: It should be like that.
        self.y = self.FC_2.y
        self.dx = self.FC_1.dx

        # NOTE: In initialize_block_layer:
        for layer in self.children:
            self.fwd_time += layer.fwd_time
            self.bwd_time += layer.bwd_time
            self.nparams += layer.nparams

    def initialize_block_layer(self) -> None:
        """Initializes the block layer components."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the feed-forward network."""
        # NOTE: It should be only one path.
        for layer in self.paths[0]:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the feed-forward network."""
        # NOTE: It should be only one path.
        dx = dy
        for layer in reversed(self.paths[0]):
            dx = layer.backward(dx)
        return dx
