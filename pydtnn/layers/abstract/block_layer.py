"""Abstract base class for layers composed of multiple sequential paths."""

import logging
from typing import Any

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.abstract.layer import Layer
from pydtnn.optimizers.abstract.optimizer import Optimizer
from pydtnn.utils.constants import Array, ArrayShape, SyncMode

__all__ = ("AbstractBlockLayer",)

logger = logging.getLogger(__name__)


class AbstractBlockLayer[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """Base class for layers that manage multiple parallel execution paths."""

    def __init__(self, paths: list[list[Layerable[T]]], multiple_inputs: bool = False, **kwargs: Any) -> None:
        """Initializes the block layer with a collection of paths.

        Parameters:
            paths (list[list[Layerable[T]]]): diferent list of layers of the block.

            multiple_inputs (bool):
                - False: the input's layer will be replicated into all the path's first layer.
                         The block layer's "prev_shape" tensor format will be the usual one (self.model.tensor_format).
                - True: every path has it's own input. The block layer's "prev_shape" tensor format will have
                        an extra dimension: [Num. Paths]self.model.tensor_format.
                        (e.g.: prev. layer's output shape: "(32, 3, 20, 20)" and len(self.paths)=3, then
                        "self.prev_shape=(3, 32, 3, 20, 20)")).
        """
        super().__init__(**kwargs)
        self.paths: list[list[Layerable[T]]] = []
        for path in paths:
            self.paths.append(path)
        self.is_block_layer = True
        self.num_inputs = len(self.paths) if multiple_inputs else 1
        self.out_shapes: list[ArrayShape] = []
        self.dx: T = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: T) -> None:
        """Initializes the model structure and calculates memory requirements."""
        super()._model_init(prev_shape, x)
        self.initialize_block_layer()

        temp_memory_size = []
        for p in self.paths:
            for layer in p:
                self.memory_used += layer.memory_used
                temp_memory_size.append(layer.tmp_memory_used)
        self.tmp_memory_used += self.model.memory_cls._total(*temp_memory_size)

    def initialize_block_layer(self) -> None:
        """Initializes all layers within the block paths and sets output shapes."""

        for p_i, p in enumerate(self.paths):
            if self.num_inputs > 1:
                prev_shape = self.prev_shape[p_i]
                x = self.x[p_i]
            else:
                prev_shape = self.prev_shape
                x = self.x

            for i, layer in enumerate(p):
                layer._init_backend_with_model(self.model)
                layer.parent_layer = self
                layer._model_init(prev_shape, x)  # pyright: ignore[reportArgumentType]
                x = layer.y
                if p_i == 0 and (len(p) - 1) == i:
                    self.y = x
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)  # pyright: ignore[reportArgumentType]
        self.shape = self.out_shapes[0]

        if self.num_inputs > 1:
            self.prev_shape = (self.num_inputs, *self.prev_shape)

    def forward(self, x: T) -> T:
        # NOTE: The following code only make sense if there is only one path (ie.: a "Sequence" of Layerables).
        for path in self.paths:
            y = x
            for layer in path:
                y = layer.forward(y)
        return y  # pyright: ignore[reportPossiblyUnboundVariable]

    def backward(self, dy: T) -> T:
        # NOTE: The following code only make sense if there is only one path (ie.: a "Sequence" of Layerables).
        for path in self.paths:
            dx = dy
            for layer in reversed(path):
                dx = layer.backward(dx)
        return dx  # pyright: ignore[reportPossiblyUnboundVariable]

    def update_weights(
        self, optimizer: Optimizer[T], update: bool = True, sync: bool = True
    ) -> None:
        """Updates weights for all layers in all paths using the provided optimizer."""
        for p in self.paths:
            for layer in p:
                layer.update_weights(optimizer, update, sync)

    def state_reduce_async(self, mode: SyncMode) -> None:
        """Initiates asynchronous weight reduction for all layers."""
        for p in self.paths:
            for layer in p:
                layer.state_reduce_async(mode=mode)

    def state_reduce_wait(self, mode: SyncMode) -> None:
        """Waits for completion of asynchronous weight reductions."""
        for p in self.paths:
            for layer in p:
                layer.state_reduce_wait(mode=mode)

    def state_reduce_sync(self, mode: SyncMode) -> None:
        """Performs synchronous weight reduction for all layers."""
        for p in self.paths:
            for layer in p:
                layer.state_reduce_sync(mode=mode)

    def print_in_convdirect_format(self) -> None:
        """Prints the layer configuration in convdirect format."""
        for p in self.paths:
            for layer in p:
                layer.print_in_convdirect_format()
