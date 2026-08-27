"""Abstract base class for layers composed of multiple sequential paths."""

import logging
from typing import Any

from pydtnn.layers.abstract.layer import Layer
from pydtnn.optimizers.abstract.optimizer import Optimizer
from pydtnn.utils.constants import Array, ArrayShape, SyncMode

__all__ = ("AbstractBlockLayer",)

logger = logging.getLogger(__name__)


class AbstractBlockLayer[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """Base class for layers that manage multiple parallel execution paths."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the block layer with a collection of paths."""
        super().__init__(**kwargs)
        self.paths = []
        for path in args:
            self.paths.append(path)
        self.is_block_layer = True
        self.out_shapes: list[ArrayShape] = []

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

    def update_weights(
        self, optimizer: Optimizer[T], update: bool = True, sync: bool = True
    ) -> None:
        """Updates weights for all layers in all paths using the provided optimizer."""
        for p in self.paths:
            for layer in p:
                layer.update_weights(optimizer, update, sync)

    def reduce_state_async(self, mode: SyncMode) -> None:
        """Initiates asynchronous weight reduction for all layers."""
        for p in self.paths:
            for layer in p:
                layer.reduce_state_async(mode=mode)

    def reduce_state_wait(self, mode: SyncMode) -> None:
        """Waits for completion of asynchronous weight reductions."""
        for p in self.paths:
            for layer in p:
                layer.reduce_state_wait(mode=mode)

    def reduce_state_sync(self, mode: SyncMode) -> None:
        """Performs synchronous weight reduction for all layers."""
        for p in self.paths:
            for layer in p:
                layer.reduce_state_sync(mode=mode)

    def print_in_convdirect_format(self) -> None:
        """Prints the layer configuration in convdirect format."""
        for p in self.paths:
            for layer in p:
                layer.print_in_convdirect_format()
