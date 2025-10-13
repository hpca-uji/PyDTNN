from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from typing import Self, TYPE_CHECKING, TypeVar
if TYPE_CHECKING:
    from pydtnn.model import Model
    from pydtnn.activations.activation import Activation
    from pydtnn.optimizers.optimizer import Optimizer

from pydtnn.utils.types import Array

drv_Stream = TypeVar("pycuda_driver_Stream")  # PyCuda's driver Stream class. The initialization is on GPU's layers classes.

class LayerAndActivationBase[T: Array](ABC):

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.nparams: int = 0
        self.shape: tuple[int, ...] = shape
        self.x: T | None = None
        self.y: T | None = None
        self.weights: T | None = None
        self.biases: T | None = None
        self.act: Activation | None = None
        self.grad_vars: dict[str, str] = {}
        self.fwd_time: np.ndarray = np.zeros((4,), dtype=np.float32)
        self.bwd_time: np.ndarray = np.zeros((4,), dtype=np.float32)
        self.paths: list[list[Self]] = []
        self.reqs_allred = {}
        # The next attributes will be initialized later
        self.id: int = None
        self.model: Model = None
        self.prev_shape: tuple[int, ...] = None
        self.is_block_layer: bool = False
        self.stream_2: drv_Stream = None
    # --- END __init__ --- #

    @property
    def _id_prefix(self) -> str:
        prefix = ''
        if self.id is not None and self.model is not None:
            try:
                model__last_layer = self.model.layers[-1]
            except IndexError:
                max_digits = 1
            else:
                model__last_id = model__last_layer.id
                if len(model__last_layer.children):
                    model__last_id = model__last_layer.children[-1].id
                max_digits = len(str(model__last_id))
            prefix = "{:0{width}d}_".format(self.id, width=max_digits)
        return prefix
    # --- END _id_prefix --- #

    def __repr__(self) -> str:
        return f"{self._id_prefix}{type(self).__name__}"

    def set_model(self, parent_model: Model) -> None:
        self.model = parent_model
        self.id = next(self.model.layer_id)

    def initialize(self, prev_shape: tuple[int, ...], x: T | None = None) -> None:
        self.prev_shape = prev_shape
        self.x = x

    @abstractmethod
    def forward(self, x: T) -> T:
        return x

    @abstractmethod
    def backward(self, dy: T) -> T:
        return dy

    @abstractmethod
    def reduce_weights_async(self, gradient: bool = True):
        pass

    @abstractmethod
    def wait_allreduce_async(self, gradient: bool = True):
        pass

    @abstractmethod
    def reduce_weights_sync(self, gradient: bool = True):
        pass

    def show(self, attrs: str | None = "") -> str:
        if not attrs:
            attrs = "|{:19s}|{:^37s}|".format("", "")
        print(f"|{self.id:^7d}|{type(self).__name__:^26s}|{self.nparams:^9d}|{str(self.shape):^15}" + attrs)

    def print_in_convdirect_format(self):
        pass

    @property
    def children(self) -> list[Self]:
        children: list = []
        for path in self.paths:
            children += [layer for layer in path]
        return children

    def update_weights(self, optimizer: Optimizer) -> None:
        optimizer.update(self)
