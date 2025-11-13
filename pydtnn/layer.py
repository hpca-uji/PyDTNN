from __future__ import annotations

from copy import deepcopy

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
    from pydtnn.activations.activation import Activation
    from pydtnn.optimizers.optimizer import Optimizer

from pydtnn.utils.types import Array
from pydtnn.utils.types import ArrayShape
from pydtnn.backends import PromoteToBackend

try:
    from pycuda.driver import Stream  # type: ignore
except:
    pass


class LayerAndActivationBase[T: Array](PromoteToBackend):
    def __init__(self, shape: ArrayShape = ()) -> None:
        self.nparams: int = 0
        self.shape: ArrayShape = shape
        self.x: T = None  # type: ignore
        self.y: T = None  # type: ignore
        self.weights: T = None  # type: ignore
        self.biases: T | None = None
        self.act: type[Activation] | None = None
        self.grad_vars: dict[str, str] = {}
        self.fwd_time: np.ndarray = np.zeros((4,), dtype=np.float32)
        self.bwd_time: np.ndarray = np.zeros((4,), dtype=np.float32)
        self.paths: list[list[LayerAndActivationBase[T]]] = []
        self.reqs_allred = {}

        # The following attributes will be initialized later
        self.id: int = None  # type: ignore
        self.model: Model = None  # type: ignore
        self.prev_shape: ArrayShape = None  # type: ignore
        self.stream_2: Stream = None  # type: ignore
        self.is_block_layer: bool = False

    @property
    def canonical_name(self) -> str:
        return self.__class__.__name__

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

    def __repr__(self) -> str:
        return f"{self._id_prefix}{type(self).__name__}"

    def set_model(self, parent_model: Model) -> None:
        super().set_model(parent_model)
        self.id = next(self.model.layer_id_generator)

    def initialize(self, prev_shape: ArrayShape, x: T | None = None) -> None:
        self.prev_shape = prev_shape
        self.x = x  # type:ignore (If it's used, it will be type "T"; if not, it will never be accesed)

    def forward(self, x: T) -> T:
        return x

    def backward(self, dy: T) -> T:
        return dy

    def reduce_weights_async(self, gradient: bool = True):
        pass

    def wait_allreduce_async(self, gradient: bool = True):
        pass

    def reduce_weights_sync(self, gradient: bool = True):
        pass

    def show(self, attrs: str | None = "") -> None:
        if not attrs:
            attrs = "|{:19s}|{:^37s}|".format("", "")
        print(f"|{self.id:^7d}|{self.canonical_name:^26s}|{self.nparams:^9d}|{str(self.shape):^15}" + attrs)

    def print_in_convdirect_format(self):
        pass

    @property
    def children(self) -> list[LayerAndActivationBase[T]]:
        children: list[LayerAndActivationBase[T]] = []
        for path in self.paths:
            children += [layer for layer in path]
        return children

    def update_weights(self, optimizer: Optimizer) -> None:
        optimizer.update(self)

    def export(self):
        data = self._export()

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = np.asarray(value, dtype=np.float64, order="C", copy=None)

        return deepcopy(data)

    def import_(self, data) -> None:
        data = deepcopy(data)

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = np.asarray(value, dtype=self.model.dtype, order="C", copy=None)

        self._import(data)

    def _export(self) -> dict:
        data = {}

        if self.x is not None:
            data["x"] = self.x

        if self.y is not None:
            data["y"] = self.y

        if self.weights is not None:
            data["weights"] = self.weights

        if self.biases is not None:
            data["biases"] = self.biases

        for key, value in self.grad_vars.items():
            data[key] = getattr(self, key)
            data[value] = getattr(self, value)

        paths = data["paths"] = []
        for path in self.paths:
            paths.append([layer.export() for layer in path])

        return data

    def _import(self, data) -> None:

        if self.x is not None:
            self.x = data["x"]

        if self.y is not None:
            self.y = data["y"]

        if self.weights is not None:
            self.weights = data["weights"]

        if self.biases is not None:
            self.biases = data["biases"]

        for key, value in self.grad_vars.items():
            setattr(self, key, data[key])
            setattr(self, value, data[value])

        for layer_path, data_path in zip(self.paths, data["paths"]):
            for layer, layer_data in zip(layer_path, data_path):
                layer.import_(layer_data)
    # -----


class FusedLayerMixIn[T: Array]():
    def __init__(self, *args, **kwargs):
        from_parent = kwargs.pop("from_parent", None)
        if from_parent is None:
            super().__init__(*args, **kwargs)
        else:
            from_parent.pop("forward", None)
            self.__dict__.update(from_parent)
