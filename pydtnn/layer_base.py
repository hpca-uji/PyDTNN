from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pydtnn.model import Model
    from pydtnn.activations.activation import Activation
    from pydtnn.optimizers.optimizer import Optimizer

from pydtnn import utils
from pydtnn.utils.constants import Array, ArrayShape, Parameters
from pydtnn.backends import PromoteToBackend

try:
    from pycuda.driver import Stream  # type: ignore
except Exception:
    pass


class LayerBase[T: Array](PromoteToBackend):
    def __init__(self, shape: ArrayShape = ()) -> None:
        self.nparams: int = 0
        self.shape: ArrayShape = shape
        self.x: T = None  # type: ignore
        self.y: T = None  # type: ignore
        self.weights: T = None  # type: ignore
        self.biases: T = None  # type: ignore
        self.act: type[Activation] | None = None
        self.grad_vars: dict[str, str] = {}
        self.fwd_time: np.ndarray = None  # type: ignore
        self.bwd_time: np.ndarray = None  # type: ignore
        self.paths: list[list[LayerBase[T]]] = []
        self.reqs_allred = {}
        self.parent_layer: LayerBase | None = None

        # The following attributes will be initialized later
        self.id: int = None  # type: ignore
        self.model: Model = None  # type: ignore
        self.prev_shape: ArrayShape = None  # type: ignore
        self.stream_2: Stream = None  # type: ignore
        self.is_block_layer: bool = False

    @property
    def canonical_name(self) -> str:
        frontend = getattr(self, "_frontend", self)
        return type(frontend).__name__

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def name_with_id(self) -> str:
        return f"{self._id_prefix}{self.name}"

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

    def _show_props(self) -> dict:
        props = {}

        props["id"] = self.id

        paths = []
        curr = self
        while curr.parent_layer is not None:
            for i, path in enumerate(curr.parent_layer.paths):
                for layer in path:
                    if layer.id == curr.id:
                        paths.insert(0, i)
            curr = curr.parent_layer
        if paths:
            props["path"] = ",".join(map(str, paths))

        props["name"] = self.name

        if self.nparams > 0:
            props["params"] = self.nparams
            props["memory"] = utils.convert_size_bytes(self.nparams * self.model.dtype.itemsize)

        if self.prev_shape:
            props["input"] = self.prev_shape

        props["output"] = self.shape

        if len(self.paths) > 0:
            props["paths"] = ", ".join(
                f"{path[0].id}-{path[-1].id}" if path else "Empty"
                for path in self.paths
            )

        if self.weights is not None:
            props["weights"] = self.weights.shape

        return props

    def __repr__(self) -> str:
        props = self._show_props()
        name = props.pop("name")

        props = " ".join(
            f"{key}={value!r}"
            for key, value in props.items()
        )

        return f"<{name} {props}>"

    def initialize(self, prev_shape: ArrayShape, x: T | None = None) -> None:
        self.id = next(self.model.layer_id_generator)
        self.prev_shape = prev_shape
        self.x = x  # type:ignore (If it's used, it will be type "T"; if not, it will never be accesed)
        self.fwd_time = np.zeros((4,), dtype=np.float32)
        self.bwd_time = np.zeros((4,), dtype=np.float32)

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

    def print_in_convdirect_format(self):
        pass

    @property
    def children(self) -> list[LayerBase[T]]:
        children: list[LayerBase[T]] = []
        for path in self.paths:
            children += [layer for layer in path]
        return children

    def update_weights(self, optimizer: Optimizer) -> None:
        optimizer.update(self)

    def _export_prop(self, key: str):
        match key:
            case Parameters.PATHS:
                return [
                    [layer.export() for layer in path]
                    for path in self.paths
                ]

            case _:
                return getattr(self, key, None)

    def _import_prop(self, key: str, value) -> None:
        match key:
            case Parameters.PATHS:
                for layer_path, data_path in zip(self.paths, value):
                    for layer, layer_data in zip(layer_path, data_path):
                        layer.import_(layer_data)

            case _:
                setattr(self, key, value)

    def export(self) -> dict[str, Any]:
        data = {}

        data[Parameters.CANONICAL_NAME] = self._export_prop(Parameters.CANONICAL_NAME)

        for key, value in self.grad_vars.items():
            data[key] = self._export_prop(key)
            data[value] = self._export_prop(value)

        if self.paths:
            data[Parameters.PATHS] = self._export_prop(Parameters.PATHS)

        return data

    def import_(self, data: dict[str, Any]) -> None:
        if data[Parameters.CANONICAL_NAME] != self.canonical_name:
            raise TypeError(f"self type must be the same as the stored data type  (self: {self.canonical_name}, stored: {data[Parameters.CANONICAL_NAME]})")

        for key, value in self.grad_vars.items():
            self._import_prop(key, data[key])
            self._import_prop(value, data[value])

        if Parameters.PATHS in data:
            self._import_prop(Parameters.PATHS, data[Parameters.PATHS])
    # -----


class FusedLayerMixIn[T: Array]():
    def __init__(self, *args, **kwargs):
        from_parent = kwargs.pop("from_parent", None)
        if from_parent is None:
            super().__init__(*args, **kwargs)
        else:
            self.__dict__.update(from_parent)
