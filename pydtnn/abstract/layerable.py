from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from pydtnn.abstract.base import Base
from pydtnn.utils.constants import Array, ArrayShape, Parameters

__all__ = ("Layerable",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation
    from pydtnn.model import Model
    from pydtnn.optimizers.optimizer import Optimizer


try:
    from pycuda.driver import Stream  # type: ignore
except Exception:
    pass


class Layerable[T: Array](Base[T]):
    def __init__(self, shape: ArrayShape = ()) -> None:
        super().__init__()
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
        self.paths: list[list[Layerable[T]]] = []
        self.reqs_allred = {}
        self.parent_layer: Layerable | None = None

        # The following attributes will be initialized later
        self.id: int = None  # type: ignore
        self.model: Model = None  # type: ignore
        self.prev_shape: ArrayShape = None  # type: ignore
        self.stream_2: Stream = None  # type: ignore
        self.is_block_layer: bool = False

    @property
    def name_with_id(self) -> str:
        return f"{self._id_prefix}{self.name}"

    @property
    def _id_prefix(self) -> str:
        prefix = ""
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

        props.update(super()._show_props())

        if self.nparams > 0:
            props["params"] = self.nparams

        if self.prev_shape:
            props["input"] = self.prev_shape

        props["output"] = self.shape

        if len(self.paths) > 0:
            props["paths"] = ", ".join(f"{path[0].id}-{path[-1].id}" if path else "Empty" for path in self.paths)

        if self.weights is not None:
            props["weights"] = self.weights.shape

        return props

    def _model_init(self, prev_shape: ArrayShape, x: T | None = None) -> None:
        super()._model_init()
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
    def children(self) -> list[Layerable[T]]:
        children: list[Layerable[T]] = []
        for path in self.paths:
            children += [layer for layer in path]
        return children

    def update_weights(self, optimizer: Optimizer) -> None:
        optimizer.update(self)

    def _export_prop(self, key: str):
        match key:
            case Parameters.PATHS:
                return [[layer.export() for layer in path] for path in self.paths]

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

        for var, dvar in self.grad_vars.items():
            data[var] = self._export_prop(var)
            if not self.model.evaluate_only:
                data[dvar] = self._export_prop(dvar)

        if self.paths:
            data[Parameters.PATHS] = self._export_prop(Parameters.PATHS)

        return data

    def import_(self, data: dict[str, Any]) -> None:
        if data[Parameters.CANONICAL_NAME] != self.canonical_name:
            raise TypeError(f"self type must be the same as the stored data type  (self: {self.canonical_name}, stored: {data[Parameters.CANONICAL_NAME]})")

        for var, dvar in self.grad_vars.items():
            self._import_prop(var, data[var])
            if not self.model.evaluate_only:
                self._import_prop(dvar, data[dvar])

        if Parameters.PATHS in data:
            self._import_prop(Parameters.PATHS, data[Parameters.PATHS])
