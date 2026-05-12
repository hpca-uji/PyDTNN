
from __future__ import annotations

"""
Module providing the base Layerable class for defining neural network layers.
"""

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
    """
    Abstract base class for all neural network layers in the PyDTNN framework.

    This class defines the common interface and attributes for layers,
    including shape, parameters, forward/backward pass methods, and
    mechanisms for model integration and state export/import.
    """
    def __init__(self, shape: ArrayShape = ()) -> None:
        """
        Initialize the layer with an optional output shape.

        Args:
            shape: The expected output shape of the layer. Defaults to an empty tuple.
        """
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
        """
        Return the layer name prefixed with its unique identifier.

        The prefix is formatted based on the layer's ID and the total number
        of digits required to represent the highest layer ID in the model,
        ensuring consistent alignment.
        """
        return f"{self._id_prefix}{self.name}"

    @property
    def _id_prefix(self) -> str:
        """
        Generate a numeric prefix for the layer ID based on model depth.

        This method calculates the necessary width for zero-padding the layer's ID
        based on the maximum ID found in the model's layers, ensuring that
        layer IDs are displayed with a consistent number of digits.
        """
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
        """
        Return a dictionary of layer properties for inspection.

        This method aggregates properties from the base class and layer-specific
        attributes like ID, path, parameter count, input/output shapes, and
        weight shapes, providing a comprehensive overview of the layer's state.
        """
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
        """
        Initialize layer state within the model context.

        This method is called by the model during its initialization phase.
        It assigns a unique ID to the layer, records the input shape,
        and initializes timing arrays.

        Args:
            prev_shape: The shape of the input data to this layer.
            x: Optional input tensor. If provided, it's stored for potential use.
        """
        super()._model_init()
        self.id = next(self.model.layer_id_generator)
        self.prev_shape = prev_shape
        self.x = x  # type:ignore (If it's used, it will be type "T"; if not, it will never be accesed)
        self.fwd_time = np.zeros((4,), dtype=np.float32)
        self.bwd_time = np.zeros((4,), dtype=np.float32)

    def forward(self, x: T) -> T:
        """
        Perform the forward pass of the layer.

        This is a placeholder method that should be overridden by concrete
        layer implementations. By default, it returns the input tensor unchanged.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after the layer's transformation.
        """
        return x

    def backward(self, dy: T) -> T:
        """
        Perform the backward pass of the layer.

        This is a placeholder method that should be overridden by concrete
        layer implementations. By default, it returns the input gradient unchanged.

        Args:
            dy: Gradient of the loss with respect to the output of this layer.

        Returns:
            Gradient of the loss with respect to the input of this layer.
        """
        return dy

    def reduce_weights_async(self, gradient: bool = True):
        """
        Initiate asynchronous weight reduction.

        This method is intended for distributed training scenarios to
        asynchronously reduce weights or gradients across multiple workers.
        It should be implemented by subclasses that require distributed
        communication.

        Args:
            gradient: If True, reduce gradients; otherwise, reduce weights.
        """
        pass

    def wait_allreduce_async(self, gradient: bool = True):
        """
        Wait for completion of asynchronous weight reduction.

        This method is used to synchronize after initiating asynchronous
        weight reduction operations, ensuring that all necessary data has
        been communicated and aggregated.

        Args:
            gradient: If True, wait for gradients; otherwise, wait for weights.
        """
        pass

    def reduce_weights_sync(self, gradient: bool = True):
        """
        Perform synchronous weight reduction.

        This method is intended for distributed training scenarios to
        synchronously reduce weights or gradients across multiple workers.
        It should be implemented by subclasses that require distributed
        communication.

        Args:
            gradient: If True, reduce gradients; otherwise, reduce weights.
        """
        pass

    def print_in_convdirect_format(self):
        """
        Print layer configuration in convdirect format.

        This method is a placeholder for generating a specific output format,
        likely for compatibility or debugging with other tools.
        """
        pass

    @property
    def children(self) -> list[Layerable[T]]:
        """
        Return a list of all child layers.

        Child layers are defined by the paths associated with this layer.
        This property aggregates all layers found within these paths.
        """
        children: list[Layerable[T]] = []
        for path in self.paths:
            children += [layer for layer in path]
        return children

    def update_weights(self, optimizer: Optimizer) -> None:
        """
        Update layer weights using the provided optimizer.

        This method delegates the weight update process to a specified optimizer,
        allowing for flexible optimization strategies.

        Args:
            optimizer: The optimizer instance to use for updating the layer's weights.
        """
        optimizer.update(self)

    def _export_prop(self, key: str):
        """
        Retrieve a property value for export.

        This method is used internally to fetch specific layer properties
        that are to be serialized, such as paths or other attributes.

        Args:
            key: The property key to retrieve.

        Returns:
            The value of the requested property.
        """
        match key:
            case Parameters.PATHS:
                return [[layer.export() for layer in path] for path in self.paths]

            case _:
                return getattr(self, key, None)

    def _import_prop(self, key: str, value) -> None:
        """
        Set a property value from imported data.

        This method is used internally to set specific layer properties
        from deserialized data, such as reconstructing paths or other attributes.

        Args:
            key: The property key to set.
            value: The value to assign to the property.
        """
        match key:
            case Parameters.PATHS:
                for layer_path, data_path in zip(self.paths, value):
                    for layer, layer_data in zip(layer_path, data_path):
                        layer.import_(layer_data)

            case _:
                setattr(self, key, value)

    def export(self) -> dict[str, Any]:
        """
        Export layer state to a dictionary.

        This method serializes the layer's essential state, including its
        canonical name, trainable parameters (weights and gradients if applicable),
        and structural information like paths, into a dictionary format.

        Returns:
            A dictionary containing the layer's serializable state.
        """
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
        """
        Import layer state from a dictionary.

        This method deserializes the layer's state from a dictionary,
        reconstructing its properties and parameters. It performs a type check
        to ensure compatibility with the current layer instance.

        Args:
            data: Dictionary containing the state to import.

        Raises:
            TypeError: If the canonical name in the data does not match the
                       layer's canonical name.
        """
        if data[Parameters.CANONICAL_NAME] != self.canonical_name:
            raise TypeError(f"self type must be the same as the stored data type  (self: {self.canonical_name}, stored: {data[Parameters.CANONICAL_NAME]})")

        for var, dvar in self.grad_vars.items():
            self._import_prop(var, data[var])
            if not self.model.evaluate_only:
                self._import_prop(dvar, data[dvar])

        if Parameters.PATHS in data:
            self._import_prop(Parameters.PATHS, data[Parameters.PATHS])
